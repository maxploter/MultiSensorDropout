import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Precision, Recall, F1Score

from models.matcher import build_matcher
from util import box_ops
from util.misc import sigmoid_focal_loss, accuracy, is_multi_head_fn, get_world_size


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, focal_alpha, focal_gamma, weight_dict, focal_loss):
        super(SetCriterion, self).__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.weight_dict = weight_dict
        self.focal_loss = focal_loss

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = 0.1 # self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_focal(self, outputs, targets, indices, num_objects, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs

        # [batch_size, number_queries, number_of_classes]
        src_logits = outputs['pred_logits']

        # (batch_ids, output_query_ids)
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        device = next(iter(outputs.values())).device
        # [batch_size, number_queries]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=device)

        target_classes[idx] = target_classes_o

        # [batch_size, number_queries, number_of_classes+1]
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=device)

        #
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # [batch_size, number_queries, number_of_classes]
        target_classes_onehot = target_classes_onehot[:,:,:-1]

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_objects,
            alpha=self.focal_alpha, gamma=self.focal_gamma)

        loss_ce *= src_logits.shape[1] # Why?
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_center_points(self, outputs, targets, indices, num_objects, log=True):
        """L1 center point loss
        targets dicts must contain the key "center_points" containing a tensor of dim [nb_target_boxes, 2]
        """
        assert 'pred_center_points' in outputs

        idx = self._get_src_permutation_idx(indices)
        src_cps = outputs['pred_center_points'][idx]
        target_cps = torch.cat([t['center_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_cp = F.l1_loss(src_cps, target_cps, reduction='none')

        losses = {}
        losses['loss_center_point'] = loss_cp.sum() / num_objects

        return losses

    def forward(self, outputs, targets):
        indecies = self.matcher(outputs, targets)

        num_objects = sum(len(t["labels"]) for t in targets)
        num_objects = torch.as_tensor(
            [num_objects], dtype=torch.float, device=next(iter(outputs.values())).device)

        losses = {}

        if self.focal_loss:
            loss_labels = self.loss_labels_focal(outputs, targets, indecies, num_objects)
        else:
            loss_labels = self.loss_labels(outputs, targets, indecies, num_objects)
        loss_center_points = self.loss_center_points(outputs, targets, indecies, num_objects)

        losses.update(loss_labels)
        losses.update(loss_center_points)
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


class MultiHeadSetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, losses, focal_alpha, focal_gamma, weight_dict, focal_loss, multi_classification_heads=False):
        super(SetCriterion, self).__init__()
        self.num_classes = num_classes
        self.coord_criterion = nn.MSELoss(reduction='sum')
        self.class_criterion = nn.CrossEntropyLoss(reduction='sum')
        self.matcher = matcher
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.weight_dict = weight_dict
        self.focal_loss = focal_loss
        self.losses = losses
        self.multi_classification_heads = multi_classification_heads

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = 0.1 # self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_binary_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)

        if not all(t['center_points'].numel() == 0 for t in targets):
            # assign GT label
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([
                torch.full(J.shape, 1, dtype=torch.int64, device=src_logits.device)
                for t, (_, J) in zip(targets, indices)])

            target_classes[idx] = target_classes_o

        target_classes = target_classes.float()  # Convert target_classes to Float

        negative_weight = 0.1
        positive_weight = 1.0

        # Create a weight tensor. It needs to match the shape of the targets
        weights = torch.ones_like(target_classes)

        weights[target_classes == 0.0] = negative_weight  # Set weight for negative class (0)
        weights[target_classes == 1.0] = positive_weight  # Set weight for positive class (1)

        loss_bce = F.binary_cross_entropy_with_logits(
            src_logits, target_classes,
            weight=weights,
            reduction='sum'
            # pos_weight=torch.tensor([9], device=src_logits.device)
        )

        losses = {
            'loss_bce': loss_bce,
            'total_effective_samples': weights.sum()
        }

        if log:
            # when have targets
            probabilities = torch.sigmoid(src_logits)

            for t in [0.5, 0.75, 0.95]:
                p = Precision(task='binary', threshold=t).to(probabilities.device)(probabilities, target_classes)
                r = Recall(task='binary', threshold=t).to(probabilities.device)(probabilities, target_classes)
                f1 = F1Score(task='binary', threshold=t).to(probabilities.device)(probabilities, target_classes)

                losses.update({
                    f'binary_precision_{t}': p.detach().cpu(),
                    f'binary_recall_{t}': r.detach().cpu(),
                    f'binary_f1_{t}': f1.detach().cpu(),
                })

        return losses


    def loss_center_points(self, outputs, targets, indices, num_objects, log=True):
        """L1 center point loss
        targets dicts must contain the key "center_points" containing a tensor of dim [nb_target_boxes, 2]
        """
        assert 'pred_center_points' in outputs

        if all(t['center_points'].numel() == 0 for t in targets):
            # no GT for the head
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_cps = outputs['pred_center_points'][idx]
        target_cps = torch.cat([t['center_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_cp = F.l1_loss(src_cps, target_cps, reduction='none')

        losses = {}
        losses['loss_center_point'] = loss_cp.sum() / num_objects

        return losses

    def forward(self, outputs, targets):
        indices_per_head = self.matcher(outputs, targets)

        losses = {}

        num_objects = sum(len(t["labels"]) for t in targets)

        # get first value form the dict outputs
        head_output = outputs[next(iter(outputs))]

        num_objects = torch.as_tensor(
            [num_objects], dtype=torch.float, device=next(iter(head_output.values())).device)

        is_multi_head = is_multi_head_fn(outputs.keys())

        # get first value from head_output
        head_output_device = next(iter(head_output.values())).device

        number_of_type_of_objects = torch.unique(torch.cat([t["labels"] for t in targets])).numel()

        for head_id, indices in indices_per_head.items():
            head_output = outputs[head_id]

            if is_multi_head:
                tgt_mask = [[t["labels"] == int(head_id)] for t in targets]

                head_targets = [
                    {
                        "center_points": t["center_points"][mask].to(head_output_device),
                        "labels": t["labels"][mask].to(head_output_device),
                    }
                    for t, mask in zip(targets, tgt_mask)
                ]
            else:
                head_targets = targets

            num_objects_stub = torch.as_tensor(
                [1], dtype=torch.float, device=next(iter(head_output.values())).device)

            losses[head_id] = {}
            for loss in self.losses:
                losses[head_id].update(self.get_loss(loss, head_output, head_targets, indices, num_objects_stub))

        losses = {k: sum(v[k] for v in losses.values() if k in v) for k in set(k for v in losses.values() for k in v.keys())}

        if 'loss_center_point' in losses:
            losses['loss_center_point'] /= num_objects
        if 'loss_bce' in losses:
            losses['loss_bce'] /= losses['total_effective_samples']

        losses_result = {}
        for k, v in losses.items():
            if k.startswith('binary_'):
                losses_result[k] = v / number_of_type_of_objects
            else:
                losses_result[k] = v

        return losses_result

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels_focal if self.focal_loss else self.loss_labels,
            'center_points': self.loss_center_points,
            'binary_labels': self.loss_binary_labels
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


class BoxSetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 focal_loss, focal_alpha, focal_gamma):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their
                         relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of
                    available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        # self.tracking = tracking
        # self.track_query_false_positive_eos_weight = track_query_false_positive_eos_weight

    def loss_labels(self, outputs, targets, indices, _, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2),
                                  target_classes,
                                  weight=self.empty_weight,
                                  reduction='none')

        if self.tracking and self.track_query_false_positive_eos_weight:
            for i, target in enumerate(targets):
                if 'track_query_boxes' in target:
                    # remove no-object weighting for false track_queries
                    loss_ce[i, target['track_queries_fal_pos_mask']] *= 1 / self.eos_coef
                    # assign false track_queries to some object class for the final weighting
                    target_classes = target_classes.clone()
                    target_classes[i, target['track_queries_fal_pos_mask']] = 0

        # weight = None
        # if self.tracking:
        #     weight = torch.stack([~t['track_queries_placeholder_mask'] for t in targets]).float()
        #     loss_ce *= weight

        loss_ce = loss_ce.sum() / self.empty_weight[target_classes].sum()

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes,
            alpha=self.focal_alpha, gamma=self.focal_gamma)

        loss_ce *= src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of
            predicted non-empty boxes. This is not really a loss, it is intended
            for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss
           and the GIoU loss targets dicts must contain the key "boxes" containing
           a tensor of dim [nb_target_boxes, 4]. The target boxes are expected in
           format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels_focal if self.focal_loss else self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses


def build_criterion(args):
    assert 'moving-mnist' in args.dataset.lower()
    num_classes = 10
    matcher = build_matcher(args)

    if hasattr(args, 'multi_classification_heads') and args.multi_classification_heads:
        criterion = MultiHeadSetCriterion(
            num_classes,
            matcher,
            losses=['center_points', 'binary_labels'],
            focal_alpha=0.25,
            focal_gamma=2,
            weight_dict={
                'loss_bce': args.weight_loss_bce,
                'loss_center_point': args.weight_loss_center_point
            },
            focal_loss=args.focal_loss,
            multi_classification_heads=getattr(args, 'multi_classification_heads', False)
        )
    elif args.object_detection:
        losses = ['labels', 'boxes', 'cardinality']


        weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
        weight_dict['loss_giou'] = args.giou_loss_coef

        criterion = BoxSetCriterion(
            num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=args.eos_coef,
            losses=losses,
            focal_loss=args.focal_loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
        )
    else:
        criterion = SetCriterion(
            num_classes,
            matcher,
            focal_alpha=0.25,
            focal_gamma=2,
            weight_dict={
                'loss_ce': 1,
                'loss_center_point': args.weight_loss_center_point
            },
            focal_loss=args.focal_loss,
        )

    return criterion
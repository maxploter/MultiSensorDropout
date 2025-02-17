import torch
import torch.nn.functional as F
from torch import nn

from models.matcher import build_matcher
from util.misc import sigmoid_focal_loss, accuracy


class SetCriterion(nn.Module):
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
        """Binary classification loss (BCE)
        targets dicts must contain the key "binary_labels" containing a tensor of dim [nb_target_boxes, 2]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([
            torch.full(J.shape, 1, dtype=torch.int64, device=src_logits.device)
            for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes = target_classes.float()  # Convert target_classes to Float
        loss_bce = F.binary_cross_entropy_with_logits(
            src_logits, target_classes, pos_weight=torch.tensor([9], device=src_logits.device))
        losses = {'loss_bce': loss_bce}

        return losses

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
        indices_per_head = self.matcher(outputs, targets)

        losses = {}
        for head_id, indices in indices_per_head.items():
            head_output = outputs[head_id]

            if self.multi_classification_heads:
                num_objects = sum((t["labels"] == int(head_id)).sum().item() for t in targets)
            else:
                num_objects = sum(len(t["labels"]) for t in targets)

            num_objects = torch.as_tensor(
                [num_objects], dtype=torch.float, device=next(iter(head_output.values())).device)

            losses[head_id] = {}
            for loss in self.losses:
                losses[head_id].update(self.get_loss(loss, head_output, targets, indices, num_objects))

        losses = {k: sum(v[k] for v in losses.values() if k in v) for k in set(k for v in losses.values() for k in v.keys())}

        return losses

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


def build_criterion(args):
    assert 'moving-mnist' in args.dataset.lower()
    num_classes = 10

    losses = ['center_points']
    if hasattr(args, 'multi_classification_heads') and args.multi_classification_heads:
        losses += ['binary_labels']
    else:
        losses += ['labels']


    matcher = build_matcher(args)
    return SetCriterion(
        num_classes,
        matcher,
        losses=losses,
        focal_alpha=0.25,
        focal_gamma=2,
        weight_dict={
            'loss_ce': 1,
            'loss_bce': 1,
            'loss_center_point': 5
        },
        focal_loss=args.focal_loss,
        multi_classification_heads=getattr(args, 'multi_classification_heads', False)
    )

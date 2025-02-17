import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, focal_loss, cost_class: float = 1, cost_center_point: float = 1,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0,):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates
                       in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the
                       matching cost
        """
        super().__init__()
        self.cost_class = cost_class

        self.cost_center_point = cost_center_point

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_loss = focal_loss

        assert cost_class != 0 or cost_center_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):

        result_indices = {}

        # Define variable number_of_heads as the number of heads in the outputs
        number_of_heads = len(outputs)

        for head_id, head_outputs in outputs.items():
            batch_size, num_queries = head_outputs["pred_logits"].shape[:2]

            # There're 2 cases for this clss
            # 1. w/o temporal dimention:
            # batch size dimention represents number of independent frames in the batch
            # 2. w/ temporal dimention:
            # batch size dimention represents number of frames in the sequence
            # (we do not support multiple sequences in the batch)

            # We flatten to compute the cost matrices in a batch
            #
            # [batch_size * num_queries, num_classes]
            if number_of_heads > 1:
                # If multiple heads we assume that head contain binary classification
                out_prob = head_outputs["pred_logits"].flatten(0, 1).sigmoid()
            elif self.focal_loss:
                out_prob = head_outputs["pred_logits"].flatten(0, 1).sigmoid()
            else:
                out_prob = head_outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

            # [batch_size * num_queries, 2]
            out_center_points = head_outputs["pred_center_points"].flatten(0, 1)

            if number_of_heads > 1:
                tgt_mask = [[t["labels"] == int(head_id)] for t in targets]
                tgt_center_points = torch.cat([t["center_points"][mask] for t, mask in zip(targets, tgt_mask)])
                # tgt_ids of the format if head_id equal to t label then true otherwise false
                tgt_ids = torch.cat([v["labels"][msk] for v, msk in zip(targets, tgt_mask)])
                tgt_ids = torch.zeros_like(tgt_ids)
            else:
                tgt_mask = [[label is not None for label in t["labels"]] for t in targets] # dummy mask to get all GTs
                tgt_ids = torch.cat([v["labels"] for v in targets])
                tgt_center_points = torch.cat([v["center_points"] for v in targets])

            if tgt_center_points.numel() == 0:
                # result_indices[head_id] = (torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))
                continue

            if number_of_heads > 1:
                cost_class = -out_prob.unsqueeze(-1)[:, tgt_ids]
            elif self.focal_loss:
                # Compute the classification cost.
                neg_cost_class = (1 - self.focal_alpha) * (out_prob ** self.focal_gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.focal_alpha * ((1 - out_prob) ** self.focal_gamma) * (-(out_prob + 1e-8).log())

                # tgt_ids - concatenated GT label ids
                # Per each query we contains logits per all labes
                # [batch_size * num_queries, batch_size]
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            else:
                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be ommitted.
                cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between center points
            cost_center_points = torch.cdist(out_center_points, tgt_center_points, p=1)

            # Final cost matrix
            # [batch_size * num_queries, batch_size]
            cost_matrix = self.cost_class * cost_class \
                          + self.cost_center_point * cost_center_points

            # [batch_size, num_queries, batch_size]
            cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

            # Number of GTs per each batch index
            # [batch_size]
            sizes = [len(v["labels"][msk]) for v, msk in zip(targets, tgt_mask)] # Changed from targets to labels

            # Split returns a tuple where each element has hape [batch_size, num_queries, size]
            # During enumeration we assign particular batch index between query and GT size
            indices = [linear_sum_assignment(c[i])
                       for i, c in enumerate(cost_matrix.split(sizes, -1))]

            indices_head = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]

            result_indices[head_id] = indices_head

        return result_indices


def build_matcher(args):
    return HungarianMatcher(
        focal_loss=args.focal_loss,
        cost_class = 2,
        cost_center_point = 5,
    )

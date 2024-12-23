import numpy as np
import torch
from torch import nn

class AverageDisplacementErrorEvaluator:

    def __init__(self, matcher, img_size):
        self.matcher = matcher  # Matcher for finding the best target for each prediction
        self.ADE = []  # List to store displacement errors
        self.result = None  # Placeholder for accumulated result
        self.img_size = img_size
        self.final_displacement_error = None
        self.average_displacement_error_sequence_second_half = None

    def update(self, outputs, targets):
        """
        Updates the evaluator with outputs and targets by computing the displacement errors.

        Parameters:
        outputs (dict): Dictionary containing the model's predicted outputs.
        targets (list): List of dictionaries containing the ground-truth target data.
        """
        # Match predictions to targets
        indices = self.matcher(outputs, targets)

        # Permute predictions based on matched indices
        idx = self._get_src_permutation_idx(indices)
        src_cps = outputs['pred_center_points'][idx] # Predicted center points
        src_cps *= torch.tensor([self.img_size, self.img_size], dtype=torch.float32)

        target_cps = torch.cat([t['center_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # True center points
        target_cps *= torch.tensor([self.img_size, self.img_size], dtype=torch.float32)

        displacements = torch.norm(src_cps - target_cps, dim=1).detach().cpu()  # Shape: [N, 1]

        batch_idx, _ = idx

        for i in range(len(batch_idx)):
            idx = batch_idx[i].item()

            if len(self.ADE) == idx:
                # no list for the timestamp (or batch id)
                self.ADE.append([])

            assert self.ADE[idx] is not None, "List for the batch index should exist"
            self.ADE[idx].append(displacements[i].item())

    def accumulate(self):
        """
        Computes the final accumulated result by averaging all stored errors.
        """
        self.result = []
        for displacements_per_timestamp in self.ADE:
            mean = float(np.mean(displacements_per_timestamp))
            self.result.append(mean)

        self.final_displacement_error = self.result[-1] # Last timestamp
        self.average_displacement_error_sequence_second_half = float(np.mean(self.result[len(self.result)//2:]))

    def summary(self):
        """
        Returns a summary of the results.

        Returns:
        dict: Dictionary containing the ADE result with the prefix as a key.
        """
        result_dict = {f'ADE_{t}': ade for t, ade in enumerate(self.result)}
        result_dict['FDE'] = self.final_displacement_error
        result_dict['ADE_seq_second_half'] = self.average_displacement_error_sequence_second_half
        return result_dict

    def _get_src_permutation_idx(self, indices):
      # permute predictions following indices
      batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
      src_idx = torch.cat([src for (src, _) in indices])
      return batch_idx, src_idx


class PostProcessTrajectory(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets):
        outputs = {
            k: v.detach().cpu() for k, v in outputs.items()
        }

        targets = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]

        return outputs, targets

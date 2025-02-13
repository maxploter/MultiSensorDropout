import torch
from torch import nn

from models.perceiver import build_model_perceiver, PerceiverDetection


class PerceiverAr(nn.Module):

    def __init__(self,
                 detection_model,
                 ):
        super().__init__()
        self.detection_model = detection_model

    def forward(self, samples, targets: list = None):
        if len(samples.shape) < 6:
            # samples without a time dimension
            raise NotImplementedError("Not implemented yet samples without a time dimension.")

        src = samples.permute(1, 0, 2, 3, 4, 5)  # change dimension order from BT___ to TB___

        device = None
        result = {'pred_logits': [], 'pred_center_points': []}

        orig_size = torch.stack([t[-1]["orig_size"] for t in targets], dim=0).to(device)

        out_baseline = None
        hs = None

        assert len(targets) == 1
        targets = targets[0] # We have an assumption that batch size is 1

        for timestamp, batch in enumerate(src):
            keep_frame = targets[timestamp]['keep_frame'].bool().item()

            if not keep_frame:
                # drop the frame
                batch = torch.zeros_like(batch)

            out, targets_resp, features, memory, hs = self.detection_model.forward(
                samples=batch, targets=None, latents=hs, keep_encoder=keep_frame,
            )

            result['pred_logits'].extend(out['pred_logits']) # [QC]
            result['pred_center_points'].extend(out['pred_center_points'])

        return {
            'pred_logits': torch.stack(result['pred_logits']), #TQC
            'pred_center_points': torch.stack(result['pred_center_points']) #TQ2
        }, targets


def build_perceiver_ar_model(args, num_classes, input_image_view_size):
    backbone, perceiver, classification_head = build_model_perceiver(args, num_classes=num_classes, input_image_view_size=input_image_view_size)

    detection_model = PerceiverDetection(
        backbone, perceiver, classification_head, grid_size = args.grid_size
    )
    model = PerceiverAr(
        detection_model = detection_model,
    )

    return model

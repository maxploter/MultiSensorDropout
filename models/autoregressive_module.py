import torch
from torch import nn

from models.perceiver import build_model_perceiver

class AutoRegressiveModule(nn.Module):

    def __init__(self,
                 backbone,
                 recurrent_module,
                 detection_heads,
                 number_of_views,
                 ):
        super().__init__()

        self.backbone = backbone
        self.recurrent_module = recurrent_module
        self.detection_head = detection_heads

        feat_h, feat_w = self.backbone.output_size
        self.pos_embed = nn.Parameter(torch.zeros(number_of_views, backbone.num_channels, feat_h, feat_w))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, samples, targets: list = None):

        src = samples.permute(1, 2, 0, 3, 4, 5)  # change dimension order from BTN___ to TNB___

        device = None
        result = {}
        hs = None

        assert len(targets) == 1
        targets = targets[0] # We have an assumption that batch size is 1

        for timestamp, batch in enumerate(src):
            active_views = targets[timestamp]['active_views'].bool()

            for view_id, batch_view in enumerate(batch):
                if active_views[view_id]:
                    batch_view = self.backbone(batch_view)
                    batch_view = batch_view.permute(0, 2, 3, 1)
                    batch_view += self.pos_embed[view_id]
                else:
                    # drop the view
                    batch_view = None

                hs = self.recurrent_module(
                    data=batch_view,
                    latents=hs,
                )

            out = {}
            for i, head in enumerate(self.detection_heads):
                out[head.detection_object_id] = head(hs)

            for detection_object_id, out_predictions in out.items():
                if detection_object_id not in result:
                    result[detection_object_id] = {'pred_logits': [], 'pred_center_points': []}

                result[detection_object_id]['pred_logits'].extend(out_predictions['pred_logits']) # [QC]
                result[detection_object_id]['pred_center_points'].extend(out_predictions['pred_center_points'])

        for _, pred in result.items():
            pred['pred_logits'] = torch.stack(pred['pred_logits'])
            pred['pred_center_points'] = torch.stack(pred['pred_center_points'])

        return result, targets


def build_perceiver_ar_model(args, num_classes, input_image_view_size):
    backbone, perceiver, detection_heads = build_model_perceiver(args, num_classes=num_classes, input_image_view_size=input_image_view_size)

    model = AutoRegressiveModule(
        backbone=backbone,
        recurrent_module=perceiver,
        detection_heads=detection_heads,
        number_of_views=args.grid_size[0] * args.grid_size[1],
    )

    return model

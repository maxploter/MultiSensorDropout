import torch
from torch import nn

from models.perceiver import build_model_perceiver

class AutoRegressiveModule(nn.Module):

    def __init__(self,
                 backbone,
                 recurrent_module,
                 detection_head,
                 number_of_views,
                 shuffle_views=False,
                 ):
        super().__init__()

        self.backbone = backbone
        self.recurrent_module = recurrent_module
        self.detection_head = detection_head

        # feat_h, feat_w = self.backbone.output_size
        # self.pos_encod = nn.Parameter(torch.zeros(number_of_views, backbone.num_channels, feat_h, feat_w))
        # nn.init.normal_(self.pos_encod, std=0.02)
        self.shuffle_views = shuffle_views

    def forward(self, samples, targets: list = None):

        src = samples.permute(1, 2, 0, 3, 4, 5)  # change dimension order from BTN___ to TNB___

        device = None
        result = {'pred_logits': [], 'pred_center_points': []}
        hs = None

        assert len(targets) == 1
        targets = targets[0] # We have an assumption that batch size is 1

        for timestamp, batch in enumerate(src):
            active_views = targets[timestamp]['active_views'].bool()

            permutations = torch.arange(batch.size(0))
            if self.shuffle_views:
                permutations = torch.randperm(batch.size(0))
                batch = batch[permutations]
                active_views = active_views[permutations]

            for view_id, batch_view in enumerate(batch):
                if active_views[view_id]:
                    batch_view = self.backbone(batch_view)
                    # batch_view = batch_view + self.pos_encod[permutations[view_id]] #TODO: check if this is still needed
                    batch_view = batch_view.permute(0, 2, 3, 1) # [B, H, W, C]
                else:
                    # drop the view
                    batch_view = None

                hs = self.recurrent_module(
                    data=batch_view,
                    sensor_id=permutations[view_id],
                    latents=hs,
                )

                #TODO: improve
                if isinstance(hs, tuple):
                    q = hs[0]
                    hs = hs[1]
                else:
                    q = hs

            out = self.detection_head(q)

            result['pred_logits'].extend(out['pred_logits']) # [QC]
            result['pred_center_points'].extend(out['pred_center_points'])

        result['pred_logits'] = torch.stack(result['pred_logits'])
        result['pred_center_points'] = torch.stack(result['pred_center_points'])

        return result, targets


class RecurrentVideoObjectModule(nn.Module):

    def __init__(self,
                 backbone,
                 recurrent_module,
                 detection_head,
                 ):
        super().__init__()

        self.backbone = backbone
        self.recurrent_module = recurrent_module
        self.detection_head = detection_head

    def forward(self, samples, targets: list = None):
        src = samples.permute(1, 0, 2, 3, 4)  # change dimension order from BT___ to TB___

        device = None
        result = {'pred_logits': [], 'pred_boxes': []}
        hs = None

        assert len(targets) == 1
        targets = targets[0] # We have an assumption that batch size is 1

        for timestamp, batch in enumerate(src):
            batch = self.backbone(batch)
            batch = batch.permute(0, 2, 3, 1)  # [B, H, W, C]

            hs = self.recurrent_module(
                data=batch,
                sensor_id=0,
                latents=hs,
            )
            q = hs
            out = self.detection_head(q)

            result['pred_logits'].extend(out['pred_logits']) # [QC]
            result['pred_boxes'].extend(out['pred_boxes'])

        result['pred_logits'] = torch.stack(result['pred_logits'])
        result['pred_boxes'] = torch.stack(result['pred_boxes'])

        return result, targets


def build_perceiver_ar_model(args, num_classes, input_image_view_size):
    backbone, perceiver, detection_head = build_model_perceiver(args, num_classes=num_classes, input_image_view_size=input_image_view_size)

    model = AutoRegressiveModule(
        backbone=backbone,
        recurrent_module=perceiver,
        detection_head=detection_head,
        number_of_views=args.grid_size[0] * args.grid_size[1],
        shuffle_views=args.shuffle_views
    )

    return model

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
                 supervision_dropout_strategy='none',  # 'none', 'random', 'fixed_window', 'variable_window', 'last_only'
                 supervision_dropout_rate=0.3,
                 supervision_window_size=5,
                 ):
        super().__init__()

        self.backbone = backbone
        self.recurrent_module = recurrent_module
        self.detection_head = detection_head

        # Supervision dropout parameters
        self.supervision_dropout_strategy = supervision_dropout_strategy
        self.supervision_dropout_rate = supervision_dropout_rate
        self.supervision_window_size = supervision_window_size

    def forward(self, samples, targets: list = None):
        src = samples.permute(1, 0, 2, 3, 4)  # change dimension order from BT___ to TB___

        device = None
        result = {'pred_logits': [], 'pred_boxes': []}
        dropped_timesteps = []
        hs = None

        assert len(targets) == 1
        targets = targets[0] # We have an assumption that batch size is 1

        num_timesteps = len(src)

        # Determine which timesteps to supervise based on the dropout strategy
        supervise_timesteps = torch.ones(num_timesteps, dtype=torch.bool)

        if self.training and self.supervision_dropout_strategy != 'none':
            if self.supervision_dropout_strategy == 'random':
                # Randomly drop supervision with probability supervision_dropout_rate
                supervise_timesteps = torch.rand(num_timesteps) > self.supervision_dropout_rate

            elif self.supervision_dropout_strategy == 'fixed_window':
                window_size = self.supervision_window_size

                # Apply a random shift to the window position
                max_start_pos = num_timesteps - window_size - 1
                start_pos = torch.randint(0, max_start_pos + 1, (1,)).item()
                supervise_timesteps[start_pos:start_pos + window_size] = False


            elif self.supervision_dropout_strategy == 'variable_window':
                # Randomly select a window size between 0 and supervision_window_size
                window_size = torch.randint(0, self.supervision_window_size + 1, (1,)).item()

                # Apply a random shift to the window position
                max_start_pos = num_timesteps - window_size - 1
                start_pos = torch.randint(0, max_start_pos + 1, (1,)).item()
                supervise_timesteps[start_pos:start_pos + window_size] = False

            elif self.supervision_dropout_strategy == 'last_only':
                # Only supervise the last timestep
                supervise_timesteps.fill_(False)

                # Check if the last timestep has ground truth objects
                last_idx = num_timesteps - 1
                if len(targets[last_idx]['boxes']) > 0:
                    # Use last timestep if it has ground truth objects
                    supervise_timesteps[last_idx] = True
                else:
                    # Find the last timestep with ground truth objects
                    valid_timestep_found = False
                    for t in range(num_timesteps - 1, -1, -1):
                        if len(targets[t]['boxes']) > 0:
                            supervise_timesteps[t] = True
                            valid_timestep_found = True
                            break

        # Always keep predictions for all timesteps (we just don't compute loss for dropped ones)
        full_pred_logits = []
        full_pred_boxes = []

        for timestamp, batch in enumerate(src):
            batch = self.backbone(batch)

            hs = self.recurrent_module(
                data=batch,
                sensor_id=0,
                latents=hs,
            )
            q = hs
            out = self.detection_head(q)

            # Always store predictions for all timesteps
            full_pred_logits.extend(out['pred_logits'])
            full_pred_boxes.extend(out['pred_boxes'])

            # But only include supervised timesteps in the result for loss computation
            if supervise_timesteps[timestamp]:
                result['pred_logits'].extend(out['pred_logits'])
                result['pred_boxes'].extend(out['pred_boxes'])
            else:
                dropped_timesteps.append(timestamp)

        # Store full predictions for inference
        result['full_pred_logits'] = torch.stack(full_pred_logits)
        result['full_pred_boxes'] = torch.stack(full_pred_boxes)

        # Store the dropped timesteps for logging/debugging
        result['dropped_timesteps'] = dropped_timesteps

        last_idx = num_timesteps - 1
        # Only stack if there are supervised timesteps
        if len(result['pred_logits']) > 0:
            result['pred_logits'] = torch.stack(result['pred_logits'])
            result['pred_boxes'] = torch.stack(result['pred_boxes'])
        else:
            for t in range(num_timesteps - 1, -1, -1):
                if len(targets[t]['boxes']) > 0:
                    last_idx = t
                    break

            # Fallback in case all timesteps were dropped (should be rare)
            result['pred_logits'] = result['full_pred_logits'][last_idx:last_idx+1]
            result['pred_boxes'] = result['full_pred_boxes'][last_idx:last_idx+1]

        # If we're not in training mode, use all predictions
        if self.training and self.supervision_dropout_strategy != 'none':
            # Filter targets to match the supervised timesteps
            # Since targets is a list (one element per timestep), we need to filter it accordingly
            filtered_targets = []
            for t in range(num_timesteps):
                if supervise_timesteps[t]:
                    filtered_targets.append(targets[t])

            # If we filtered out all timesteps, use just the last one as fallback
            if len(filtered_targets) == 0:
                filtered_targets = [targets[last_idx:last_idx+1]]

            targets = filtered_targets

        return result, targets


class SingleFrameModule(nn.Module):
    def __init__(self,
                 backbone,
                 recurrent_module,
                 detection_head,
                 ):
        super().__init__()

        self.backbone = backbone
        self.view_module = recurrent_module
        self.detection_head = detection_head

    def forward(self, samples, targets: list = None):
        assert samples.shape[1] == 1, "SingleFrameModule expects input with T=1"

        src = samples.squeeze(1)  # Remove the time dimension, shape should be [B, C, H, W]

        result = {'pred_logits': [], 'pred_boxes': []}

        # No assumption about batch size here
        # Process the single frame (T=1) for all batches
        batch = self.backbone(src)

        # Process through view module
        out = self.view_module(data=batch)

        # Generate detection output
        out = self.detection_head(out)

        result['pred_logits'] = out['pred_logits']
        result['pred_boxes'] = out['pred_boxes']

        return result, [t[0] for t in targets]

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

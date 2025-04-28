import math
import sys
from collections import defaultdict

import torch
import torchmetrics
from tqdm import tqdm

from models.ade_post_processor import AverageDisplacementErrorEvaluator, MultiHeadPostProcessTrajectory, \
    MultiHeadAverageDisplacementErrorEvaluator
from util.box_ops import box_cxcywh_to_xyxy


def train_one_epoch(model, dataloader, optimizer, criterion, epoch, device):
    model.train()
    metric_logger = defaultdict(lambda: torchmetrics.MeanMetric().to(device))

    add_running_metrics(device, metric_logger)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)

    print_freq = 50

    for i, (samples, targets) in enumerate(progress_bar):
        samples = samples.to(device)

        targets = [[{k: v.to(device) for k, v in t.items()} for t in batch_targets] for batch_targets in targets]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        out, targets_flat = model(samples, targets)
        loss_dict = criterion(out, targets_flat)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Logic related to reduce
        loss_dict_unscaled = {
            f'{k}_unscaled': v for k, v in loss_dict.items()}
        loss_dict_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
        losses_scaled = sum(loss_dict_scaled.values())

        loss_value = losses_scaled.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        # Backpropagation and optimization
        losses.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Add gradient clipping
        optimizer.step()

        # Update metric logger with main loss and each component
        metric_logger['loss'].update(loss_value)
        metric_logger['loss_running'].update(loss_value)

        if 'class_error' in loss_dict:
            metric_logger["class_error"].update(loss_dict['class_error'].item())
            metric_logger["class_error_running"].update(loss_dict['class_error'].item())

        for k, v in loss_dict_unscaled.items():
            if k == 'class_error_unscaled':
                continue
            metric_logger[k].update(v.item())
        for k, v in loss_dict_scaled.items():
            metric_logger[k].update(v.item())
            if f'{k}_running' in metric_logger:
                metric_logger[f'{k}_running'].update(v.item())

        for k, v in loss_dict.items():
            if 'binary_precision' in k or 'binary_recall' in k or 'binary_f1' in k:
                metric_logger[k].update(loss_dict[k].item())

        if i % print_freq == 0 or i == len(dataloader) - 1:
            progress_bar.set_postfix({
                **{k: metric.compute().item() for k, metric in metric_logger.items()},
                "lr": optimizer.param_groups[0]["lr"],
                'view_dropout_prob': dataloader.dataset.view_dropout_prob,
            })

    avg_values = {k: metric.compute().item() for k, metric in metric_logger.items()}

    for metric in metric_logger.values():
      metric.reset()

    return avg_values


def add_running_metrics(device, metric_logger):
    window = 10
    metric_logger['loss_running'] = torchmetrics.RunningMean(window=window).to(device)
    metric_logger['class_error_running'] = torchmetrics.RunningMean(window=window).to(device)
    metric_logger['loss_center_point_running'] = torchmetrics.RunningMean(window=window).to(device)
    metric_logger['loss_ce_running'] = torchmetrics.RunningMean(window=window).to(device)


def evaluate(model, dataloader, criterion, postprocessors, epoch, device, evaluators=None):
    average_displacement_error_evaluator = None
    map_metric = None # Initialize map_metric variable
    if 'trajectory' in postprocessors:
        if postprocessors['trajectory'] is MultiHeadPostProcessTrajectory:
            average_displacement_error_evaluator = MultiHeadAverageDisplacementErrorEvaluator(
                matcher=criterion.matcher,
                coordinate_norm_const=dataloader.dataset.coordinate_norm_const,
            )
        else:
            average_displacement_error_evaluator = AverageDisplacementErrorEvaluator(
                matcher=criterion.matcher,
                coordinate_norm_const=dataloader.dataset.coordinate_norm_const,
            )

    # --- Initialize TorchMetrics mAP Calculator (if applicable) ---
    if 'bbox' in postprocessors:
        # Use torchmetrics for mAP calculation
        # Specify box format ('xyxy', 'xywh', 'cxcywh') based on your model output/postprocessor
        # Make sure it matches the format in both predictions and targets
        map_metric = torchmetrics.detection.MeanAveragePrecision(box_format='xyxy', iou_type='bbox').to(device)
        # coco_evaluator = CocoEvaluator(base_ds, 'bbox') # Remove or comment out old evaluator


    model.eval()

    print_freq = 50

    metric_logger = defaultdict(lambda: torchmetrics.MeanMetric().to(device))
    add_running_metrics(device, metric_logger)

    # Disabling gradient calculation for evaluation
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Eval {epoch}:", leave=True)

        for i, (samples, targets) in enumerate(progress_bar):
            samples = samples.to(device)
            targets = [[{k: v.to(device) for k, v in t.items()} for t in batch_targets] for batch_targets in targets]

            # Forward pass
            out, targets_flat = model(samples, targets)

            loss_dict = criterion(out, targets_flat)

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # Logic related to reduce
            loss_dict_unscaled = {
                f'{k}_unscaled': v for k, v in loss_dict.items()}
            loss_dict_scaled = {
                k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
            losses_scaled = sum(loss_dict_scaled.values())

            loss_value = losses_scaled.item()

            metric_logger['loss'].update(loss_value)
            metric_logger['loss_running'].update(loss_value)

            if 'class_error' in loss_dict:
                metric_logger["class_error"].update(loss_dict['class_error'].item())
                metric_logger["class_error_running"].update(loss_dict['class_error'].item())

            for k, v in loss_dict_unscaled.items():
                if k == 'class_error_unscaled':
                    continue
                metric_logger[k].update(v.item())
            for k, v in loss_dict_scaled.items():
                metric_logger[k].update(v.item())
                if f'{k}_running' in metric_logger:
                    metric_logger[f'{k}_running'].update(v.item())

            if i % print_freq == 0 or i == len(dataloader) - 1:
              progress_bar.set_postfix({
                  **{k: metric.compute().item() for k, metric in metric_logger.items()},
                  'view_dropout_prob': dataloader.dataset.view_dropout_prob,
              })

            if average_displacement_error_evaluator:
              average_displacement_error_evaluator.update(*postprocessors['trajectory'](out, targets_flat))

            if evaluators:
              for evaluator in evaluators:
                evaluator.update(out, targets_flat)

            if map_metric is not None:
                # 1. Prepare Predictions using Postprocessor
                # The postprocessor likely needs original image sizes.
                # This part requires accessing the original `targets` structure from the dataloader.
                # Adapt the loops below based on how your `targets` are structured.
                orig_target_sizes = []
                targets_for_metric = []
                try:
                    for batch_target_list in targets:
                        for target_dict in batch_target_list:
                            # Extract original size for postprocessor
                            orig_target_sizes.append(target_dict['orig_size'])  # Assumes 'orig_size' key exists
                            img_h, img_w = target_dict['orig_size']
                            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                            boxes = target_dict['boxes']
                            if boxes.numel() > 0:
                                boxes = box_cxcywh_to_xyxy(boxes) * scale_fct

                            targets_for_metric.append({
                                'boxes': boxes,
                                'labels': target_dict['labels']
                            })

                    orig_target_sizes_tensor = torch.stack(orig_target_sizes, dim=0).to(device)

                    # Call the postprocessor with model output and original sizes
                    predictions_for_metric = postprocessors['bbox'](out, orig_target_sizes_tensor)

                    # 2. Update Metric
                    # Ensure the number of prediction dicts matches the number of target dicts
                    if len(predictions_for_metric) == len(targets_for_metric):
                        map_metric.update(predictions_for_metric, targets_for_metric)
                    else:
                        # Log a warning if counts mismatch for this batch
                        print(
                            f"Warning: Batch {i}: Prediction count ({len(predictions_for_metric)}) != Target count ({len(targets_for_metric)}). Skipping mAP update.")

                except (KeyError, TypeError, IndexError, AttributeError) as e:
                    # Catch potential errors if keys ('orig_size', 'boxes', 'labels') are missing
                    # or if targets structure is not as expected.
                    print(f"Warning: Batch {i}: Error preparing data for mAP metric: {e}. Skipping mAP update.")
            # --- End mAP Metric Update ---

    avg_values = {k: metric.compute().item() for k, metric in metric_logger.items()}

    if average_displacement_error_evaluator:
      average_displacement_error_evaluator.accumulate()
      avg_values.update(average_displacement_error_evaluator.summary())

    if evaluators:
      for evaluator in evaluators:
        evaluator.accumulate()
        avg_values.update(evaluator.summary())

    for metric in metric_logger.values():
      metric.reset()

    if map_metric is not None:
        try:
            map_results = map_metric.compute()

            # --- Process results: Convert only SCALAR tensors to items ---
            if map_results: # Check if the results dictionary is not empty
                 print(f"\nRaw mAP Results: {map_results}\n") # Optional: Print raw results for debugging
                 processed_map_results = {}
                 for k, v in map_results.items():
                     # Check if the value is a tensor AND has only one element
                     if isinstance(v, torch.Tensor) and v.numel() == 1:
                         processed_map_results[f'mAP_{k}'] = v.item()
                         # Example: Adds 'mAP_map', 'mAP_map_50', etc. to the dict
                     # else:
                         # Optional: You could handle multi-element tensors differently if needed
                         # print(f"Skipping non-scalar mAP result key: {k}") # Uncomment for debug
                         pass # Otherwise, just skip non-scalar tensors like 'classes'

                 # Check if any metrics were actually processed before updating
                 if processed_map_results:
                      avg_values.update(processed_map_results)
                 else:
                      print("Warning: No scalar mAP metrics were found in map_results.")
                      avg_values['mAP_computation_warning'] = 1

            else:
                 print("Warning: mAP metric computation returned empty results.")
                 avg_values['mAP_computation_warning'] = 1
            # --- End Result Processing ---

        except Exception as e:
            print(f"Error computing or processing final mAP metric: {e}")
            avg_values['mAP_error'] = 1

    return avg_values

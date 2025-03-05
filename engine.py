import math
import sys
from collections import defaultdict

import torch
import torchmetrics
from tqdm import tqdm

from models.ade_post_processor import AverageDisplacementErrorEvaluator, MultiHeadPostProcessTrajectory, \
    MultiHeadAverageDisplacementErrorEvaluator


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


def evaluate(model, dataloader, criterion, postprocessors, epoch, device):
    average_displacement_error_evaluator = None
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

    avg_values = {k: metric.compute().item() for k, metric in metric_logger.items()}

    if average_displacement_error_evaluator:
      average_displacement_error_evaluator.accumulate()
      avg_values.update(average_displacement_error_evaluator.summary())

    for metric in metric_logger.values():
      metric.reset()

    return avg_values

import torch
import numpy as np
import motmetrics as mm
from collections import defaultdict
from util import box_ops

class MOTMetricsEvaluator:
    """Evaluator for Multiple Object Tracking metrics"""

    def __init__(self, postprocessor, iou_threshold=0.5, score_threshold=0.5):
        """
        Args:
            postprocessor: Optional postprocessor to convert model outputs to final predictions
            iou_threshold: IoU threshold for considering a detection as matched
            score_threshold: Score threshold for keeping detections
        """
        self.postprocessor = postprocessor
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.reset()

    def reset(self):
        """Reset the accumulator for a new evaluation"""
        self.accumulators = []
        self.sequence_names = []
        self.frame_count = 0

    def update(self, outputs, targets, samples=None):
        """Update with predictions and ground truth for current batch

        In this implementation:
        - outputs: contains predictions for a single video, where batch_size equals number of frames
        - targets: already flattened list of targets for each frame
        """
        orig_sizes = torch.stack([t['orig_size'] for t in targets]).to(outputs['pred_logits'].device)
        processed_results = self.postprocessor(outputs, orig_sizes)

        # Create a single accumulator for the entire video sequence
        acc = mm.MOTAccumulator(auto_id=True)
        self.accumulators.append(acc)
        self.sequence_names.append(f"seq_{len(self.sequence_names)}")

        # Process each frame in the sequence
        num_frames = len(outputs['pred_logits'])  # batch size is the number of frames

        for frame_idx in range(num_frames):
            self.frame_count += 1
            target = targets[frame_idx]

            gt_boxes = target['boxes']  # [num_objects, 4] in normalized [cx, cy, w, h] format
            gt_ids = target.get('track_ids', torch.arange(len(gt_boxes)))  # Use provided IDs or assign sequential IDs

            # Use processed results if available (after post-processing)
            pred_scores = processed_results[frame_idx]['scores']
            pred_filtered_ids = torch.nonzero(pred_scores > self.score_threshold).squeeze(-1)
            
            # Handle case where pred_filtered_ids is a 0-dim tensor (single value)
            if pred_filtered_ids.ndim == 0 and pred_filtered_ids.numel() == 1:
                pred_filtered_ids = pred_filtered_ids.unsqueeze(0)
                
            # Filter boxes using the same score threshold
            pred_filtered_boxes = processed_results[frame_idx]['boxes'][pred_filtered_ids]

            # If no predictions or ground truths, update with empty arrays and continue
            if len(pred_filtered_boxes) == 0 or len(gt_boxes) == 0:
                acc.update(
                    gt_ids.cpu().numpy() if len(gt_boxes) > 0 else [],
                    pred_filtered_ids.cpu().numpy() if len(pred_filtered_boxes) > 0 else [],
                    np.empty((0, 0))
                )
                continue

            # Scale ground truth boxes to match the same format as prediction boxes
            # First convert from [cx, cy, w, h] to [x0, y0, x1, y1]
            gt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes)

            # Then scale from normalized [0, 1] to absolute coordinates
            img_h, img_w = target['orig_size']
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=gt_boxes.device)
            gt_boxes_scaled = gt_boxes_xyxy * scale_fct

            # Calculate IoU distances between predictions and scaled ground truth
            distances = mm.distances.iou_matrix(
                gt_boxes_scaled.cpu().numpy(),
                pred_filtered_boxes.cpu().numpy(),
                max_iou=1 - self.iou_threshold
            )

            # Update accumulator
            acc.update(
                gt_ids.cpu().numpy(),
                pred_filtered_ids.cpu().numpy(),
                distances
            )

    def accumulate(self):
        """Accumulate metrics from all batches"""
        self.metrics_host = defaultdict(list)

        # Filter out empty accumulators
        valid_accumulators = []
        valid_seq_names = []

        for seq_idx, acc in enumerate(self.accumulators):
            if len(acc.events) > 0:
                valid_accumulators.append(acc)
                valid_seq_names.append(self.sequence_names[seq_idx])

        # Skip if no valid accumulators
        if not valid_accumulators:
            return

        # Create metrics host
        mh = mm.metrics.create()

        # Compute metrics across all sequences
        summary = mh.compute_many(
            valid_accumulators,
            metrics=mm.metrics.motchallenge_metrics,
            names=valid_seq_names,
            generate_overall=True
        )

        # Generate string summary for logging
        self.summary_str = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )

        # Store metrics for each sequence
        for seq_idx, name in enumerate(valid_seq_names):
            seq_metrics = summary.loc[name]
            for metric, value in seq_metrics.items():
                if isinstance(value, (int, float, np.number)):
                    self.metrics_host[metric].append(float(value))

        # Store overall metrics
        if 'OVERALL' in summary.index:
            overall_metrics = summary.loc['OVERALL']
            for metric, value in overall_metrics.items():
                if isinstance(value, (int, float, np.number)):
                    self.metrics_host[f'overall_{metric}'] = float(value)

    def summary(self):
        """Return summary of evaluation metrics"""
        result = {}

        # Calculate mean of metrics across all sequences
        for metric, values in self.metrics_host.items():
            if values:
                result[f'mot_{metric}'] = np.mean(values)

        # Add frame count
        result['mot_frame_count'] = self.frame_count

        return result

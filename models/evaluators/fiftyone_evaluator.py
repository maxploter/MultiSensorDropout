import os
import shutil
import fiftyone as fo
import numpy as np
import torch
import torch.nn.functional as F
import imageio
from util.box_ops import box_cxcywh_to_xyxy  # Assuming you have this utility


# Helper function to denormalize images, can be placed in a utils file
def denormalize_image(tensor, mean, std):
    """Denormalizes a tensor image with mean and standard deviation."""
    tensor = tensor.clone()
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


class FiftyOneEvaluator:
    """
    An evaluator that generates a FiftyOne dataset to visualize model predictions.

    This class is designed to be integrated into an evaluation loop. It uses a
    postprocessor to convert model outputs into detections, saves the input frames
    as images, and populates a FiftyOne dataset with ground truth and predicted
    detections.
    """

    def __init__(self, postprocessor, save_dir, dataset_name, class_names, model):
        """
        Initializes the FiftyOne evaluator.

        Args:
            postprocessor: The postprocessor module to convert model outputs to final predictions.
            save_dir (str): The root directory to save FiftyOne data and exported datasets.
            dataset_name (str): The base name for the FiftyOne dataset.
            class_names (list): A list of class name strings, where the index corresponds to the class ID.
        """
        self.postprocessor = postprocessor
        self.dataset_name = dataset_name
        self.class_names = class_names

        # --- 1. Setup Paths ---
        self.save_dir = save_dir
        self.frames_dir = os.path.join(self.save_dir, f"frames_{self.dataset_name}")
        self.export_dir = os.path.join(self.save_dir, f"{self.dataset_name}_export")

        # --- 2. Clean Up Previous Runs ---
        print(f"[FiftyOne] Cleaning up previous data for dataset '{self.dataset_name}'...")
        if fo.dataset_exists(self.dataset_name):
            fo.delete_dataset(self.dataset_name)
        if os.path.exists(self.export_dir):
            shutil.rmtree(self.export_dir)
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)

        os.makedirs(self.frames_dir, exist_ok=True)

        # --- 3. Initialize FiftyOne Dataset ---
        self.dataset = fo.Dataset(self.dataset_name)
        self.dataset.persistent = True

        # --- 4. Initialize Data Containers ---
        self.fiftyone_samples = []
        self.video_idx = 0  # To give each video/batch a unique ID

        # --- 5. Model-specific stats for denormalization ---
        # This is for visualizing the input images correctly.
        self.mmnist_stat = {
            'perceiver': (0.030643088476670285, 0.15920598247588932),  # (mean, std)
            'YOLO': (0, 1),  # No normalization
        }
        self.model = model

    def _convert_gt_boxes_to_fo_format(self, boxes_cxcywh):
        """Converts normalized [cx, cy, w, h] to FiftyOne's relative [x, y, w, h] format."""
        if not isinstance(boxes_cxcywh, torch.Tensor) or boxes_cxcywh.numel() == 0:
            return np.array([])

        boxes_cxcywh = boxes_cxcywh.detach().cpu()
        cx, cy, w, h = boxes_cxcywh.T
        # FiftyOne format is top-left-x, top-left-y, width, height
        return torch.stack([cx - w / 2, cy - h / 2, w, h], dim=1).numpy()

    def _convert_pred_boxes_to_fo_format(self, boxes_xyxy_pixel, img_size):
        """Converts pixel [x1, y1, x2, y2] to FiftyOne's relative [x, y, w, h] format."""
        if not isinstance(boxes_xyxy_pixel, torch.Tensor) or boxes_xyxy_pixel.numel() == 0:
            return np.array([])

        img_h, img_w = img_size.tolist()
        boxes_xyxy_pixel = boxes_xyxy_pixel.detach().cpu()

        x1, y1, x2, y2 = boxes_xyxy_pixel.T
        w = x2 - x1
        h = y2 - y1

        # Convert to relative [top-left-x, top-left-y, width, height]
        fo_boxes_relative = torch.stack([x1 / img_w, y1 / img_h, w / img_w, h / img_h], dim=1)
        return fo_boxes_relative.numpy()

    def update(self, out, targets, samples, **kwargs):
        """
        Processes a single batch of model outputs and targets to create FiftyOne samples.
        This method is called for each batch in the evaluation loop.
        """
        # --- 1. Process Predictions using Postprocessor ---
        # The postprocessor expects a batch of original image sizes.
        # For video, we assume all frames have the same size.
        num_frames = samples.shape[1]  # T from B, T, C, H, W
        orig_sizes = torch.stack([t['orig_size'] for t in targets]).to(out['pred_logits'].device)

        # `results` is a list of dicts, one per frame, with 'scores', 'labels', 'boxes'
        results = self.postprocessor(out, orig_sizes)

        # --- 2. Process and Save Input Images ---
        if samples.shape[0] != 1 and samples.shape[1] == 1:
            video_tensor = samples.squeeze(1)
        elif samples.shape[0] == 1:
            video_tensor = samples.squeeze(0)
        else:
            raise ValueError("Expected samples to have shape [B, T, C, H, W] or [T, C, H, W] for single frame.")

        # Infer model name for denormalization; default to (0, 1) if not found
        mean, std = self.mmnist_stat[self.model]

        image_filepaths = []
        for i in range(num_frames):
            img_tensor = denormalize_image(video_tensor[i], mean, std)
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

            if img_np.shape[2] == 1:
                img_np = np.squeeze(img_np, axis=-1)

            img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            filepath = os.path.join(self.frames_dir, f"video_{self.video_idx:04d}_frame_{i:04d}.png")
            imageio.imwrite(filepath, img_uint8)
            image_filepaths.append(filepath)

        # --- 3. Create FiftyOne Samples for each frame ---
        for i in range(num_frames):
            sample = fo.Sample(filepath=image_filepaths[i])

            # a) Add Ground Truth Detections
            gt_targets_for_frame = targets[i]
            gt_boxes_norm = gt_targets_for_frame["boxes"]

            if gt_boxes_norm.numel() > 0:
                fo_gt_boxes = self._convert_gt_boxes_to_fo_format(gt_boxes_norm)
                sample["ground_truth"] = fo.Detections(
                    detections=[
                        fo.Detection(label=self.class_names[label.item()], bounding_box=box)
                        for box, label in zip(fo_gt_boxes, gt_targets_for_frame["labels"])
                    ]
                )

            # b) Add Predicted Detections from Postprocessor
            res_for_frame = results[i]
            pred_boxes_pixel = res_for_frame['boxes']

            if pred_boxes_pixel.numel() > 0:
                fo_pred_boxes = self._convert_pred_boxes_to_fo_format(pred_boxes_pixel, orig_sizes[0])
                pred_scores = res_for_frame['scores'].cpu().numpy()
                pred_labels = res_for_frame['labels'].cpu().numpy()

                sample["predictions"] = fo.Detections(
                    detections=[
                        fo.Detection(
                            label=self.class_names[label_idx],
                            bounding_box=box,
                            confidence=float(score)
                        )
                        for box, score, label_idx in zip(fo_pred_boxes, pred_scores, pred_labels)
                    ]
                )
            self.fiftyone_samples.append(sample)

        self.video_idx += 1

    def accumulate(self):
        """
        Adds all processed samples to the FiftyOne dataset, saves it, and exports it.
        This should be called once after the evaluation loop is complete.
        """
        if not self.fiftyone_samples:
            print("[FiftyOne] No samples were generated to accumulate.")
            return

        print(f"\n[FiftyOne] Adding {len(self.fiftyone_samples)} samples to dataset '{self.dataset_name}'...")
        self.dataset.add_samples(self.fiftyone_samples)
        self.dataset.save()
        print(f"[FiftyOne] Dataset saved successfully.")

        print(f"[FiftyOne] Exporting dataset to '{self.export_dir}'...")
        self.dataset.export(
            export_dir=self.export_dir,
            dataset_type=fo.types.FiftyOneDataset,
            export_media=True
        )
        print(f"[FiftyOne] Export complete. You can now load this directory in the FiftyOne App.")

    def summary(self):
        """
        Returns a summary dictionary containing the path to the exported dataset.
        """
        return {
            "fiftyone_dataset_name": self.dataset_name,
            "fiftyone_export_path": self.export_dir
        }

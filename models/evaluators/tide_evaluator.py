import torch
import json
from pathlib import Path


class CocoJsonEvaluator:
    """
    Accumulates predictions and (optionally) ground truth in COCO format.
    At the end of evaluation, it saves them to JSON files.
    This allows using external tools like pycocotools for metric calculation.
    """

    def __init__(self, postprocessor, output_dir, checkpoint, save_gt=False, categories=None):
        """
        Initializes the CocoJsonEvaluator.

        Args:
            postprocessor: The postprocessor module to convert model outputs to final predictions.
            output_dir (str): Directory where the JSON files will be saved.
            save_gt (bool): If True, saves a ground_truth.json file. Defaults to False.
            categories (list, optional): A list of COCO-style category dictionaries.
                                         e.g., [{'id': 1, 'name': 'car'}, ...]. Defaults to None.
        """
        self.postprocessor = postprocessor
        self.output_dir = Path(output_dir)
        self.checkpoint = checkpoint
        self.save_gt = save_gt
        self.categories = categories if categories is not None else []

        # In-memory storage
        self.predictions = []
        self.ground_truths = []
        # Use a list for images since we will generate sequential IDs
        self.images = []
        # Counter for generating unique sequential image IDs
        self.img_count = 0
        # Unique ID for ground truth annotations
        self.ann_id_counter = 0

        print(f"CocoJsonEvaluator initialized. Saving files to: {self.output_dir}")
        if self.save_gt:
            print("Ground truth saving is ENABLED.")
        else:
            print("Ground truth saving is DISABLED.")

    def update(self, model_outputs, targets, samples):
        """
        Processes a batch of predictions and ground truths, converting them to COCO format.
        This method is called by the evaluation engine for each batch.

        Args:
            model_outputs (dict): The raw output from the model.
            targets (list[dict]): A list of ground truth dictionaries for each image in the batch.
        """
        # The postprocessor needs original image sizes to un-normalize the bounding boxes.
        orig_sizes = torch.stack([t['orig_size'] for t in targets]).to(model_outputs['pred_logits'].device)

        # Use the provided postprocessor to get results in the desired format
        # This typically returns a list of dicts, one per image, with 'scores', 'labels', 'boxes'.
        results = self.postprocessor(model_outputs, orig_sizes)

        # Iterate over each image's results and ground truth in the batch
        for i, (res, target) in enumerate(zip(results, targets)):
            # Generate a unique, sequential image_id
            image_id = self.img_count

            # Store unique image information (will be part of the final JSON)
            h, w = target['orig_size'].tolist()
            self.images.append({
                'id': image_id,
                'file_name': f'./i/i_{image_id}.jpg',
                'height': h,
                'width': w
            })

            # --- Process Predictions ---
            boxes = res['boxes']  # Expected in [x1, y1, x2, y2] format
            labels = res['labels']
            scores = res['scores']

            for box, label, score in zip(boxes, labels, scores):
                # COCO format requires [x, y, width, height]
                x1, y1, x2, y2 = box.tolist()
                bbox_coco = [x1, y1, x2 - x1, y2 - y1]

                self.predictions.append({
                    'image_id': image_id,
                    'category_id': label.item(),
                    'bbox': bbox_coco,
                    'score': score.item(),
                })

            # --- Process Ground Truth (if enabled) ---
            if self.save_gt:
                gt_boxes = target['boxes']  # Expected in [center_x, center_y, width, height] (normalized)
                gt_labels = target['labels']

                # Check if there are any ground truth boxes before processing
                if gt_boxes.numel() > 0:
                    # Un-normalize boxes
                    img_h, img_w = target['orig_size']
                    scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=gt_boxes.device)
                    gt_boxes_unscaled_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes) * scale_fct

                    for box, label in zip(gt_boxes_unscaled_xyxy, gt_labels):
                        x1, y1, x2, y2 = box.tolist()
                        bbox_coco = [x1, y1, x2 - x1, y2 - y1]
                        area = (x2 - x1) * (y2 - y1)

                        self.ground_truths.append({
                            'id': self.ann_id_counter,
                            'image_id': image_id,
                            'category_id': label.item(),
                            'bbox': bbox_coco,
                            'area': area,
                            'iscrowd': 0,  # Assuming no crowd annotations
                            'segmentation': {'counts': 0}
                        })
                        self.ann_id_counter += 1

            # Increment the image counter for the next image in the dataset
            self.img_count += 1

    def accumulate(self):
        """
        Saves the accumulated data to JSON files. This is called once at the end of evaluation.
        """
        print("Accumulating results and saving to JSON files...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Prepare final JSON structure ---
        images_list = self.images

        # Save predictions as a simple list of annotations
        pred_json_path = self.output_dir / f'{self.checkpoint[:-4]}_predictions.json'
        with open(pred_json_path, 'w') as f:
            # The user's parsing script expects a simple list of annotation dictionaries.
            json.dump(self.predictions, f, indent=4)
        print(f"✅ Predictions saved to: {pred_json_path}")

        # Save ground truth if enabled (using the standard COCO format)
        if self.save_gt:
            gt_json_path = self.output_dir / f'{self.checkpoint[:-4]}_ground_truth.json'
            final_gt = {
                'images': images_list,
                'annotations': self.ground_truths,
                'categories': self.categories
            }
            with open(gt_json_path, 'w') as f:
                json.dump(final_gt, f, indent=4)
            print(f"✅ Ground truth saved to: {gt_json_path}")

    def summary(self):
        """
        Returns a dictionary of metrics. For this evaluator, we just return a status message.
        """
        # This evaluator's job is to save files, not compute metrics like mAP.
        # The actual metrics can be computed offline using the saved JSON files.
        status = {
            'coco_files_saved_to': str(self.output_dir)
        }
        return status
import torch
import torch.nn as nn
from transformers import DetrForObjectDetection, DetrConfig
from einops import rearrange


class DetrWrapper(nn.Module):
    """Wrapper around HuggingFace's DETR implementation."""

    def __init__(self, num_classes, hidden_dim, nheads, enc_layers, dec_layers, backbone, dilation):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Initialize DETR configuration
        self.config = DetrConfig(
            num_labels=num_classes,
            d_model=hidden_dim,
            encoder_layers=enc_layers,
            decoder_layers=dec_layers,
            encoder_attention_heads=nheads,
            decoder_attention_heads=nheads,
            backbone=backbone,  # Specify ResNet18 as the backbone
            dilation=dilation,  # No dilation for ResNet18
            position_embedding_type="sine",  # Standard positional embedding for DETR
            use_pretrained_backbone=True,  # Use pretrained weights for faster convergence
        )

        # Initialize DETR model with custom configuration
        self.detr = DetrForObjectDetection(self.config)

    def forward(self, data, **kwargs):
        """
        Process input through DETR model.
        Compatible with the recurrent module interface expected by SingleFrameModule.
        """
        # Handle grayscale inputs
        if data.shape[1] == 1:
            data = data.repeat(1, 3, 1, 1)

        # Forward through DETR with its internal backbone
        outputs = self.detr(
            pixel_values=data,
            return_dict=True
        )

        # Return the full outputs to be used by the detection head
        return outputs


class DummyDetrDetectionHead(nn.Module):
    """
    Simple dummy detection head that extracts and formats the outputs
    from DETR's built-in detection head.
    """
    def __init__(self):
        super().__init__()

    def forward(self, detr_outputs, **kwargs):
        """
        Format DETR outputs into the expected dictionary structure.
        """
        return {
            "pred_logits": detr_outputs.logits,
            "pred_boxes": detr_outputs.pred_boxes
        }


def build_detr_model(args, num_classes, input_image_view_size):
    """Build DETR model using ResNet18 backbone."""

    detr_model = DetrWrapper(
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        nheads=args.detr_nheads,
        enc_layers=args.detr_enc_layers,
        dec_layers=args.detr_dec_layers,
        backbone=args.backbone,
        dilation=args.dilation
    )

    # Create a dummy backbone that just passes through the images
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_channels = 3  # RGB channels
            self.output_size = input_image_view_size

            # ResNet18 output channels in final feature map is 512
            # This might be helpful information for other parts of your code
            self.out_channels = 512

        def forward(self, x):
            return x

    backbone = DummyBackbone()

    # Use a dummy detection head that just formats the outputs
    detection_head = DummyDetrDetectionHead()

    return backbone, detr_model, detection_head

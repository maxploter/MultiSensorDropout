from models.autoregressive_module import build_perceiver_ar_model, AutoRegressiveModule
from models.backbone import build_backbone
from models.center_point_conv_lstm import CenterPointConvLSTM
from models.center_point_lstm import CenterPointLSTM
from models.perceiver import build_model_perceiver, ObjectDetectionHead
from models.perceiver_with_lstm import PerceiverWithLstm


def build_model(args, input_image_view_size):
	assert 'moving-mnist' in args.dataset.lower()
	num_classes = 10

	if args.model == 'lstm':
		gh, gw = args.grid_size
		num_sensors = gh * gw
		backbone = build_backbone(args, input_image_view_size=input_image_view_size)
		recurrent_module = CenterPointLSTM(
			num_latents=args.num_queries,
			latent_dim=args.hidden_dim,
			feature_channels=backbone.num_channels,
			feature_size=backbone.output_size,
			num_sensors=num_sensors
		)
		detection_head = ObjectDetectionHead(
			num_classes=num_classes,
			latent_dim=args.hidden_dim
		)
	elif args.model == 'conv-lstm':
		gh, gw = args.grid_size
		num_sensors = gh * gw
		backbone = build_backbone(args, input_image_view_size=input_image_view_size)
		recurrent_module = CenterPointConvLSTM(
			num_latents=args.num_queries,
			latent_dim=args.hidden_dim,
			feature_channels=backbone.num_channels,
			feature_size=backbone.output_size,
			num_sensors=num_sensors,
			depth=args.enc_layers,  # depth of net. The shape of the final attention mechanism will be:
			#   depth * ConvLstm
		)
		detection_head = ObjectDetectionHead(
			num_classes=num_classes,
			latent_dim=args.hidden_dim
		)
	elif args.model == 'perceiver-lstm':
		gh, gw = args.grid_size
		num_sensors = gh * gw
		backbone = build_backbone(args, input_image_view_size=input_image_view_size)
		recurrent_module = PerceiverWithLstm(
			num_latents=args.num_queries,
			latent_dim=args.hidden_dim,
			feature_channels=backbone.num_channels,
			feature_size=backbone.output_size,
			num_sensors=num_sensors,
			latent_heads=args.nheads,
			latent_dim_head=args.hidden_dim // args.nheads,
			attn_dropout=args.dropout,
			ff_dropout=args.dropout,
			self_per_cross_attn=args.self_per_cross_attn
		)

		detection_head = ObjectDetectionHead(
			num_classes=num_classes,
			latent_dim=args.hidden_dim
		)
	elif args.model == 'perceiver':
		backbone, recurrent_module, detection_head = build_model_perceiver(
			args, num_classes=num_classes, input_image_view_size=input_image_view_size)
	else:
		raise NotImplementedError(f"Model {args.model} not implemented.")

	model = AutoRegressiveModule(
		backbone=backbone,
		recurrent_module=recurrent_module,
		detection_head=detection_head,
		number_of_views=args.grid_size[0] * args.grid_size[1],
		shuffle_views=args.shuffle_views
	)
	return model

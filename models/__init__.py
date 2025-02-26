from models.autoregressive_module import build_perceiver_ar_model
from models.center_point_lstm import SimpleCenterNetWithLSTM


def build_model(args, input_image_view_size):
    assert 'moving-mnist' in args.dataset.lower()
    num_classes = 10

    if args.model == 'lstm':
        return SimpleCenterNetWithLSTM(num_classes=num_classes, lstm_hidden_size=args.lstm_hidden_size)
    elif args.model == 'perceiver':
        return build_perceiver_ar_model(args, num_classes=num_classes, input_image_view_size=input_image_view_size)

from models.center_point_lstm import SimpleCenterNetWithLSTM
from models.perceiver_ar import build_perceiver_ar_model


def build_model(args):
    assert 'moving-mnist' in args.dataset.lower()
    num_classes = 10

    if args.model == 'lstm':
        return SimpleCenterNetWithLSTM(num_objects=args.num_queries, num_classes=num_classes, lstm_hidden_size=args.lstm_hidden_size)
    elif args.model == 'perceiver':
        return build_perceiver_ar_model(args, num_classes=num_classes)

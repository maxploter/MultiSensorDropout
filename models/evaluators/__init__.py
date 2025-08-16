from .fiftyone_evaluator import FiftyOneEvaluator
from .mot_evaluator import MOTMetricsEvaluator
from .tide_evaluator import CocoJsonEvaluator

def build_evaluators(args, postprocessor):
    """
    Builds and returns a list of evaluators based on the provided arguments.

    Args:
        args: The arguments object containing configuration
        dataset_name: Name of the dataset being evaluated

    Returns:
        List of evaluator objects or None if no evaluators are specified
    """
    evaluators = []

    # Check if any evaluators are specified in the args
    if not hasattr(args, 'evaluators') or not args.evaluators:
        return None

    evaluator_names = args.evaluators

    for name in evaluator_names:
        name = name.strip().lower()

        if name == 'fiftyone':
            CLASS_NAMES = [str(k) for k in range(0, 11)]
            evaluators.append(FiftyOneEvaluator(
                    postprocessor=postprocessor,
                    save_dir= args.output_dir + "/fiftyone/",
                    dataset_name="mmnist-predictions",
                    class_names=CLASS_NAMES,
                    model = args.model
            ))

        elif name == 'tide':
            evaluators.append(CocoJsonEvaluator(
                    postprocessor = postprocessor, output_dir=args.output_dir, checkpoint=args.resume, save_gt=True
            ))
        elif name == 'mot':
            evaluators.append(MOTMetricsEvaluator(
                postprocessor=postprocessor,
            ))

    return evaluators if evaluators else None

# Add necessary imports

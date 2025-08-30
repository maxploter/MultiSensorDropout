import argparse
import fiftyone as fo
import fiftyone.brain as fob
from fiftyone import ViewField as F


def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze model prediction mistakes with FiftyOne."
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Path to the directory of the FiftyOneDataset to load."
    )
    parser.add_argument(
        "--name",
        default="mistakenness-analysis",
        help="A persistent name for the dataset in FiftyOne's database."
    )
    parser.add_argument(
        "--pred_field",
        default="predictions",
        help="The sample field containing your model's predictions."
    )
    parser.add_argument(
        "--gt_field",
        default="ground_truth",
        help="The sample field containing your ground truth labels."
    )
    parser.add_argument(
        "--filter_label",
        default="10",
        help="A specific label to filter out before analysis."
    )
    parser.add_argument(
        "--force_delete",
        action="store_true",
        help="If set, force delete the dataset with the given name before loading/ingesting."
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use (between 0 and 1, default: 1.0)"
    )
    return parser.parse_args()


def main():
    """
    Loads a dataset, computes mistakenness, and launches the FiftyOne App.
    """
    args = get_args()

    # Optionally force delete the dataset if requested
    if args.force_delete and args.name in fo.list_datasets():
        print(f"Force deleting existing dataset '{args.name}'...")
        fo.delete_dataset(args.name)

    # 1. Load the dataset from the FiftyOne database if it exists.
    #    Otherwise, ingest it from the specified directory.
    if args.name in fo.list_datasets():
        print(f"Loading existing dataset '{args.name}'")
        dataset = fo.load_dataset(args.name)
    else:
        print(f"Ingesting new dataset from '{args.dataset_dir}'")
        dataset = fo.Dataset.from_dir(
            dataset_dir=args.dataset_dir,
            dataset_type=fo.types.FiftyOneDataset,
            name=args.name,
            persistent=True,  # Saves the dataset to the database for future runs
        )

    # 2. Create a view filtering out the specified label
    print(f"Filtering labels where label is '{args.filter_label}'...")
    view = dataset.filter_labels(args.pred_field, F("label") != args.filter_label)

    # Apply fraction if less than 1.0
    if args.fraction < 1.0:
        num_samples = int(len(view) * args.fraction)
        print(f"Taking a fraction of the dataset: {num_samples} samples out of {len(view)}")
        view = view.limit(num_samples)

    # 3. Compute mistakenness on the view only if not already present
    if not view.exists("mistakenness"):
        print("Computing mistakenness...")
        fob.compute_mistakenness(
            view,
            pred_field=args.pred_field,
            label_field=args.gt_field,
        )
        view.save()  # Persist the computed field
    else:
        print("Mistakenness already computed. Skipping computation.")

    # 4. Create a new view sorted by the most mistaken predictions
    print("Sorting samples by mistakenness...")
    mistaken_view = view.sort_by("mistakenness", reverse=True)

    # 5. Launch the FiftyOne App with the final, sorted view
    # print("Launching FiftyOne App...")
    # session = fo.launch_app(view=mistaken_view)
    # print("App is live. Press Ctrl+C in your terminal to exit.")

    # 6. Keep the script alive until you close the App tab or terminal
    # session.wait()


if __name__ == "__main__":
    main()
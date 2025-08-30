import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Tuple


def plot_video_frames(dataset_dir: str, video_id: str, frame_range: Tuple[int, int], output_dir: str):
    """
    Reads, plots, and saves a sequence of video frames from a dataset directory.

    Args:
        dataset_dir (str): The path to the directory containing the frame images.
        video_id (str): The identifier for the video, e.g., '0016'.
        frame_range (Tuple[int, int]): The start and end frame numbers (inclusive).
        output_dir (str): The directory where the output plot image will be saved.
    """
    start_frame, end_frame = frame_range

    # --- 1. Find and load the images ---
    images_to_plot = []
    frame_titles = []

    print(f"Searching for frames {start_frame} to {end_frame} for video '{video_id}'...")

    for frame_num in range(start_frame, end_frame + 1):
        expected_filename = f"video_{video_id}_frame_{frame_num:04d}.png"
        file_path = os.path.join(dataset_dir, expected_filename)

        if os.path.exists(file_path):
            try:
                img = mpimg.imread(file_path)
                images_to_plot.append(img)
                frame_titles.append(f"Frame {frame_num}")
            except Exception as e:
                print(f"Could not read file '{file_path}': {e}")
        else:
            print(f"⚠️ Warning: File not found, skipping: {file_path}")

    if not images_to_plot:
        print("\n❌ Error: No images were found for the specified video and frame range.")
        return

    # --- 2. Plot the loaded images ---
    num_images = len(images_to_plot)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))

    if num_images == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images_to_plot, frame_titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    fig.suptitle(f"Frames from Video ID: {video_id}", fontsize=16, y=0.95)
    plt.tight_layout()

    # --- 3. Save the plot to the output directory ---
    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct a descriptive filename
    output_filename = f"video_{video_id}_frames_{start_frame}-{end_frame}.png"
    full_output_path = os.path.join(output_dir, output_filename)

    # Save the figure to the specified path with good resolution
    plt.savefig(full_output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)  # Close the figure to free up memory

    print(f"\n✅ Plot successfully saved to:\n{full_output_path}")


# ==============================================================================
# --- Entry point for the script ---
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to read, plot, and save a sequence of video frames.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="The path to the directory containing the frame images."
    )
    parser.add_argument(
        "--video_id",
        type=str,
        required=True,
        help="The identifier for the video (e.g., '0016')."
    )
    parser.add_argument(
        "--frame_range",
        type=int,
        nargs=2,
        required=True,
        metavar=('START_FRAME', 'END_FRAME'),
        help="The start and end frame numbers (inclusive), e.g., 9 12."
    )
    # NEW ARGUMENT for the output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The path to the directory where the output plot will be saved."
    )

    args = parser.parse_args()

    plot_video_frames(
        dataset_dir=args.dataset_dir,
        video_id=args.video_id,
        frame_range=tuple(args.frame_range),
        output_dir=args.output_dir
    )
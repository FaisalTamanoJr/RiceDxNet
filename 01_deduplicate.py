"""
Run once to download and remove the duplicates from the dataset
"""

from pathlib import Path

import fastdup as fd
import kagglehub  # basic usage: https://github.com/Kaggle/kagglehub

data_path = Path("data/")
image_path = (
    data_path
    / "philippine_rice_diseases"
    / "extra_resized_raw_images"
    / "extra_resized_raw_images"
)
# Check if dataset is locally available
if image_path.is_dir():
    print(f"{image_path} directory exists")
else:
    print(f"Did not find {image_path} directory, creating it...")
    image_path.mkdir(parents=True, exist_ok=True)

    # Download dataset
    kagglehub.dataset_download(
        "shrupyag001/philippines-rice-diseases", output_dir=image_path
    )

fd.remove_duplicates(
    str(image_path), dry_run=False
)  # remove duplicates at 96% similiarity (permanently delete duplicates)

from collections import Counter
from pathlib import Path

import torch
import torch.nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, v2

# SETUP
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32  # Change and test to see which is better, trial and error

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

print(DEVICE)

# LOAD DATASET
data_path = Path("data/")
image_path = (
    data_path
    / "philippine_rice_diseases"
    / "extra_resized_raw_images"
    / "extra_resized_raw_images"
)

ds = ImageFolder(image_path)

# STRATIFIED DATASET SPLIT: 80/10/10
targets = ds.targets
idx = list(range(len(ds)))
train_idx, holdout_idx = train_test_split(
    idx, test_size=0.20, stratify=targets, random_state=SEED
)

holdout_targets = []
for i in holdout_idx:
    holdout_targets.append(targets[i])
val_idx, test_idx = train_test_split(
    holdout_idx, test_size=0.5, stratify=holdout_targets, random_state=SEED
)

# Create PyTorch Subsets
train_dataset = Subset(ds, train_idx)
val_dataset = Subset(ds, val_idx)
test_dataset = Subset(ds, test_idx)


# FIND CLASS COUNT
def class_count_dict(subset):
    class_names = subset.dataset.classes
    counts = Counter([subset.dataset.targets[i] for i in subset.indices])

    named_counts = {}
    for idx, count in counts.items():
        named_counts[class_names[idx]] = count

    return named_counts


"""
# Print class counts in all sets
train_counts = class_count_dict(train_dataset)
val_counts = class_count_dict(val_dataset)
test_counts = class_count_dict(test_dataset)

# Print train, validation, and test
for key in sorted(train_counts.keys()):
    print(
        f"{key} | train: {train_counts[key]}, validation: {val_counts[key]}, test: {test_counts[key]}"
    )
"""

# IMAGE TRANSFORMS
normalize = (v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),)

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToImage(),  # Convert Image to Tensor
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_test_transform = transforms.Compose(
    [
        transforms.Resize(224, 224),
        transforms.ToImage(),  # Convert Image to Tensor
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# TODO: RiceNet Class

# TODO: RiceNet Loader

# TODO: RiceNet freeze

# TODO: RiceNet finetune

# TODO: Ricenet small cnn

# NOTE: Separate into modules when done and rerun experiments

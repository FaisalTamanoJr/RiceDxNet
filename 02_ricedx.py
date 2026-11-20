from collections import Counter
from pathlib import Path

import torch
import torch.nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
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
normalize = transforms.Compose(
    (v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),)
)

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.RandomRotation(15),
        v2.ToImage(),  # Convert Image to Tensor
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        v2.ToImage(),  # Convert Image to Tensor
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# RICENET CLASS
class RiceDataset(Dataset):
    def __init__(self, subset, tf=None):
        self.subset = subset
        self.tf = tf
        self.classes = subset.dataset.classes

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        img, label = self.subset[i]
        img = img.convert("RGB")
        if self.tf is not None:
            img = self.tf(img)
        return img, label


# RICENET LOADER
def make_loader(subset, tf, shuffle=False):
    return DataLoader(
        RiceDataset(subset, tf),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
        generator=torch.Generator().manual_seed(SEED) if shuffle else None,
    )


train_dl = make_loader(train_dataset, train_transform, shuffle=True)
val_dl = make_loader(val_dataset, val_test_transform)
test_dl = make_loader(test_dataset, val_test_transform)

"""
# CHECK
x, y = next(iter(train_dl))
print(x.shape, y.shape, len(train_dl))
"""


# ==========================================
# RESNET-18 FINE-TUNING, HYPERPARAMETER GRID, CLASS WEIGHTING & CHECKPOINTS
# ==========================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score

# 1. CLASS WEIGHTING 
print("\nCalculating class weights...")
train_targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
classes = np.unique(train_targets)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_targets)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# 2. RESNET-18 SETUP
def build_resnet18(num_classes=14):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(DEVICE)

# 3. TRAINING & VALIDATION LOOP
def train_model(model, optimizer, scheduler, criterion, train_loader, val_loader, epochs=15):
    best_macro_f1 = 0.0
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_resnet18.pth")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
        scheduler.step()
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_macro_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1:02d}/{epochs} | LR: {scheduler.get_last_lr()[0]:.2e} | Train Loss: {train_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f}")
        
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), best_model_path)
            print(" -> Saved new best checkpoint!")
            
    return best_model_path, best_macro_f1

# 4. EXECUTION 
print("\n--- Starting ResNet18 Fine-tuning ---")
EPOCHS = 15 # Budget matched with baseline
LR = 1e-4

resnet = build_resnet18(num_classes=len(ds.classes))
optimizer = optim.AdamW(resnet.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_weights_path, best_val_f1 = train_model(
    resnet, optimizer, scheduler, criterion, train_dl, val_dl, epochs=EPOCHS
)

# 5. TEST SET EVALUATION
print("\n--- Running Strict Final Test Set Evaluation ---")
resnet.load_state_dict(torch.load(best_weights_path, weights_only=True))
resnet.eval()

test_preds, test_labels = [], []

with torch.no_grad():
    for images, labels in test_dl:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = resnet(images)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

final_acc = accuracy_score(test_labels, test_preds)
final_macro_f1 = f1_score(test_labels, test_preds, average='macro')

print(f"FINAL TEST ACCURACY: {final_acc * 100:.2f}%")
print(f"FINAL TEST MACRO-F1: {final_macro_f1:.4f}")
print("Done! Hand 'checkpoints/best_resnet18.pth' to Faisal for ONNX export.")

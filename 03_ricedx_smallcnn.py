"""RiceDxNet - SmallCNN baseline"""

import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # only saving figures, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

# ----------------------------------------------------------------------------
# SETUP
# ----------------------------------------------------------------------------
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15  # keep equal to the ResNet18 runs
LRS = [1e-3, 3e-4, 1e-4]  # small sweep so the baseline cannot be called under-tuned
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
RUNS_DIR = Path("runs")

# Success criteria from the proposal
MIN_ACC, MIN_MACRO_F1, MIN_CLASS_RECALL = 0.85, 0.80, 0.60

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("data/")
IMAGE_PATH = (
    data_path
    / "philippine_rice_diseases"
    / "extra_resized_raw_images"
    / "extra_resized_raw_images"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# DATA  (stratified 80/10/10, seed 42)
# ----------------------------------------------------------------------------
NORM = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = v2.Compose(
    [
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(0.3, 0.3, 0.3),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        NORM,
    ]
)

# Used for validation AND test: no random augmentation
eval_transform = v2.Compose(
    [
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        NORM,
    ]
)


def get_data():
    # One bare ImageFolder just to get the labels and split the indices.
    ds = ImageFolder(IMAGE_PATH)
    targets = ds.targets
    classes = ds.classes
    print(f"Found {len(ds)} images in {len(classes)} classes")

    idx = list(range(len(ds)))
    train_idx, holdout_idx = train_test_split(
        idx, test_size=0.20, stratify=targets, random_state=SEED
    )
    holdout_targets = [targets[i] for i in holdout_idx]
    val_idx, test_idx = train_test_split(
        holdout_idx, test_size=0.5, stratify=holdout_targets, random_state=SEED
    )

    # A Subset shares its parent's transform, so each split gets its own
    # ImageFolder: augmentation for train, none for val/test.
    train_ds = Subset(ImageFolder(IMAGE_PATH, transform=train_transform), train_idx)
    val_ds = Subset(ImageFolder(IMAGE_PATH, transform=eval_transform), val_idx)
    test_ds = Subset(ImageFolder(IMAGE_PATH, transform=eval_transform), test_idx)
    print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    pin = DEVICE == "cuda"
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin)
    return train_loader, val_loader, test_loader, classes


# ----------------------------------------------------------------------------
# SMALL CNN (the baseline)
# ----------------------------------------------------------------------------
class SmallCNN(nn.Module):

    def __init__(self, num_classes=14):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, criterion=None):
    """Returns true labels, predicted labels, and the average loss (if criterion given)."""
    model.eval()
    ys, ps, total_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        if criterion is not None:
            total_loss += criterion(out, y).item() * x.size(0)
        n += x.size(0)
        ys.append(y.cpu())
        ps.append(out.argmax(1).cpu())
    loss = total_loss / n if criterion is not None else None
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), loss


def compute_metrics(y_true, y_pred, class_names):
    # labels=... forces all classes to appear even if one is never predicted
    labels = list(range(len(class_names)))
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = pd.DataFrame(
        {"class": class_names, "precision": p, "recall": r, "f1": f1, "support": support}
    )
    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "min_class_recall": float(r.min()),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


@torch.no_grad()
def measure_latency_ms(model, runs=50):
    """Average time to classify ONE 224x224 image (batch size 1)."""
    model.eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    for _ in range(10):  # warm-up
        model(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        model(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000


def check_criteria(m):
    """Success-criteria checks that need only the test metrics.
    (The baseline, robustness and resolution criteria need more runs.)"""
    weak = m["per_class"][m["per_class"]["recall"] < MIN_CLASS_RECALL]["class"].tolist()
    return {
        f"accuracy >= {MIN_ACC}": m["accuracy"] >= MIN_ACC,
        f"macro-F1 >= {MIN_MACRO_F1}": m["macro_f1"] >= MIN_MACRO_F1,
        f"every class recall >= {MIN_CLASS_RECALL}": len(weak) == 0,
        "classes below recall threshold": weak,
    }


def save_confusion_matrix(cm, class_names, path, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_and_save(model, loader, class_names, run_dir, name):
    """Score any model on a loader and write everything to run_dir.
    Use this on the TEST loader once, for the final chosen models only."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred, _ = predict(model, loader)
    m = compute_metrics(y_true, y_pred, class_names)
    crit = check_criteria(m)

    summary = {
        "model": name,
        "accuracy": m["accuracy"],
        "macro_precision": m["macro_precision"],
        "macro_recall": m["macro_recall"],
        "macro_f1": m["macro_f1"],
        "min_class_recall": m["min_class_recall"],
        "params": count_params(model),
        "latency_ms_per_image": round(measure_latency_ms(model), 3),
        "device": DEVICE,
        "criteria": crit,
        "per_class": m["per_class"].to_dict("records"),
    }
    (run_dir / "test_metrics.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    )
    m["per_class"].to_csv(run_dir / "per_class.csv", index=False)
    (run_dir / "test_report.txt").write_text(
        classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                              target_names=class_names, digits=3, zero_division=0)
    )
    np.savetxt(run_dir / "confusion_matrix.csv", m["confusion_matrix"], fmt="%d", delimiter=",")
    save_confusion_matrix(m["confusion_matrix"], class_names,
                          run_dir / "confusion_matrix.png", f"{name} - test set")
    return summary, m

# ----------------------------------------------------------------------------
# TRAINING  (validation only, no test)
# ----------------------------------------------------------------------------
def train_smallcnn(lr, train_loader, val_loader, classes, run_name):
    set_seed(SEED)
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = SmallCNN(len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history, best_f1, best_epoch = [], -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)
        scheduler.step()

        y_true, y_pred, val_loss = predict(model, val_loader, criterion)
        val = compute_metrics(y_true, y_pred, classes)
        history.append({
            "epoch": epoch,
            "train_loss": round(total_loss / seen, 4),
            "train_acc": round(correct / seen, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val["accuracy"], 4),
            "val_macro_f1": round(val["macro_f1"], 4),
            "seconds": round(time.time() - start, 1),
        })
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        improved = val["macro_f1"] > best_f1
        if improved:  # keep the checkpoint with the best validation macro-F1
            best_f1, best_epoch = val["macro_f1"], epoch
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "cfg": {"model": "smallcnn", "lr": lr, "epochs": EPOCHS,
                                "batch_size": BATCH_SIZE, "img_size": IMG_SIZE}},
                       run_dir / "best.pt")
        h = history[-1]
        print(f"[{run_name}] epoch {epoch:2d} | train loss {h['train_loss']:.3f} "
              f"acc {h['train_acc']:.3f} | val loss {h['val_loss']:.3f} "
              f"acc {h['val_acc']:.3f} F1 {h['val_macro_f1']:.3f} "
              f"{'*saved*' if improved else ''}")
    return run_dir, best_f1, best_epoch


def main():
    print("device:", DEVICE)
    train_loader, val_loader, test_loader, classes = get_data()
    print("SmallCNN parameters:", f"{count_params(SmallCNN(len(classes))):,}")

    sweep = []
    for lr in LRS:
        run_name = f"smallcnn_lr{lr:g}_ep{EPOCHS}"
        _, f1, ep = train_smallcnn(lr, train_loader, val_loader, classes, run_name)
        sweep.append({"run": run_name, "lr": lr, "best_epoch": ep, "best_val_macro_f1": f1})
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(RUNS_DIR / "smallcnn_sweep.csv", index=False)
    print("\nValidation sweep:\n", sweep_df.to_string(index=False))

    best_run = sweep_df.loc[sweep_df["best_val_macro_f1"].idxmax(), "run"]
    print(f"\nChosen run (by validation macro-F1): {best_run}")
    ckpt = torch.load(RUNS_DIR / best_run / "best.pt", map_location=DEVICE)
    model = SmallCNN(len(classes)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    summary, m = evaluate_and_save(model, test_loader, classes, RUNS_DIR / best_run,
                                   "SmallCNN")
    print(f"\nTEST  acc {summary['accuracy']:.4f} | macro-F1 {summary['macro_f1']:.4f} | "
          f"macro-P {summary['macro_precision']:.4f} | macro-R {summary['macro_recall']:.4f} | "
          f"min class recall {summary['min_class_recall']:.2f}")
    print(f"params {summary['params']:,} | {summary['latency_ms_per_image']} ms/image on {DEVICE}")
    print("Criteria:", summary["criteria"])
    print("\nPer-class (test):\n", m["per_class"].round(3).to_string(index=False))

if __name__ == "__main__":
    main()

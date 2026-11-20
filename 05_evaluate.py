"""
RiceDxNet - 05_evaluate.py
Evaluate trained ResNet18 and/or SmallCNN checkpoints on the reproducible test split.
Exports metrics, confusion matrices, predictions, and Grad-CAM visualizations.

Run from the RiceDxNet project directory:
    python 05_evaluate.py
    python 05_evaluate.py --resnet checkpoints/best_resnet18.pth
    python 05_evaluate.py --smallcnn runs/smallcnn_lr0.001_ep10/best.pt
    python 05_evaluate.py --model resnet18 --num-gradcam 12
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms import v2


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "philippine_rice_diseases" / "extra_resized_raw_images" / "extra_resized_raw_images"
OUT_DIR = ROOT / "runs" / "evaluation"
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NORM = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
EVAL_TRANSFORM = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    NORM,
])


class SmallCNN(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_test_loader():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATA_PATH}\n"
            "Place the dataset in the same folder structure used by 02_ricedx.py and 03_ricedx_smallcnn.py."
        )

    dataset = ImageFolder(DATA_PATH)
    targets = dataset.targets
    indices = list(range(len(dataset)))

    # Same stratified split used in 02_ricedx.py and 03_ricedx_smallcnn.py.
    _, holdout_indices = train_test_split(
        indices, test_size=0.20, stratify=targets, random_state=SEED
    )
    holdout_targets = [targets[i] for i in holdout_indices]
    _, test_indices = train_test_split(
        holdout_indices, test_size=0.5, stratify=holdout_targets, random_state=SEED
    )

    test_dataset = Subset(ImageFolder(DATA_PATH, transform=EVAL_TRANSFORM), test_indices)
    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )
    return loader, dataset.classes, test_dataset


def load_resnet(checkpoint_path, num_classes):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    return model.to(DEVICE)


def load_smallcnn(checkpoint_path, num_classes):
    model = SmallCNN(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    return model.to(DEVICE)


@torch.no_grad()
def infer(model, loader):
    model.eval()
    y_true, y_pred, probabilities = [], [], []
    for images, labels in loader:
        logits = model(images.to(DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())
        probabilities.extend(probs.tolist())
    return np.asarray(y_true), np.asarray(y_pred), np.asarray(probabilities)


def evaluate(model, loader, classes, model_name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred, probs = infer(model, loader)
    labels = list(range(len(classes)))

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    per_class = pd.DataFrame({
        "class": classes,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    summary = {
        "model": model_name,
        "device": DEVICE,
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "min_class_recall": float(recall.min()),
        "per_class": per_class.to_dict("records"),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    np.savetxt(output_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    (output_dir / "classification_report.txt").write_text(
        classification_report(
            y_true, y_pred, labels=labels, target_names=classes,
            digits=4, zero_division=0
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{model_name} - Test Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    threshold = cm.max() / 2 if cm.max() else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            if cm[row, col]:
                ax.text(col, row, str(cm[row, col]), ha="center", va="center",
                        color="white" if cm[row, col] > threshold else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # Export per-image predictions and confidence.
    rows = []
    for i, (true_idx, pred_idx, p) in enumerate(zip(y_true, y_pred, probs)):
        rows.append({
            "sample_index": i,
            "true_class": classes[int(true_idx)],
            "predicted_class": classes[int(pred_idx)],
            "correct": bool(true_idx == pred_idx),
            "confidence": float(p[int(pred_idx)]),
        })
    pd.DataFrame(rows).to_csv(output_dir / "predictions.csv", index=False)

    print(f"\n[{model_name}] Test results")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Macro precision: {macro_p:.4f}")
    print(f"Macro recall   : {macro_r:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Results saved  : {output_dir}")
    return summary


class GradCAM:
    """Grad-CAM implementation using forward and backward hooks."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = logits[:, target_class]
        score.backward()

        # Average gradients spatially to get channel weights.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().detach().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, target_class

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def get_target_layer(model, model_type):
    if model_type == "resnet18":
        return model.layer4[-1]
    # Last convolution in the final SmallCNN feature block.
    return model.features[-1][0]


def tensor_to_rgb(tensor):
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = np.clip(image * std + mean, 0, 1)
    return image


def export_gradcam(model, model_type, test_subset, classes, output_dir, num_images=12):
    output_dir.mkdir(parents=True, exist_ok=True)
    cam_engine = GradCAM(model, get_target_layer(model, model_type))
    count = min(num_images, len(test_subset))
    indices = np.linspace(0, len(test_subset) - 1, count, dtype=int)

    manifest = []
    for position, index in enumerate(indices):
        image_tensor, true_label = test_subset[index]
        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        cam, predicted_idx = cam_engine.generate(input_tensor)
        rgb = tensor_to_rgb(image_tensor)
        heatmap = plt.get_cmap("jet")(cam)[..., :3]
        overlay = np.clip(0.55 * rgb + 0.45 * heatmap, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rgb)
        axes[0].set_title("Original")
        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")

        true_name = classes[int(true_label)]
        pred_name = classes[int(predicted_idx)]
        fig.suptitle(f"True: {true_name} | Predicted: {pred_name}")
        fig.tight_layout()
        filename = f"gradcam_{position+1:03d}_true-{true_name}_pred-{pred_name}.png"
        fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

        manifest.append({
            "file": filename,
            "test_subset_index": int(index),
            "true_class": true_name,
            "predicted_class": pred_name,
            "correct": bool(int(true_label) == int(predicted_idx)),
        })

    cam_engine.close()
    pd.DataFrame(manifest).to_csv(output_dir / "gradcam_manifest.csv", index=False)
    print(f"Grad-CAM visualizations exported to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RiceDxNet models and export Grad-CAM.")
    parser.add_argument("--model", choices=["resnet18", "smallcnn", "both"], default="both")
    parser.add_argument("--resnet", default=str(ROOT / "checkpoints" / "best_resnet18.pth"))
    parser.add_argument("--smallcnn", default=None, help="Path to SmallCNN best.pt checkpoint")
    parser.add_argument("--num-gradcam", type=int, default=12)
    args = parser.parse_args()

    set_seed(SEED)
    test_loader, classes, test_subset = get_test_loader()
    print(f"Device: {DEVICE}")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Test images: {len(test_subset)}")

    models = []
    if args.model in ("resnet18", "both"):
        path = Path(args.resnet)
        if path.exists():
            models.append(("ResNet18", "resnet18", load_resnet(path, len(classes))))
        else:
            print(f"WARNING: ResNet checkpoint not found; skipping: {path}")

    if args.model in ("smallcnn", "both"):
        path = Path(args.smallcnn) if args.smallcnn else None
        if path is None:
            # Prefer the sweep-selected run, if available; otherwise choose the
            # highest validation macro-F1 row from the sweep CSV.
            sweep = ROOT / "runs" / "smallcnn_sweep.csv"
            if sweep.exists():
                df = pd.read_csv(sweep)
                path = ROOT / "runs" / df.loc[df["best_val_macro_f1"].idxmax(), "run"] / "best.pt"
            else:
                candidates = sorted((ROOT / "runs").glob("smallcnn_*/best.pt"))
                path = candidates[0] if candidates else None
        if path and path.exists():
            models.append(("SmallCNN", "smallcnn", load_smallcnn(path, len(classes))))
        else:
            print("WARNING: SmallCNN checkpoint not found; skipping.")

    if not models:
        raise SystemExit("No model checkpoint found. Train a model before running evaluation.")

    all_summaries = {}
    for name, model_type, model in models:
        model.eval()
        model_dir = OUT_DIR / name
        summary = evaluate(model, test_loader, classes, name, model_dir)
        all_summaries[name] = summary
        export_gradcam(
            model, model_type, test_subset, classes,
            model_dir / "gradcam", num_images=args.num_gradcam
        )

    with open(OUT_DIR / "all_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nEvaluation complete. Outputs are under: {OUT_DIR}")


if __name__ == "__main__":
    main()

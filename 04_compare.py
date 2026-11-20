
import argparse
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # so the relative data/ and runs/ paths always resolve


def load_module(name, filename):
    """Import a file whose name starts with a digit (can't use a normal import)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_smallcnn_ckpt(runs_dir):
    """Pick the SmallCNN run with the best validation macro-F1."""
    sweep_csv = runs_dir / "smallcnn_sweep.csv"
    if sweep_csv.exists():
        df = pd.read_csv(sweep_csv)
        best_run = df.loc[df["best_val_macro_f1"].idxmax(), "run"]
        path = runs_dir / best_run / "best.pt"
        if path.exists():
            return path
    candidates = sorted(runs_dir.glob("smallcnn_*/best.pt"))
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet", default=str(ROOT / "checkpoints" / "best_resnet18.pth"))
    ap.add_argument("--smallcnn", default=None)
    args = ap.parse_args()

    # 03 provides the shared data loading, SmallCNN class and evaluation code
    sc = load_module("smallcnn_script", "03_ricedx_smallcnn.py")
    DEVICE = sc.DEVICE
    runs_dir = sc.RUNS_DIR
    runs_dir.mkdir(exist_ok=True)

    # Shared test loader (same split code and seed in both training scripts)
    _, _, test_loader, classes = sc.get_data()

    # ResNet18
    resnet_path = Path(args.resnet)
    if not resnet_path.exists():
        raise SystemExit(f"ResNet18 checkpoint not found: {resnet_path}")
    resnet_model = resnet18(weights=None)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, len(classes))
    resnet_model.load_state_dict(
        torch.load(resnet_path, map_location=DEVICE, weights_only=True)
    )
    resnet_model = resnet_model.to(DEVICE)

    # SmallCNN
    small_path = Path(args.smallcnn) if args.smallcnn else find_smallcnn_ckpt(runs_dir)
    if small_path is None or not small_path.exists():
        raise SystemExit(
            "SmallCNN checkpoint not found. Run 03_ricedx_smallcnn.py first, "
            "or pass --smallcnn path/to/best.pt"
        )
    ckpt = torch.load(small_path, map_location=DEVICE, weights_only=True)
    smallcnn_model = sc.SmallCNN(len(classes)).to(DEVICE)
    smallcnn_model.load_state_dict(ckpt["model_state"])

    # Evaluation and comparison
    out_dir = runs_dir / "comparison"
    summaries = {}
    per_class = {}
    for name, model in [("ResNet18", resnet_model), ("SmallCNN", smallcnn_model)]:
        summary, m = sc.evaluate_and_save(model, test_loader, classes, out_dir / name, name)
        summaries[name] = summary
        per_class[name] = m["per_class"].set_index("class")

    # Summary table
    rows = [
        ("Accuracy", "accuracy"),
        ("Macro precision", "macro_precision"),
        ("Macro recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
        ("Min class recall", "min_class_recall"),
        ("Parameters", "params"),
        (f"Latency ms/image ({DEVICE})", "latency_ms_per_image"),
    ]
    table = pd.DataFrame(
        {name: [s[key] for _, key in rows] for name, s in summaries.items()},
        index=[r[0] for r in rows],
    )

    # Per-class comparison
    a, b = list(summaries)
    pc = pd.DataFrame(
        {
            "support": per_class[a]["support"],
            f"recall {a}": per_class[a]["recall"],
            f"recall {b}": per_class[b]["recall"],
            f"f1 {a}": per_class[a]["f1"],
            f"f1 {b}": per_class[b]["f1"],
        }
    )

    # Proposal success criteria
    crit = {name: s["criteria"] for name, s in summaries.items()}

    pd.set_option("display.width", 200)
    print("\n=== OVERALL (test set) ===")
    print(table.round(4).to_string())
    print("\n=== PER-CLASS ===")
    print(pc.round(3).to_string())
    print("\n=== SUCCESS CRITERIA ===")
    for name, c in crit.items():
        print(name, "->", c)

    better = max(summaries, key=lambda k: summaries[k]["macro_f1"])
    print(f"\nHigher macro-F1: {better}")

    table.to_csv(out_dir / "comparison_overall.csv")
    pc.to_csv(out_dir / "comparison_per_class.csv")
    (out_dir / "comparison.json").write_text(
        json.dumps({"overall": table.to_dict(), "criteria": crit}, indent=2, default=str)
    )
    print(f"\nSaved results to {out_dir}/")


if __name__ == "__main__":
    main()
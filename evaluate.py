"""
Evaluation Script for DenWide-Net on the Test Set.

Usage:
    python evaluate.py --model_path checkpoints/denwidenet_best.pth
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_curve
from tqdm import tqdm

import config
from models import build_denwidenet
from dataset import get_data_loaders
from utils import (
    calculate_metrics,
    print_evaluation_summary,
    plot_confusion_matrix,
    plot_roc_curve
)


def evaluate_test_set(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets, _ in tqdm(test_loader, desc="Evaluating Test Set"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = calculate_metrics(all_targets, all_preds, all_probs)
    return metrics, all_targets, all_probs[:, 1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DenWide-Net model on Test Set.")
    parser.add_argument("--model_path", type=str, default=str(config.BEST_MODEL_PATH), help="Path to checkpoint .pth file")
    parser.add_argument("--data_dir", type=str, default=str(config.DATA_DIR), help="Path to MRI dataset root")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    model_file = Path(args.model_path)
    if not model_file.exists():
        print(f"[Error] Checkpoint not found at {model_file}. Please run train.py first.")
        sys.exit(1)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Device] Using: {device}")

    # Load Data Loaders
    _, _, test_loader = get_data_loaders(
        args.data_dir,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE
    )

    if len(test_loader.dataset) == 0:
        print(f"[Error] No test samples found in {args.data_dir}/test.")
        sys.exit(1)

    # Load Model Weights
    print(f"[Model] Loading checkpoint from {model_file}...")
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    saved_config = checkpoint.get("config", {})
    growth_rate = saved_config.get("growth_rate", config.GROWTH_RATE)
    widen_factor = saved_config.get("widen_factor", config.WIDEN_FACTOR)

    model = build_denwidenet(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        growth_rate=growth_rate,
        widen_factor=widen_factor,
        dropout_rate=config.DROPOUT_RATE,
        use_attention=config.USE_ATTENTION
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print("[Model] Weights successfully loaded!")

    # Evaluate
    metrics, y_true, y_probs = evaluate_test_set(model, test_loader, device)
    print_evaluation_summary(metrics, prefix="DenWide-Net Test Set")

    # Plot & Save Confusion Matrix
    cm_path = config.OUTPUT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), config.CLASS_NAMES, cm_path)
    print(f"[Plot] Confusion matrix saved to: {cm_path}")

    # Plot & Save ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_path = config.OUTPUT_DIR / "roc_curve.png"
    plot_roc_curve(fpr, tpr, metrics["roc_auc"], roc_path)
    print(f"[Plot] ROC curve saved to: {roc_path}")


if __name__ == "__main__":
    main()

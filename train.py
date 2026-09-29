"""
Training Script for DenWide-Net on Brain MRI Scans for ASD Detection.

Usage:
    python train.py --epochs 30 --batch_size 16 --lr 0.0001
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from models import build_denwidenet
from dataset import get_data_loaders, generate_synthetic_mri_dataset
from utils import (
    calculate_metrics,
    print_evaluation_summary,
    get_optimizer_and_scheduler,
    plot_training_curves
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets, _ in tqdm(loader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping to stabilize training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_targets))
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets, _ in tqdm(loader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_targets, all_preds, np.array(all_probs))
    metrics["loss"] = epoch_loss
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train DenWide-Net for ASD Classification from Brain MRI.")
    parser.add_argument("--data_dir", type=str, default=str(config.DATA_DIR), help="Path to MRI dataset root")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--growth_rate", type=int, default=config.GROWTH_RATE, help="DenseNet growth rate")
    parser.add_argument("--widen_factor", type=int, default=config.WIDEN_FACTOR, help="WideResNet widening factor (k)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--auto_generate_data", action="store_true", default=True, help="Generate synthetic data if directory is empty")
    args = parser.parse_args()

    # Determine Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # Verify Data Directory
    data_path = Path(args.data_dir)
    train_asd = data_path / "train" / "ASD"
    if not train_asd.exists() or len(list(train_asd.glob("*.*"))) == 0:
        if args.auto_generate_data:
            print("[Info] MRI dataset directory is empty. Generating synthetic brain MRI scans for training...")
            generate_synthetic_mri_dataset(data_path, num_train=60, num_val=20, num_test=20)
        else:
            print(f"[Error] Dataset not found in {data_path}. Please populate data/train/ASD and data/train/TC.")
            sys.exit(1)

    # Data Loaders
    print("[Data] Initializing MRI data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_path,
        batch_size=args.batch_size,
        image_size=config.IMAGE_SIZE
    )
    print(f"[Data] Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples | Test: {len(test_loader.dataset)} samples")

    # Build DenWide-Net Model
    print(f"[Model] Building DenWide-Net (growth_rate={args.growth_rate}, widen_factor={args.widen_factor})...")
    model = build_denwidenet(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        growth_rate=args.growth_rate,
        widen_factor=args.widen_factor,
        dropout_rate=config.DROPOUT_RATE,
        use_attention=config.USE_ATTENTION
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total Trainable Parameters: {total_params:,}")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(
        model,
        optimizer_type="adamw",
        learning_rate=args.lr,
        weight_decay=config.WEIGHT_DECAY,
        num_epochs=args.epochs
    )

    # Tracking History
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": []
    }

    best_val_auc = 0.0
    best_val_acc = 0.0

    print("\n[Start Training] Commencing DenWide-Net training loop...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_auc = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else 0.0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        elapsed = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | "
              f"Val AUC: {val_auc:.3f} | Time: {elapsed:.1f}s")

        # Save Best Checkpoint based on Validation AUC or Accuracy
        if val_auc > best_val_auc or (val_auc == best_val_auc and val_acc > best_val_acc):
            best_val_auc = val_auc
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": {
                    "growth_rate": args.growth_rate,
                    "widen_factor": args.widen_factor,
                    "dropout_rate": config.DROPOUT_RATE
                }
            }, config.BEST_MODEL_PATH)
            print(f"  --> Saved new best checkpoint to {config.BEST_MODEL_PATH} (AUC: {best_val_auc:.4f}, Acc: {best_val_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n[Finished] Total Training Time: {total_time / 60:.2f} minutes")

    # Save Last Checkpoint
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "history": history
    }, config.LAST_MODEL_PATH)

    # Save Training Curves Plot
    curve_path = config.OUTPUT_DIR / "training_curves.png"
    plot_training_curves(history, curve_path)
    print(f"[Plot] Training curves saved to {curve_path}")


if __name__ == "__main__":
    main()

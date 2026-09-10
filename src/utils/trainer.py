"""
Training loop for Medical Image Diagnosis Assistant.
Includes MLflow tracking, early stopping, and model checkpointing.

Usage:
    python src/utils/trainer.py --config configs/config.yaml
"""

import sys
import os
from pathlib import Path

# Ensure project root is in path so 'src' package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force unbuffered stdout so logs appear immediately
sys.stdout.reconfigure(line_buffering=True)

import argparse
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models

from src.data.dataset import ChestXRayDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.efficientnet import EfficientNetDiagnosisModel
from src.models.cnn_model import CustomCNN
from src.utils.metrics import compute_metrics


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Data Loaders ───────────────────────────────────────────────────────────────
def build_dataloaders(config: dict):
    data_cfg = config["data"]
    train_cfg = config["training"]

    base_dir = Path(data_cfg["raw_dir"])
    img_size = data_cfg["image_size"]

    train_dataset = ChestXRayDataset(
        root_dir=base_dir / "train",
        transform=get_train_transforms(img_size),
        split="train",
    )
    val_dataset = ChestXRayDataset(
        root_dir=base_dir / "val",
        transform=get_val_transforms(img_size),
        split="val",
    )
    test_dataset = ChestXRayDataset(
        root_dir=base_dir / "test",
        transform=get_val_transforms(img_size),
        split="test",
    )

    # Weighted sampler to handle class imbalance
    class_weights = train_dataset.get_class_weights()
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # pin_memory is not supported on MPS
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        sampler=sampler,
        num_workers=data_cfg["num_workers"],
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


# ── Model Builder ──────────────────────────────────────────────────────────────
def build_model(config: dict, device: torch.device) -> nn.Module:
    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]
    dropout = config["model"]["dropout"]

    if model_name == "custom_cnn":
        model = CustomCNN(num_classes=num_classes, dropout=dropout)
    elif model_name in ("efficientnet_b3", "efficientnet"):
        model = EfficientNetDiagnosisModel(
            num_classes=num_classes,
            pretrained=config["model"]["pretrained"],
            dropout=dropout,
            freeze_backbone=config["model"]["freeze_backbone"],
        )
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=config["model"]["pretrained"])
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


# ── Training & Validation ──────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(
                f"  [Epoch {epoch}] Batch {batch_idx+1}/{len(loader)} "
                f"| Loss: {loss.item():.4f} | Acc: {correct/total:.4f}"
            )

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = total_loss / len(loader)
    metrics["accuracy"] = correct / total
    return metrics


# ── Main Training Loop ─────────────────────────────────────────────────────────
def train(config: dict):
    set_seed(config["training"]["seed"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n🚀 Training on: {device}", flush=True)
    print(f"📦 Model: {config['model']['name']}\n", flush=True)

    # Directories
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Data
    print("📂 Loading datasets...")
    train_loader, val_loader, test_loader = build_dataloaders(config)

    # Model
    model = build_model(config, device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Trainable parameters: {total_params:,}\n")

    # Loss — weighted for class imbalance
    train_dataset = train_loader.dataset
    class_weights = torch.tensor(
        train_dataset.get_class_weights(), dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
    )

    # MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    best_val_auc = 0.0
    patience_counter = 0
    patience = config["training"]["patience"]

    with mlflow.start_run(run_name=config["model"]["name"]):
        # Log hyperparameters
        mlflow.log_params({
            "model": config["model"]["name"],
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "lr": config["training"]["learning_rate"],
            "image_size": config["data"]["image_size"],
        })

        for epoch in range(1, config["training"]["epochs"] + 1):
            t0 = time.time()
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{config['training']['epochs']}")
            print(f"{'='*60}")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )
            val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            val_auc = val_metrics.get("auc_roc", 0.0)
            elapsed = time.time() - t0

            print(f"\n📊 Epoch {epoch} Summary ({elapsed:.1f}s):")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"   Val   Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"   Val AUC-ROC: {val_auc:.4f} | Val F1: {val_metrics.get('f1', 0):.4f}")

            # MLflow logging
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_auc": val_auc,
                "val_f1": val_metrics.get("f1", 0),
            }, step=epoch)

            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                save_path = model_dir / "best_model.pth"
                torch.save(model.state_dict(), save_path)
                print(f"   ✅ New best model saved (AUC: {best_val_auc:.4f})")
            else:
                patience_counter += 1
                print(f"   ⏳ No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered after {epoch} epochs.")
                break

        # Final test evaluation
        print(f"\n{'='*60}")
        print("🏁 Final Test Evaluation")
        print(f"{'='*60}")
        model.load_state_dict(torch.load(model_dir / "best_model.pth", map_location=device))
        test_metrics = evaluate(model, test_loader, criterion, device)

        print(f"   Test Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"   Test AUC-ROC:   {test_metrics.get('auc_roc', 0):.4f}")
        print(f"   Test F1-Score:  {test_metrics.get('f1', 0):.4f}")
        print(f"   Test Precision: {test_metrics.get('precision', 0):.4f}")
        print(f"   Test Recall:    {test_metrics.get('recall', 0):.4f}")

        mlflow.log_metrics({
            "test_acc": test_metrics["accuracy"],
            "test_auc": test_metrics.get("auc_roc", 0),
            "test_f1": test_metrics.get("f1", 0),
        })
        mlflow.pytorch.log_model(
            model, "model",
            serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    print(f"\n🎉 Training complete! Best model saved to models/best_model.pth")
    print(f"   Best Val AUC-ROC: {best_val_auc:.4f}")


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medical Image Diagnosis Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)

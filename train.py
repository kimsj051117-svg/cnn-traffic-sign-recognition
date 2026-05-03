# train.py
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from config import *


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(out, 1)
        total_loss += loss.item()
        total_acc  += (preds == labels).float().mean().item()
    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        _, preds = torch.max(out, 1)
        total_loss += loss.item()
        total_acc  += (preds == labels).float().mean().item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    import numpy as np
    return (total_loss / len(loader), total_acc / len(loader),
            np.array(all_labels), np.array(all_preds))


def run_training(model, loaders, model_name="model", epochs=EPOCHS_BASELINE, lr=LR_BASELINE):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    # Cosine Annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val_acc  = 0.0
    best_ckpt     = os.path.join(OUTPUT_DIR, f"best_{model_name}.pth")

    print(f"\n{'='*50}")
    print(f"  Training: {model_name}  |  epochs={epochs}  |  lr={lr}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, DEVICE)
        vl_loss, vl_acc, _, _ = eval_one_epoch(model, loaders["val"], criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_ckpt)

        print(f"[{epoch+1:3d}/{epochs}] "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f}"
              + (" ★" if vl_acc == best_val_acc else ""))

    print(f"\nBest val acc: {best_val_acc:.4f} → saved to {best_ckpt}")
    _plot_curves(history, model_name)
    return history, best_ckpt


def _plot_curves(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["acc", "loss"]):
        ax.plot(history[f"train_{metric}"], label=f"Train {metric}")
        ax.plot(history[f"val_{metric}"],   label=f"Val {metric}")
        ax.set_title(f"{model_name} – {metric.capitalize()}")
        ax.set_xlabel("Epoch"); ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"curve_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved learning curve → {path}")
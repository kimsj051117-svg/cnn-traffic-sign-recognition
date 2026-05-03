# evaluate.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

import torch
from torch.utils.data import DataLoader
from config import *
from train import eval_one_epoch
import torch.nn as nn


def full_evaluate(model, loader, model_name="model", class_names=None):
    """테스트셋 전체 평가: accuracy, confusion matrix, 실패 사례"""
    criterion = nn.CrossEntropyLoss()
    loss, acc, y_true, y_pred = eval_one_epoch(model, loader, criterion, DEVICE)

    print(f"\n{'='*50}")
    print(f"  [{model_name}] Test Loss: {loss:.4f}  |  Test Acc: {acc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, digits=4))

    _plot_confusion_matrix(y_true, y_pred, model_name, class_names)
    return loss, acc, y_true, y_pred


def _plot_confusion_matrix(y_true, y_pred, model_name, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names or range(NUM_CLASSES),
                yticklabels=class_names or range(NUM_CLASSES))
    ax.set_title(f"Confusion Matrix – {model_name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"confusion_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {path}")


def show_failure_cases(model, dataset, n=12, model_name="model"):
    """
    틀린 예측 이미지 시각화 – 어떤 클래스를 왜 못 맞추는지 파악용
    """
    model.eval()
    failures = []  # (img_tensor, true, pred)

    with torch.no_grad():
        for img, label in dataset:
            out  = model(img.unsqueeze(0).to(DEVICE))
            pred = out.argmax(1).item()
            if pred != label:
                failures.append((img, label, pred))
            if len(failures) >= n:
                break

    cols = 4
    rows = (len(failures) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    mean = np.array([0.5, 0.5, 0.5])
    std  = np.array([0.5, 0.5, 0.5])

    for i, (img_t, true, pred) in enumerate(failures):
        img_np = img_t.permute(1, 2, 0).numpy() * std + mean
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].set_title(f"T:{true} / P:{pred}", color="red", fontsize=9)
        axes[i].axis("off")

    for ax in axes[len(failures):]:
        ax.axis("off")

    plt.suptitle(f"Failure Cases – {model_name}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"failures_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved failure cases → {path}")
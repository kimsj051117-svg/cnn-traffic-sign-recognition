# train_synset_mix_v3.py
"""
V3 버전: 입력 96x96 + 좌우반전 제거
목표:
- V2 구조 유지
- RandomHorizontalFlip 제거
- Keep left / Keep right, Turn left / Turn right 같은 방향 표지판 혼동 감소
"""

import os
import io
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns

from config import *
from dataset import set_seed, find_or_download_gtsrb, GTSRBDataset
from model_improved import build_efficientnet
from train import train_one_epoch, eval_one_epoch
import matplotlib.pyplot as plt


set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 하이퍼파라미터 ──────────────────────────────────────
IMG_SIZE_V3   = 96
EPOCHS_V3     = 15
BATCH_SIZE_V3 = 32
LR_V3         = 3e-4


# ── 1. Synset Dataset 클래스 ────────────────────────────
class SynsetDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        label     = int(row["label"])
        img_bytes = row["image"]["bytes"]
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# ── 2. Transform: 96x96, 좌우반전 제거 ───────────────────
def get_transform_v3(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE_V3, IMG_SIZE_V3)),
            transforms.RandomRotation(15),

            # V3 핵심 수정:
            # RandomHorizontalFlip 제거
            # 이유: 좌우반전은 Keep left/right, Turn left/right의 의미를 바꿔 라벨 오류를 만들 수 있음.

            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE_V3, IMG_SIZE_V3)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])


# GTSDB용 transform
gtsdb_train_transform_v3 = transforms.Compose([
    transforms.Lambda(lambda img: img.resize((IMG_SIZE_V3, IMG_SIZE_V3), Image.LANCZOS)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.2
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

gtsdb_eval_transform_v3 = transforms.Compose([
    transforms.Lambda(lambda img: img.resize((IMG_SIZE_V3, IMG_SIZE_V3), Image.LANCZOS)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


# Synset용 transform
synset_train_transform_v3 = transforms.Compose([
    transforms.Resize((IMG_SIZE_V3, IMG_SIZE_V3)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

synset_eval_transform_v3 = transforms.Compose([
    transforms.Resize((IMG_SIZE_V3, IMG_SIZE_V3)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


# ── 3. GTSRB 데이터 로딩 ────────────────────────────────
print("\n[1] GTSRB 데이터 로딩...")

data_root = find_or_download_gtsrb()

train_df = pd.read_csv(os.path.join(data_root, "Train.csv"))
test_df  = pd.read_csv(os.path.join(data_root, "Test.csv"))

train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["ClassId"],
    random_state=SEED
)

print(f"  Train: {len(train_split)} | Val: {len(val_split)} | Test: {len(test_df)}")


# ── 4. GTSDB 데이터 분할 ────────────────────────────────
print("\n[2] GTSDB 데이터 분할...")

gtsdb_df = pd.read_csv(GTSDB_CSV)

class_counts = gtsdb_df["ClassId"].value_counts()
rare_classes = class_counts[class_counts < 2].index.tolist()

rare_df   = gtsdb_df[gtsdb_df["ClassId"].isin(rare_classes)]
common_df = gtsdb_df[~gtsdb_df["ClassId"].isin(rare_classes)]

gtsdb_train_common, gtsdb_test_df = train_test_split(
    common_df,
    test_size=0.2,
    stratify=common_df["ClassId"],
    random_state=SEED
)

gtsdb_train_df = pd.concat([gtsdb_train_common, rare_df]).reset_index(drop=True)

print(f"  GTSDB Train: {len(gtsdb_train_df)} | Test: {len(gtsdb_test_df)}")


# ── 5. Synset 데이터 분할 ───────────────────────────────
print("\n[3] Synset 데이터 로딩...")

SYNSET_PARQUET = "/home/work/AI/data/synset-signset-germany/cycles/validation.parquet"

synset_df   = pd.read_parquet(SYNSET_PARQUET)
synset_df43 = synset_df[synset_df["label"] <= 42].reset_index(drop=True)

synset_train_df, synset_test_df = train_test_split(
    synset_df43,
    test_size=0.95,
    stratify=synset_df43["label"],
    random_state=SEED
)

print(f"  Synset Train 5%: {len(synset_train_df)} | Test 95%: {len(synset_test_df)}")


# ── 6. Dataset 구성 ─────────────────────────────────────
print("\n[4] Dataset 구성 (96x96, no horizontal flip)...")

gtsrb_train_ds = GTSRBDataset(
    train_split,
    data_root,
    get_transform_v3(augment=True)
)

gtsdb_train_ds = GTSRBDataset(
    gtsdb_train_df,
    GTSDB_ROOT,
    gtsdb_train_transform_v3
)

synset_train_ds = SynsetDataset(
    synset_train_df,
    synset_train_transform_v3
)

mixed_ds = ConcatDataset([
    gtsrb_train_ds,
    gtsdb_train_ds,
    synset_train_ds
])

val_ds = GTSRBDataset(
    val_split,
    data_root,
    get_transform_v3(augment=False)
)

gtsrb_test_ds = GTSRBDataset(
    test_df,
    data_root,
    get_transform_v3(augment=False)
)

gtsdb_test_ds = GTSRBDataset(
    gtsdb_df,
    GTSDB_ROOT,
    gtsdb_eval_transform_v3
)

synset_test_ds = SynsetDataset(
    synset_test_df,
    synset_eval_transform_v3
)

print(f"  Total train: {len(mixed_ds)}")


# ── 7. DataLoader ───────────────────────────────────────
train_loader = DataLoader(
    mixed_ds,
    batch_size=BATCH_SIZE_V3,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE_V3,
    shuffle=False,
    num_workers=2
)

gtsrb_test_loader = DataLoader(
    gtsrb_test_ds,
    batch_size=BATCH_SIZE_V3,
    shuffle=False,
    num_workers=2
)

gtsdb_test_loader = DataLoader(
    gtsdb_test_ds,
    batch_size=BATCH_SIZE_V3,
    shuffle=False,
    num_workers=2
)

synset_test_loader = DataLoader(
    synset_test_ds,
    batch_size=BATCH_SIZE_V3,
    shuffle=False,
    num_workers=2
)


# ── 8. 모델 준비 ────────────────────────────────────────
print("\n[5] EfficientNet-B0 with 96x96 input")

model = build_efficientnet(freeze_backbone=False).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR_V3
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS_V3
)


# ── 9. 학습 ─────────────────────────────────────────────
print(f"\n[6] 학습 시작 | epochs={EPOCHS_V3} | lr={LR_V3} | img={IMG_SIZE_V3}")
print("=" * 60)

best_val_acc = 0.0
patience = 5
patience_cnt = 0

best_ckpt = os.path.join(
    OUTPUT_DIR,
    "best_efficientnet_v3.pth"
)

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

for epoch in range(EPOCHS_V3):
    tr_loss, tr_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        DEVICE
    )

    vl_loss, vl_acc, _, _ = eval_one_epoch(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        patience_cnt = 0
        torch.save(model.state_dict(), best_ckpt)
        improved_mark = " ★"
    else:
        patience_cnt += 1
        improved_mark = ""

    print(
        f"[{epoch + 1:3d}/{EPOCHS_V3}] "
        f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
        f"val loss={vl_loss:.4f} acc={vl_acc:.4f}"
        f"{improved_mark}"
        + (f" (patience {patience_cnt}/{patience})" if patience_cnt > 0 else "")
    )

    if patience_cnt >= patience:
        print(f"\n⚠️ Early stopping at epoch {epoch + 1}")
        break

print(f"\nBest val acc: {best_val_acc:.4f}")
print(f"Best checkpoint saved to: {best_ckpt}")


# ── 학습 곡선 저장 ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, metric in zip(axes, ["acc", "loss"]):
    ax.plot(history[f"train_{metric}"], label=f"Train {metric}")
    ax.plot(history[f"val_{metric}"], label=f"Val {metric}")
    ax.set_title(f"V3 – {metric.capitalize()}")
    ax.set_xlabel("Epoch")
    ax.legend()

plt.tight_layout()

curve_path = os.path.join(OUTPUT_DIR, "curve_v3.png")
plt.savefig(curve_path, dpi=150)
plt.close()

print(f"Training curve saved to: {curve_path}")


# ── 10. 테스트 ──────────────────────────────────────────
model.load_state_dict(
    torch.load(best_ckpt, map_location=DEVICE)
)

print("\n" + "=" * 60)
print("  [테스트 1] GTSRB Test Set")
print("=" * 60)

_, gtsrb_acc, _, _ = eval_one_epoch(
    model,
    gtsrb_test_loader,
    criterion,
    DEVICE
)

print(f"  GTSRB Test Acc: {gtsrb_acc * 100:.2f}%")


print("\n" + "=" * 60)
print("  [테스트 2] GTSDB Full")
print("=" * 60)

_, gtsdb_acc, _, _ = eval_one_epoch(
    model,
    gtsdb_test_loader,
    criterion,
    DEVICE
)

print(f"  GTSDB Full Acc: {gtsdb_acc * 100:.2f}%")


print("\n" + "=" * 60)
print("  [테스트 3] Synset 95%")
print("=" * 60)

_, synset_acc, y_true, y_pred = eval_one_epoch(
    model,
    synset_test_loader,
    criterion,
    DEVICE
)

print(f"  Synset Test Acc: {synset_acc * 100:.2f}%")
print(classification_report(y_true, y_pred, digits=4, zero_division=0))

# ── Confusion Matrix 저장 (V3) ─────────────────────────
cm = confusion_matrix(y_true, y_pred)

# numpy로 저장 (비교용)
cm_npy_path = os.path.join(OUTPUT_DIR, "cm_v3_synset.npy")
np.save(cm_npy_path, cm)

# 이미지로도 저장
plt.figure(figsize=(14, 12))
sns.heatmap(cm, cmap="Blues", cbar=True)
plt.title("Confusion Matrix - V3 (Synset Test)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()

cm_img_path = os.path.join(OUTPUT_DIR, "cm_v3_synset.png")
plt.savefig(cm_img_path, dpi=150)
plt.close()

print(f"V3 confusion matrix saved to: {cm_img_path}")
print(f"V3 confusion matrix npy saved to: {cm_npy_path}")

# ── 11. 최종 비교 ───────────────────────────────────────
print(f"\n{'=' * 60}")
print("  FINAL COMPARISON")
print(f"{'=' * 60}")
print(f"  {'모델':<30} | {'GTSRB':>6} | {'GTSDB':>6} | {'Synset':>6}")
print(f"  {'-' * 30}-|{'-' * 8}|{'-' * 8}|{'-' * 8}")
print(f"  {'EfficientNet SynsetMix':<30} | {'94.8%':>6} | {'59.2%':>6} | {'69.5%':>6}")
print(
    f"  {'EfficientNet V3 no flip':<30} | "
    f"{gtsrb_acc * 100:.1f}% | "
    f"{gtsdb_acc * 100:.1f}% | "
    f"{synset_acc * 100:.1f}%  ← 이번"
)
print(f"{'=' * 60}")

print("\n✅ Done!")
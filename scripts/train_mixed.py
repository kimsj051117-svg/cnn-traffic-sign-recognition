# train_mixed.py
"""
GTSRB + GTSDB 혼합 학습으로 Domain Shift 개선 실험
"""
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split

from config import *
from dataset import (set_seed, find_or_download_gtsrb,
                     build_dataloaders, GTSRBDataset, get_transforms)
from model_improved import build_efficientnet
from train import run_training
from evaluate import full_evaluate, show_failure_cases
from external_test import run_external_test

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. GTSRB 데이터 준비 ────────────────────────────────
data_root = find_or_download_gtsrb()
loaders, splits = build_dataloaders(data_root)


# ── 2. GTSDB 데이터 분할 (80% 학습, 20% 테스트) ─────────
gtsdb_df = pd.read_csv(GTSDB_CSV)

# 1장짜리 클래스는 stratify 불가 → 학습에만 포함
class_counts = gtsdb_df["ClassId"].value_counts()
rare_classes = class_counts[class_counts < 2].index.tolist()
rare_df      = gtsdb_df[gtsdb_df["ClassId"].isin(rare_classes)]
common_df    = gtsdb_df[~gtsdb_df["ClassId"].isin(rare_classes)]

print(f"  희귀 클래스(학습 전용): {rare_classes}")

# common만 stratify 분할
gtsdb_train_common, gtsdb_test_df = train_test_split(
    common_df,
    test_size=0.2,
    stratify=common_df["ClassId"],
    random_state=SEED
)

# 희귀 클래스는 학습에만 추가
gtsdb_train_df = pd.concat([gtsdb_train_common, rare_df]).reset_index(drop=True)

print(f"\n[GTSDB Split]")
print(f"  Train mix : {len(gtsdb_train_df)}장")
print(f"  Test hold : {len(gtsdb_test_df)}장")



# ── 3. GTSDB transform (저화질 대응) ────────────────────
# LANCZOS 보간법으로 작은 이미지 upscaling 품질 개선
import torchvision.transforms as transforms
from PIL import Image

class ResizeWithLanczos:
    """작은 이미지를 LANCZOS 보간으로 고품질 upscaling"""
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        return img.resize((self.size, self.size), Image.LANCZOS)

gtsdb_train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.resize((48, 48), Image.LANCZOS)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

gtsdb_eval_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.resize((48, 48), Image.LANCZOS)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ── 4. 혼합 Dataset 구성 ────────────────────────────────
gtsrb_train_ds = GTSRBDataset(
    splits["train"], data_root,
    get_transforms(augment=True)
)
gtsdb_train_ds = GTSRBDataset(
    gtsdb_train_df, GTSDB_ROOT,
    gtsdb_train_transform
)

# GTSRB + GTSDB 합치기
mixed_train_ds = ConcatDataset([gtsrb_train_ds, gtsdb_train_ds])

print(f"\n[Mixed Dataset]")
print(f"  GTSRB train : {len(gtsrb_train_ds)}장")
print(f"  GTSDB train : {len(gtsdb_train_ds)}장")
print(f"  Total mixed : {len(mixed_train_ds)}장")

mixed_loaders = {
    "train": DataLoader(mixed_train_ds, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=2),
    "val":   loaders["val"],   # GTSRB validation 그대로 사용
    "test":  loaders["test"],  # GTSRB test 그대로 사용
}

# ── 5. 혼합 학습 ────────────────────────────────────────
print("\n>>> Training Mixed EfficientNet (GTSRB + GTSDB)...")
mixed_model = build_efficientnet(freeze_backbone=False).to(DEVICE)
_, mixed_ckpt = run_training(
    mixed_model, mixed_loaders,
    model_name="efficientnet_mixed",
    epochs=EPOCHS_IMPROVED, lr=LR_IMPROVED
)

# ── 6. GTSRB 테스트 평가 ────────────────────────────────
mixed_model.load_state_dict(torch.load(mixed_ckpt, map_location=DEVICE))
print("\n>>> GTSRB Test evaluation:")
full_evaluate(mixed_model, loaders["test"], model_name="efficientnet_mixed")

# ── 7. GTSDB 외부 테스트 (hold-out 20%) ─────────────────
gtsdb_test_ds = GTSRBDataset(
    gtsdb_test_df, GTSDB_ROOT,
    gtsdb_eval_transform
)
gtsdb_test_loader = DataLoader(
    gtsdb_test_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2
)

import torch.nn as nn
from train import eval_one_epoch
from sklearn.metrics import classification_report

criterion = nn.CrossEntropyLoss()
loss, acc, y_true, y_pred = eval_one_epoch(
    mixed_model, gtsdb_test_loader, criterion, DEVICE
)

print(f"\n{'='*60}")
print(f"  [GTSDB Hold-out Test] Loss: {loss:.4f}  |  Acc: {acc:.4f}")
print(f"{'='*60}")
print(classification_report(y_true, y_pred, digits=4,
                             zero_division=0))

# ── 8. 전체 GTSDB로도 테스트 (기존 방식과 비교) ──────────
print("\n>>> Full GTSDB external test (for comparison):")
run_external_test(mixed_model, model_name="efficientnet_mixed")

# ── 최종 비교 ────────────────────────────────────────────
print("\n" + "="*60)
print("  FINAL COMPARISON")
print("="*60)
print("  모델                  | GTSRB  | GTSDB(전체)")
print("  ---------------------|--------|------------")
print("  Baseline CNN         | 96.9%  | 4.2%")
print("  EfficientNet         | 95.2%  | 3.8%")
print("  EfficientNet Robust  | 87.0%  | 2.1%")
print(f"  EfficientNet Mixed   | ??     | {acc*100:.1f}%  ← 이번 결과")
print("="*60)
print("\n✅ Done! Results saved to ./outputs/")
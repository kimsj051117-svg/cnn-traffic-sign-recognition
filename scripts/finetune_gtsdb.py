# finetune_gtsdb.py
"""
2단계 Fine-tuning:
1단계: 이미 학습된 EfficientNet Mixed 모델 불러오기
2단계: GTSDB 데이터만으로 낮은 LR로 추가 fine-tuning
"""
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from config import *
from dataset import set_seed, find_or_download_gtsrb, build_dataloaders, GTSRBDataset, get_transforms
from model_improved import build_efficientnet
from train import train_one_epoch, eval_one_epoch
from evaluate import full_evaluate
from external_test import run_external_test

import torchvision.transforms as transforms

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. GTSDB 데이터 분할 ────────────────────────────────
gtsdb_df = pd.read_csv(GTSDB_CSV)

class_counts = gtsdb_df["ClassId"].value_counts()
rare_classes = class_counts[class_counts < 2].index.tolist()
rare_df      = gtsdb_df[gtsdb_df["ClassId"].isin(rare_classes)]
common_df    = gtsdb_df[~gtsdb_df["ClassId"].isin(rare_classes)]

gtsdb_train_common, gtsdb_test_df = train_test_split(
    common_df, test_size=0.2,
    stratify=common_df["ClassId"], random_state=SEED
)
gtsdb_train_df = pd.concat([gtsdb_train_common, rare_df]).reset_index(drop=True)

print(f"[GTSDB Split] Train: {len(gtsdb_train_df)}장 | Test: {len(gtsdb_test_df)}장")

# ── 2. GTSDB Transform ──────────────────────────────────
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

# ── 3. DataLoader ───────────────────────────────────────
gtsdb_train_ds = GTSRBDataset(gtsdb_train_df, GTSDB_ROOT, gtsdb_train_transform)
gtsdb_test_ds  = GTSRBDataset(gtsdb_test_df,  GTSDB_ROOT, gtsdb_eval_transform)
gtsdb_all_ds   = GTSRBDataset(gtsdb_df,       GTSDB_ROOT, gtsdb_eval_transform)

gtsdb_train_loader = DataLoader(gtsdb_train_ds, batch_size=32, shuffle=True,  num_workers=2)
gtsdb_test_loader  = DataLoader(gtsdb_test_ds,  batch_size=32, shuffle=False, num_workers=2)
gtsdb_all_loader   = DataLoader(gtsdb_all_ds,   batch_size=32, shuffle=False, num_workers=2)

# ── 4. 1단계 모델 불러오기 ──────────────────────────────
print("\n[1단계] Mixed 모델 불러오는 중...")
mixed_ckpt = os.path.join(OUTPUT_DIR, "best_efficientnet_mixed.pth")
assert os.path.exists(mixed_ckpt), f"Mixed 모델 없음: {mixed_ckpt}\n먼저 train_mixed.py를 실행하세요!"

model = build_efficientnet(freeze_backbone=False).to(DEVICE)
model.load_state_dict(torch.load(mixed_ckpt, map_location=DEVICE))
print(f"  → {mixed_ckpt} 로드 완료")

# ── 5. Fine-tuning 전 GTSDB 성능 확인 ──────────────────
criterion = nn.CrossEntropyLoss()
loss_before, acc_before, _, _ = eval_one_epoch(model, gtsdb_all_loader, criterion, DEVICE)
print(f"\n[Fine-tuning 전] GTSDB Acc: {acc_before:.4f}")

# ── 6. 2단계 Fine-tuning (낮은 LR) ─────────────────────
FINETUNE_EPOCHS = 20
FINETUNE_LR     = 1e-5  # 매우 낮은 LR → 기존 지식 유지하면서 살짝 적응

optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

print(f"\n[2단계] GTSDB Fine-tuning 시작 | epochs={FINETUNE_EPOCHS} | lr={FINETUNE_LR}")
print("="*50)

best_acc   = acc_before
best_ckpt  = os.path.join(OUTPUT_DIR, "best_efficientnet_finetuned.pth")
history    = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

for epoch in range(FINETUNE_EPOCHS):
    tr_loss, tr_acc = train_one_epoch(model, gtsdb_train_loader, criterion, optimizer, DEVICE)
    vl_loss, vl_acc, _, _ = eval_one_epoch(model, gtsdb_test_loader, criterion, DEVICE)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)

    if vl_acc > best_acc:
        best_acc = vl_acc
        torch.save(model.state_dict(), best_ckpt)

    print(f"[{epoch+1:3d}/{FINETUNE_EPOCHS}] "
          f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val loss={vl_loss:.4f} acc={vl_acc:.4f}"
          + (" ★" if vl_acc == best_acc else ""))

print(f"\nBest GTSDB val acc: {best_acc:.4f}")

# ── 7. Fine-tuning 후 전체 GTSDB 테스트 ────────────────
if os.path.exists(best_ckpt):
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))

loss_after, acc_after, y_true, y_pred = eval_one_epoch(
    model, gtsdb_all_loader, criterion, DEVICE
)

print(f"\n{'='*60}")
print(f"  [Fine-tuning 후] GTSDB 전체 Acc: {acc_after:.4f}")
print(f"{'='*60}")
print(classification_report(y_true, y_pred, digits=4, zero_division=0))

# ── 8. GTSRB 성능도 확인 (혹시 떨어졌는지) ─────────────
print("\n[GTSRB 성능 유지 확인]")
data_root = find_or_download_gtsrb()
_, splits = build_dataloaders(data_root)
gtsrb_test_ds     = GTSRBDataset(splits["test"], data_root, get_transforms(augment=False))
gtsrb_test_loader = DataLoader(gtsrb_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

_, gtsrb_acc, _, _ = eval_one_epoch(model, gtsrb_test_loader, criterion, DEVICE)

# ── 9. 최종 비교 ────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  FINAL COMPARISON")
print(f"{'='*60}")
print(f"  모델                   | GTSRB  | GTSDB")
print(f"  -----------------------|--------|-------")
print(f"  Baseline CNN           | 96.9%  | 4.2%")
print(f"  EfficientNet           | 95.2%  | 3.8%")
print(f"  EfficientNet Robust    | 87.0%  | 2.1%")
print(f"  EfficientNet Mixed     | 95.1%  | 50.6%")
print(f"  EfficientNet Finetuned | {gtsrb_acc*100:.1f}%  | {acc_after*100:.1f}%  ← 이번 결과")
print(f"{'='*60}")
print("\n✅ Done!")
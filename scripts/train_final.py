# train_final.py
"""
최종 실험:
학습: GTSRB + GTSDB 혼합
테스트 1: GTSRB test set     (내부 성능)
테스트 2: GTSDB hold-out     (실제 도로)
테스트 3: Synset Signset     (합성 환경 - 최종 일반화 검증)
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

from config import *
from dataset import (set_seed, find_or_download_gtsrb,
                     build_dataloaders, GTSRBDataset, get_transforms)
from model_improved import build_efficientnet
from train import run_training, eval_one_epoch
from evaluate import full_evaluate

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Synset Dataset 클래스 ────────────────────────────
class SynsetDataset(Dataset):
    """Parquet에서 직접 이미지 읽기"""
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


# ── 2. GTSRB 데이터 준비 ────────────────────────────────
print("\n[1] GTSRB 데이터 로딩...")
data_root = find_or_download_gtsrb()
loaders, splits = build_dataloaders(data_root)

# ── 3. GTSDB 데이터 분할 (80% 학습, 20% 테스트) ─────────
print("\n[2] GTSDB 데이터 분할...")
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

print(f"  GTSDB Train: {len(gtsdb_train_df)}장 | Test: {len(gtsdb_test_df)}장")

# ── 4. GTSDB Transform ──────────────────────────────────
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

# ── 5. Synset 데이터 로드 (43클래스만) ──────────────────
print("\n[3] Synset Signset 로딩...")
SYNSET_PARQUET = "/home/work/AI/data/synset-signset-germany/cycles/validation.parquet"
assert os.path.exists(SYNSET_PARQUET), f"Synset 파일 없음: {SYNSET_PARQUET}"

synset_df     = pd.read_parquet(SYNSET_PARQUET)
synset_df43   = synset_df[synset_df["label"] <= 42].reset_index(drop=True)
print(f"  Synset 43클래스: {len(synset_df43)}장 (클래스당 {len(synset_df43)//43}장)")

synset_eval_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ── 6. 혼합 학습 Dataset (GTSRB + GTSDB) ────────────────
print("\n[4] 혼합 Dataset 구성...")
gtsrb_train_ds = GTSRBDataset(splits["train"], data_root, get_transforms(augment=True))
gtsdb_train_ds = GTSRBDataset(gtsdb_train_df,  GTSDB_ROOT, gtsdb_train_transform)
mixed_train_ds = ConcatDataset([gtsrb_train_ds, gtsdb_train_ds])

print(f"  GTSRB : {len(gtsrb_train_ds)}장")
print(f"  GTSDB : {len(gtsdb_train_ds)}장")
print(f"  Total : {len(mixed_train_ds)}장")

mixed_loaders = {
    "train": DataLoader(mixed_train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2),
    "val":   loaders["val"],
    "test":  loaders["test"],
}

# ── 7. 학습 ─────────────────────────────────────────────
print("\n[5] 학습 시작 (GTSRB + GTSDB 혼합)...")
model = build_efficientnet(freeze_backbone=False).to(DEVICE)
_, ckpt = run_training(
    model, mixed_loaders,
    model_name="efficientnet_final",
    epochs=EPOCHS_IMPROVED, lr=LR_IMPROVED
)
model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

criterion = nn.CrossEntropyLoss()

# ── 8. 테스트 1: GTSRB ──────────────────────────────────
print("\n" + "="*60)
print("  [테스트 1] GTSRB Test Set (내부 성능)")
print("="*60)
_, gtsrb_acc, y_true, y_pred = eval_one_epoch(
    model, loaders["test"], criterion, DEVICE
)
print(f"  GTSRB Test Acc: {gtsrb_acc*100:.2f}%")

# ── 9. 테스트 2: GTSDB hold-out ─────────────────────────
print("\n" + "="*60)
print("  [테스트 2] GTSDB Hold-out (실제 도로 환경)")
print("="*60)
gtsdb_test_ds     = GTSRBDataset(gtsdb_test_df, GTSDB_ROOT, gtsdb_eval_transform)
gtsdb_test_loader = DataLoader(gtsdb_test_ds, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=2)
_, gtsdb_acc, y_true, y_pred = eval_one_epoch(
    model, gtsdb_test_loader, criterion, DEVICE
)
print(f"  GTSDB Hold-out Acc: {gtsdb_acc*100:.2f}%")
print(classification_report(y_true, y_pred, digits=4, zero_division=0))

# ── 10. 테스트 3: GTSDB 전체 ────────────────────────────
print("\n" + "="*60)
print("  [테스트 2-전체] GTSDB Full (참고용)")
print("="*60)
gtsdb_all_ds     = GTSRBDataset(gtsdb_df, GTSDB_ROOT, gtsdb_eval_transform)
gtsdb_all_loader = DataLoader(gtsdb_all_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)
_, gtsdb_full_acc, _, _ = eval_one_epoch(
    model, gtsdb_all_loader, criterion, DEVICE
)
print(f"  GTSDB Full Acc: {gtsdb_full_acc*100:.2f}%")

# ── 11. 테스트 3: Synset Signset (최종 일반화 검증) ──────
print("\n" + "="*60)
print("  [테스트 3] Synset Signset (합성 환경 - 최종 검증)")
print("="*60)
synset_test_ds     = SynsetDataset(synset_df43, synset_eval_transform)
synset_test_loader = DataLoader(synset_test_ds, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=2)
_, synset_acc, y_true, y_pred = eval_one_epoch(
    model, synset_test_loader, criterion, DEVICE
)
print(f"  Synset Test Acc: {synset_acc*100:.2f}%")
print(classification_report(y_true, y_pred, digits=4, zero_division=0))

# ── 12. 최종 비교 ────────────────────────────────────────
print("\n" + "="*60)
print("  FINAL COMPARISON SUMMARY")
print("="*60)
print(f"  {'모델':<25} | {'GTSRB':>6} | {'GTSDB':>6} | {'Synset':>6}")
print(f"  {'-'*25}-|{'-'*8}|{'-'*8}|{'-'*8}")
print(f"  {'Baseline CNN':<25} | {'96.9%':>6} | {'4.2%':>6} | {'??':>6}")
print(f"  {'EfficientNet':<25} | {'95.2%':>6} | {'3.8%':>6} | {'??':>6}")
print(f"  {'EfficientNet Robust':<25} | {'87.0%':>6} | {'2.1%':>6} | {'??':>6}")
print(f"  {'EfficientNet Mixed':<25} | {'95.1%':>6} | {'50.6%':>6} | {'??':>6}")
print(f"  {'EfficientNet Final':<25} | {gtsrb_acc*100:.1f}% | {gtsdb_full_acc*100:.1f}% | {synset_acc*100:.1f}%  ← 이번")
print("="*60)
print("\n✅ Done! Results saved to ./outputs/")
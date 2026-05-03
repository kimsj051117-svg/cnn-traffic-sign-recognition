# finetune_synset.py
"""
2단계 Fine-tuning:
1단계: 이미 학습된 EfficientNet Final 모델 불러오기
2단계: Synset 5%로 낮은 LR로 추가 fine-tuning
테스트: Synset 나머지 95%로 최종 검증
"""
import os
import io
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torchvision.transforms as transforms

from config import *
from dataset import set_seed, find_or_download_gtsrb, build_dataloaders, GTSRBDataset, get_transforms
from model_improved import build_efficientnet
from train import train_one_epoch, eval_one_epoch

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


# ── 2. Synset 데이터 로드 및 분할 ───────────────────────
print("\n[1] Synset 데이터 로딩...")
SYNSET_PARQUET = "/home/work/AI/data/synset-signset-germany/cycles/validation.parquet"
synset_df    = pd.read_parquet(SYNSET_PARQUET)
synset_df43  = synset_df[synset_df["label"] <= 42].reset_index(drop=True)
print(f"  43클래스: {len(synset_df43)}장")

# 5% finetune / 95% test 분할 (클래스별 균일하게)
synset_finetune_df, synset_test_df = train_test_split(
    synset_df43,
    test_size=0.95,
    stratify=synset_df43["label"],
    random_state=SEED
)
print(f"  Fine-tune용 5%  : {len(synset_finetune_df)}장 (클래스당 {len(synset_finetune_df)//43}장)")
print(f"  Test용    95%   : {len(synset_test_df)}장 (클래스당 {len(synset_test_df)//43}장)")

# ── 3. Transform ─────────────────────────────────────────
synset_train_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

synset_eval_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ── 4. DataLoader ────────────────────────────────────────
finetune_ds     = SynsetDataset(synset_finetune_df, synset_train_transform)
synset_test_ds  = SynsetDataset(synset_test_df,     synset_eval_transform)
synset_all_ds   = SynsetDataset(synset_df43,        synset_eval_transform)

finetune_loader     = DataLoader(finetune_ds,    batch_size=16,         shuffle=True,  num_workers=2)
synset_test_loader  = DataLoader(synset_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
synset_all_loader   = DataLoader(synset_all_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ── 5. Final 모델 불러오기 ───────────────────────────────
print("\n[2] Final 모델 불러오는 중...")
final_ckpt = os.path.join(OUTPUT_DIR, "best_efficientnet_final.pth")
assert os.path.exists(final_ckpt), f"모델 없음: {final_ckpt}\n먼저 train_final.py를 실행하세요!"

model = build_efficientnet(freeze_backbone=False).to(DEVICE)
model.load_state_dict(torch.load(final_ckpt, map_location=DEVICE))
print(f"  → {final_ckpt} 로드 완료")

# ── 6. Fine-tuning 전 성능 확인 ─────────────────────────
criterion = nn.CrossEntropyLoss()
_, acc_before, _, _ = eval_one_epoch(model, synset_all_loader, criterion, DEVICE)
print(f"\n[Fine-tuning 전] Synset 전체 Acc: {acc_before*100:.2f}%")

# ── 7. 2단계 Fine-tuning ─────────────────────────────────
FINETUNE_EPOCHS = 10
FINETUNE_LR     = 1e-5  # 매우 낮은 LR

optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

best_acc  = 0.0
best_ckpt = os.path.join(OUTPUT_DIR, "best_efficientnet_finetuned_synset.pth")

print(f"\n[3] Synset Fine-tuning | epochs={FINETUNE_EPOCHS} | lr={FINETUNE_LR}")
print("="*50)

for epoch in range(FINETUNE_EPOCHS):
    tr_loss, tr_acc = train_one_epoch(
        model, finetune_loader, criterion, optimizer, DEVICE
    )
    vl_loss, vl_acc, _, _ = eval_one_epoch(
        model, synset_test_loader, criterion, DEVICE
    )
    scheduler.step()

    if vl_acc > best_acc:
        best_acc = vl_acc
        torch.save(model.state_dict(), best_ckpt)

    print(f"[{epoch+1:3d}/{FINETUNE_EPOCHS}] "
          f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val loss={vl_loss:.4f} acc={vl_acc:.4f}"
          + (" ★" if vl_acc == best_acc else ""))

# ── 8. 최적 모델로 최종 테스트 ──────────────────────────
model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))

_, synset_acc, y_true, y_pred = eval_one_epoch(
    model, synset_test_loader, criterion, DEVICE
)
print(f"\n[Fine-tuning 후] Synset 95% Test Acc: {synset_acc*100:.2f}%")
print(classification_report(y_true, y_pred, digits=4, zero_division=0))

# ── 9. GTSRB/GTSDB 성능 유지 확인 ───────────────────────
print("\n[4] GTSRB/GTSDB 성능 유지 확인...")
data_root = find_or_download_gtsrb()
loaders, splits = build_dataloaders(data_root)
_, gtsrb_acc, _, _ = eval_one_epoch(model, loaders["test"], criterion, DEVICE)
print(f"  GTSRB Test Acc: {gtsrb_acc*100:.2f}%")

# GTSDB 전체 테스트
gtsdb_df = pd.read_csv(GTSDB_CSV)
gtsdb_ds = GTSRBDataset(
    gtsdb_df, GTSDB_ROOT,
    transforms.Compose([
        transforms.Lambda(lambda img: img.resize((48, 48), Image.LANCZOS)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
)
gtsdb_loader = DataLoader(gtsdb_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
_, gtsdb_acc, _, _ = eval_one_epoch(model, gtsdb_loader, criterion, DEVICE)
print(f"  GTSDB Full Acc: {gtsdb_acc*100:.2f}%")

# ── 10. 최종 비교 ────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  FINAL COMPARISON SUMMARY")
print(f"{'='*60}")
print(f"  {'모델':<25} | {'GTSRB':>6} | {'GTSDB':>6} | {'Synset':>6}")
print(f"  {'-'*25}-|{'-'*8}|{'-'*8}|{'-'*8}")
print(f"  {'Baseline CNN':<25} | {'96.9%':>6} | {'4.2%':>6} | {'??':>6}")
print(f"  {'EfficientNet':<25} | {'95.2%':>6} | {'3.8%':>6} | {'??':>6}")
print(f"  {'EfficientNet Robust':<25} | {'87.0%':>6} | {'2.1%':>6} | {'??':>6}")
print(f"  {'EfficientNet Mixed':<25} | {'95.1%':>6} | {'50.6%':>6} | {'??':>6}")
print(f"  {'EfficientNet Final':<25} | {'94.9%':>6} | {'58.4%':>6} | {'56.2%':>6}")
print(f"  {'EfficientNet Finetuned':<25} | {gtsrb_acc*100:.1f}% | {gtsdb_acc*100:.1f}% | {synset_acc*100:.1f}%  ← 이번")
print(f"{'='*60}")
print("\n✅ Done!")
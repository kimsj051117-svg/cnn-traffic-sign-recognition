# main.py – 전체 파이프라인 + Robust 실험
import os
import torch
from torch.utils.data import DataLoader
from config import *
from dataset import set_seed, find_or_download_gtsrb, build_dataloaders, GTSRBDataset, get_transforms
from model_baseline import BaselineCNN
from model_improved import build_efficientnet
from train import run_training
from evaluate import full_evaluate, show_failure_cases
from external_test import run_external_test

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. 데이터 준비 ──────────────────────────────────────
data_root = find_or_download_gtsrb()
loaders, splits = build_dataloaders(data_root)

# ── 2. Baseline 학습 ────────────────────────────────────
baseline = BaselineCNN().to(DEVICE)
_, baseline_ckpt = run_training(
    baseline, loaders,
    model_name="baseline",
    epochs=EPOCHS_BASELINE, lr=LR_BASELINE
)

# ── 3. Baseline 평가 ────────────────────────────────────
baseline.load_state_dict(torch.load(baseline_ckpt, map_location=DEVICE))
full_evaluate(baseline, loaders["test"], model_name="baseline")
show_failure_cases(baseline, loaders["test"].dataset, model_name="baseline")

# ── 4. EfficientNet 학습 ────────────────────────────────
efficientnet = build_efficientnet(freeze_backbone=False).to(DEVICE)
_, eff_ckpt = run_training(
    efficientnet, loaders,
    model_name="efficientnet",
    epochs=EPOCHS_IMPROVED, lr=LR_IMPROVED
)

# ── 5. EfficientNet 평가 ────────────────────────────────
efficientnet.load_state_dict(torch.load(eff_ckpt, map_location=DEVICE))
full_evaluate(efficientnet, loaders["test"], model_name="efficientnet")
show_failure_cases(efficientnet, loaders["test"].dataset, model_name="efficientnet")

# ── 6. GTSDB 외부 테스트 (기존 모델) ────────────────────
print("\n>>> External test: Baseline")
run_external_test(baseline, model_name="baseline")

print("\n>>> External test: EfficientNet")
run_external_test(efficientnet, model_name="efficientnet")

# ── 7. Robust EfficientNet 학습 (GTSDB 대응 augmentation) ─
print("\n>>> Training Robust EfficientNet with stronger augmentation...")

robust_train_ds = GTSRBDataset(splits["train"], data_root, get_transforms(robust=True))
robust_val_ds   = GTSRBDataset(splits["val"],   data_root, get_transforms(augment=False))
robust_test_ds  = GTSRBDataset(splits["test"],  data_root, get_transforms(augment=False))

robust_loaders = {
    "train": DataLoader(robust_train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2),
    "val":   DataLoader(robust_val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    "test":  DataLoader(robust_test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
}

robust_model = build_efficientnet(freeze_backbone=False).to(DEVICE)
_, robust_ckpt = run_training(
    robust_model, robust_loaders,
    model_name="efficientnet_robust",
    epochs=EPOCHS_IMPROVED, lr=LR_IMPROVED
)

# ── 8. Robust 모델 평가 ─────────────────────────────────
robust_model.load_state_dict(torch.load(robust_ckpt, map_location=DEVICE))
full_evaluate(robust_model, robust_loaders["test"], model_name="efficientnet_robust")
show_failure_cases(robust_model, robust_loaders["test"].dataset, model_name="efficientnet_robust")

# ── 9. Robust 모델 GTSDB 외부 테스트 ───────────────────
print("\n>>> External test: Robust EfficientNet")
run_external_test(robust_model, model_name="efficientnet_robust")

# ── 최종 비교 출력 ──────────────────────────────────────
print("\n" + "="*60)
print("  FINAL COMPARISON SUMMARY")
print("="*60)
print(f"  Baseline CNN        - GTSRB: ~96.4%  | GTSDB: ~3.9%")
print(f"  EfficientNet        - GTSRB: ~95.4%  | GTSDB: ~3.7%")
print(f"  EfficientNet Robust - GTSRB: ??       | GTSDB: ??")
print("="*60)
print("\n✅ All done! Results saved to ./outputs/")
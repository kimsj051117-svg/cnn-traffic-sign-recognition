# external_test.py
"""
GTSRB로 학습한 모델을 GTSDB 실사 도로 이미지로 외부 테스트.
→ Domain shift 분석: 깨끗한 crop vs 실제 도로 환경
"""
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import *
from dataset import GTSRBDataset, get_transforms
from train import eval_one_epoch
from evaluate import full_evaluate


def run_external_test(model, model_name="model"):
    assert os.path.exists(GTSDB_CSV), f"GTSDB CSV not found: {GTSDB_CSV}"

    gtsdb_df = pd.read_csv(GTSDB_CSV)
    print(f"\n[GTSDB] Total external samples: {len(gtsdb_df)}")

    gtsdb_ds = GTSRBDataset(gtsdb_df, GTSDB_ROOT, get_transforms(augment=False))
    gtsdb_loader = DataLoader(gtsdb_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)

    loss, acc, y_true, y_pred = full_evaluate(
        model, gtsdb_loader,
        model_name=f"{model_name}_GTSDB_external"
    )

    print("\n[Domain Shift Analysis]")
    print("GTSRB Test (in-distribution) vs GTSDB External (out-of-distribution)")
    print("→ 성능 차이가 크다면 도메인 이동(domain shift)이 발생한 것")
    print("→ 원인: 배경 노이즈, 조명 변화, 이미지 크기 불균일 등")

    return loss, acc
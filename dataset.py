# dataset.py
import os, glob, random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config import *


# ── 재현성 ──────────────────────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── 데이터 탐색 & 다운로드 ──────────────────────────────
def find_or_download_gtsrb() -> str:
    candidates = [
        Path("/home/work/AI/data/gtsrb-german-traffic-sign"),
        Path("/home/work/AI/gtsrb-german-traffic-sign"),
        Path("./gtsrb-german-traffic-sign"),
        Path("./data/gtsrb-german-traffic-sign"),
    ]
    for p in candidates:
        if (p / "Train.csv").exists() and (p / "Test.csv").exists():
            print(f"[INFO] Found GTSRB at: {p}")
            return str(p)

    matches = glob.glob("/home/work/**/Train.csv", recursive=True)
    for m in matches:
        root = Path(m).parent
        if (root / "Test.csv").exists():
            print(f"[INFO] Found GTSRB by search at: {root}")
            return str(root)

    print("[INFO] Downloading GTSRB from KaggleHub...")
    import kagglehub
    path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    return str(path)


# ── Transform 정의 ──────────────────────────────────────
def get_transforms(img_size=IMG_SIZE, augment=False, robust=False):
    if robust:
        # GTSDB 도메인 대응: 저화질 + 크기 불균일 + 조명 변화 시뮬레이션
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    elif augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])


# ── Dataset 클래스 ──────────────────────────────────────
class GTSRBDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.root_dir  = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = os.path.join(self.root_dir, row["Path"])
        label = int(row["ClassId"])
        img   = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── DataLoader 빌더 ─────────────────────────────────────
def build_dataloaders(data_root):
    train_df = pd.read_csv(os.path.join(data_root, "Train.csv"))
    test_df  = pd.read_csv(os.path.join(data_root, "Test.csv"))

    train_split, val_split = train_test_split(
        train_df, test_size=0.2,
        stratify=train_df["ClassId"], random_state=SEED
    )

    print(f"Train: {len(train_split)} | Val: {len(val_split)} | Test: {len(test_df)}")

    train_ds = GTSRBDataset(train_split, data_root, get_transforms(augment=True))
    val_ds   = GTSRBDataset(val_split,  data_root, get_transforms(augment=False))
    test_ds  = GTSRBDataset(test_df,    data_root, get_transforms(augment=False))

    loaders = {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2),
        "val":   DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
        "test":  DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    }
    splits = {"train": train_split, "val": val_split, "test": test_df}
    return loaders, splits
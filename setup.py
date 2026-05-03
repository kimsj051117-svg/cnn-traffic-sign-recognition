# setup.py
# 세션 재시작 후 가장 먼저 실행하는 파일
# 필요한 데이터 자동 다운로드 및 생성

import os
import pandas as pd
from PIL import Image
from pathlib import Path

print("="*50)
print("  환경 설정 시작")
print("="*50)

# ── 1. GTSRB 다운로드 ───────────────────────────────────
print("\n[1] GTSRB 확인 중...")
import kagglehub
gtsrb_path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
print(f"  GTSRB: {gtsrb_path}")

# ── 2. GTSDB 다운로드 ───────────────────────────────────
print("\n[2] GTSDB 확인 중...")
gtsdb_path = kagglehub.dataset_download("safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb")
print(f"  GTSDB: {gtsdb_path}")

# ── 3. gtsdb_crops.csv 생성 ─────────────────────────────
GTSDB_ROOT = "/home/work/.cache/kagglehub/datasets/safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb/versions/1"
CSV_PATH   = os.path.join(GTSDB_ROOT, "gtsdb_crops.csv")

if os.path.exists(CSV_PATH):
    print(f"\n[3] gtsdb_crops.csv 이미 존재 → 스킵")
else:
    print(f"\n[3] gtsdb_crops.csv 생성 중...")
    CROPS_DIR = os.path.join(GTSDB_ROOT, "gtsdb_crops")
    GT_PATH   = os.path.join(GTSDB_ROOT, "gt.txt")
    IMG_DIR   = os.path.join(GTSDB_ROOT, "TrainIJCNN2013/TrainIJCNN2013")

    records = []
    with open(GT_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts    = line.split(";")
            filename = parts[0]
            x1, y1, x2, y2 = map(int, parts[1:5])
            class_id = int(parts[5])
            records.append((filename, x1, y1, x2, y2, class_id))

    os.makedirs(CROPS_DIR, exist_ok=True)
    rows = []

    for i, (filename, x1, y1, x2, y2, class_id) in enumerate(records):
        img_path = os.path.join(IMG_DIR, filename)
        if not os.path.exists(img_path):
            continue
        img  = Image.open(img_path).convert("RGB")
        crop = img.crop((x1, y1, x2, y2))

        class_dir     = os.path.join(CROPS_DIR, f"{class_id:02d}")
        os.makedirs(class_dir, exist_ok=True)

        base          = os.path.splitext(filename)[0]
        crop_filename = f"{base}_{i:05d}.png"
        crop.save(os.path.join(class_dir, crop_filename))

        rows.append({"Path": f"gtsdb_crops/{class_id:02d}/{crop_filename}",
                     "ClassId": class_id})

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"  완료: {len(rows)}개 crop 생성")

# ── 4. Synset 다운로드 ───────────────────────────────────
print("\n[4] Synset 확인 중...")
SYNSET_PATH = "/home/work/AI/data/synset-signset-germany/cycles/validation.parquet"

if os.path.exists(SYNSET_PATH):
    print(f"  Synset 이미 존재 → 스킵")
else:
    print(f"  Synset 다운로드 중... (약 2분 소요)")
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="FraunhoferIOSB/Synset-Signset-Germany",
        repo_type="dataset",
        filename="cycles/validation.parquet",
        local_dir="/home/work/AI/data/synset-signset-germany"
    )
    print(f"  완료: {SYNSET_PATH}")

# ── 5. 완료 ─────────────────────────────────────────────
print("\n" + "="*50)
print("  ✅ 환경 설정 완료!")
print("="*50)
print("\n이제 실행 순서:")
print("  1. python main.py")
print("  2. python train_mixed.py")
print("  3. python train_final.py")
print("  4. python train_synset_mix.py")
print("  5. python train_synset_mix_v2.py")
print("  6. train_synset_mix_v3")
print("  7. python demo.py")
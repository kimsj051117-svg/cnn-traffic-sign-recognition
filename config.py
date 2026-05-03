# config.py
import torch

# ── 경로 ──────────────────────────────────────────────
DATA_ROOT        = None          # find_or_download_gtsrb()로 자동 탐색
GTSDB_ROOT       = "/home/work/.cache/kagglehub/datasets/safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb/versions/1"
GTSDB_CSV        = f"{GTSDB_ROOT}/gtsdb_crops.csv"

OUTPUT_DIR       = "./outputs"   # 결과물 저장 폴더

# ── 학습 설정 ──────────────────────────────────────────
IMG_SIZE         = 48
BATCH_SIZE       = 64
EPOCHS_BASELINE  = 20
EPOCHS_IMPROVED  = 15
LR_BASELINE      = 1e-3
LR_IMPROVED      = 3e-4          # fine-tuning은 낮은 LR 권장
NUM_CLASSES      = 43
SEED             = 42

# ── 디바이스 ───────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
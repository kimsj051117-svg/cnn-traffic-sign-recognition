# demo_이상한거 삭제버전
# demo_v3_clean.py
"""
Gradio 인퍼런스 데모
- Baseline CNN
- Final V2 EfficientNet-B0
- Final V3 EfficientNet-B0: 좌우반전 제거 버전

주의:
- 이 모델은 detection 모델이 아니라 classification 모델입니다.
- 따라서 입력 이미지는 표지판만 crop된 이미지여야 합니다.
"""

from pathlib import Path
from functools import lru_cache

import torch
import torchvision.transforms as transforms
import gradio as gr

from config import *
from model_baseline import BaselineCNN
from model_improved import build_efficientnet


# ── GTSRB 43개 클래스 이름 ──────────────────────────────
CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing >3.5t", "Right-of-way",
    "Priority road", "Yield", "Stop", "No vehicles", "No vehicles >3.5t",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware ice/snow", "Wild animals", "End restrictions",
    "Turn right ahead", "Turn left ahead", "Ahead only", "Straight or right",
    "Straight or left", "Keep right", "Keep left", "Roundabout mandatory",
    "End no passing", "End no passing >3.5t"
]


# ── 데모에서 선택할 모델 목록 ───────────────────────────
MODEL_CONFIGS = {
    "Final V3 EfficientNet-B0 (No Flip, 96px)": {
        "arch": "efficientnet",
        "ckpt": "best_efficientnet_v3.pth",
        "img_size": 96,
        "description": "V3 최종 모델: GTSRB + GTSDB + Synset, 좌우반전 제거",
    },

    "Final V2 EfficientNet-B0 (Synset Mix, 96px)": {
        "arch": "efficientnet",
        "ckpt": "best_efficientnet_v2.pth",
        "img_size": 96,
        "description": "V2 모델: GTSRB + GTSDB + Synset",
    },

    "Baseline CNN": {
        "arch": "baseline",
        "ckpt": "best_baseline.pth",
        "img_size": IMG_SIZE,
        "description": "기본 CNN 모델",
    },
}


def make_transform(img_size):
    """
    데모용 전처리.
    학습 때와 동일하게 resize, tensor 변환, normalize 수행.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


def resolve_checkpoint(filename):
    """
    모델 파일을 여러 위치에서 자동으로 찾는 함수.
    현재 모델들은 주로 /home/work/test/outputs 에 저장되어 있음.
    """
    script_dir = Path(__file__).resolve().parent

    candidate_paths = [
        Path(OUTPUT_DIR) / filename,
        script_dir / "outputs" / filename,
        script_dir.parent / "outputs" / filename,
        Path("/home/work/test/outputs") / filename,
        Path("/home/work/test/ai_team/outputs") / filename,
    ]

    for path in candidate_paths:
        if path.exists():
            return str(path)

    checked = "\n".join(str(p) for p in candidate_paths)

    raise FileNotFoundError(
        f"\n모델 파일을 찾을 수 없습니다: {filename}\n\n"
        f"확인한 경로:\n{checked}\n\n"
        f"터미널에서 아래 명령어로 확인하세요:\n"
        f"find /home/work -name '{filename}'"
    )


@lru_cache(maxsize=8)
def load_model(model_name):
    """
    선택한 모델을 한 번만 로드하고 캐시에 저장.
    같은 모델을 다시 선택하면 재로딩하지 않음.
    """
    cfg = MODEL_CONFIGS[model_name]

    if cfg["arch"] == "baseline":
        model = BaselineCNN()
    else:
        model = build_efficientnet(freeze_backbone=False)

    ckpt_path = resolve_checkpoint(cfg["ckpt"])

    print("=" * 60)
    print(f"[INFO] Loading model: {model_name}")
    print(f"[INFO] Checkpoint: {ckpt_path}")
    print("=" * 60)

    state_dict = torch.load(ckpt_path, map_location=DEVICE)

    # DataParallel로 저장된 경우 module. 제거
    if isinstance(state_dict, dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def predict(image, model_name):
    """
    이미지 입력 → 선택한 모델로 예측 → Top-5 클래스 확률 출력
    """
    if image is None:
        raise gr.Error("이미지를 먼저 업로드하세요.")

    cfg = MODEL_CONFIGS[model_name]

    model = load_model(model_name)
    transform = make_transform(cfg["img_size"])

    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]

    top5_probs, top5_idx = probs.topk(5)

    result = {
        CLASS_NAMES[int(idx)]: float(prob.detach().cpu().item())
        for idx, prob in zip(top5_idx, top5_probs)
    }

    return result


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Traffic Sign Image"),

        gr.Radio(
            choices=list(MODEL_CONFIGS.keys()),
            label="Model",
            value="Final V3 EfficientNet-B0 (No Flip, 96px)",
        ),
    ],
    outputs=gr.Label(num_top_classes=5, label="Prediction"),
    title="🚦 GTSRB Traffic Sign Classifier Demo",
    description=(
        "Upload a cropped traffic sign image. "
        "This demo compares the final V3 model, the previous V2 model, and the baseline CNN. "
        "The V3 model was trained with GTSRB + GTSDB + Synset mixed data, "
        "96x96 input size, and no horizontal flip augmentation."
    ),
)


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
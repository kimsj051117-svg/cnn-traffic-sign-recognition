# model_improved.py
import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES


def build_efficientnet(num_classes=NUM_CLASSES, freeze_backbone=False):
    """
    EfficientNet-B0 전이학습 모델.

    전략:
    - ImageNet pretrained weights 사용
    - freeze_backbone=True  → feature extractor 고정, 마지막 FC만 학습 (빠른 수렴)
    - freeze_backbone=False → 전체 fine-tuning (더 높은 성능)
    기본값은 전체 fine-tuning (교통표지 도메인이 ImageNet과 달라 효과적)
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # 마지막 분류기 교체 (1000 → NUM_CLASSES)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model
"""Baseline model adapters and pretrained checkpoint identifiers."""
from __future__ import annotations
from torch import nn
TV = {'MobileNetV3-Large': ('mobilenet_v3_large', 'MobileNet_V3_Large_Weights', 'IMAGENET1K_V2'), 'EfficientNet-B0': ('efficientnet_b0', 'EfficientNet_B0_Weights', 'IMAGENET1K_V1'), 'ResNet-50': ('resnet50', 'ResNet50_Weights', 'IMAGENET1K_V2'), 'DenseNet-121': ('densenet121', 'DenseNet121_Weights', 'IMAGENET1K_V1'), 'EfficientNetV2-S': ('efficientnet_v2_s', 'EfficientNet_V2_S_Weights', 'IMAGENET1K_V1'), 'ConvNeXt-Tiny': ('convnext_tiny', 'ConvNeXt_Tiny_Weights', 'IMAGENET1K_V1'), 'ViT-B/16': ('vit_b_16', 'ViT_B_16_Weights', 'IMAGENET1K_V1')}
TIMM = {'Swin-T': 'swin_tiny_patch4_window7_224.ms_in1k', 'TinyViT-21M': 'tiny_vit_21m_224.in1k', 'EfficientFormerV2-S2': 'efficientformerv2_s2.snap_dist_in1k', 'EdgeNeXt-S': 'edgenext_small.usi_in1k', 'FastViT-SA12': 'fastvit_sa12.apple_in1k'}

class DINOv2Small(nn.Module):

    def __init__(self, k: int, pretrained: bool):
        super().__init__()
        try:
            from transformers import Dinov2Model, Dinov2Config
        except ImportError as e:
            raise ImportError('Install transformers to use DINOv2') from e
        self.base = Dinov2Model.from_pretrained('facebook/dinov2-small') if pretrained else Dinov2Model(Dinov2Config())
        self.classifier = nn.Linear(self.base.config.hidden_size, k)

    def forward(self, x):
        return self.classifier(self.base(pixel_values=x).last_hidden_state[:, 0, :])

def create_baseline(name: str, num_classes: int, pretrained: bool=False):
    if name in TV:
        from torchvision import models
        constructor, enum, version = TV[name]
        weights = getattr(getattr(models, enum), version) if pretrained else None
        model = getattr(models, constructor)(weights=weights)
        if name == 'ResNet-50':
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif name == 'DenseNet-121':
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif name == 'ViT-B/16':
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        elif name == 'Swin-T':
            model.head = nn.Linear(model.head.in_features, num_classes)
        else:
            last = model.classifier[-1]
            model.classifier[-1] = nn.Linear(last.in_features, num_classes)
        return model
    if name in TIMM:
        try:
            import timm
        except ImportError as e:
            raise ImportError('Install timm to use this model') from e
        return timm.create_model(TIMM[name], pretrained=pretrained, num_classes=num_classes)
    if name == 'DINOv2 ViT-S/14':
        return DINOv2Small(num_classes, pretrained)
    if name == 'FasterViT-0':
        # Official NVIDIA FasterViT release (NVlabs/FasterViT), 224px ImageNet-1K model.
        # Load the original pretrained 1000-class head first, then replace it.
        # This avoids a classification-head shape mismatch during weight loading.
        try:
            from fastervit import create_model as create_fastervit
        except ImportError as exc:
            raise ImportError(
                'FasterViT-0 requires the official NVIDIA fastervit package '
                '(pip install "fastervit>=0.9.8").'
            ) from exc
        model = create_fastervit('faster_vit_0_224', pretrained=pretrained)
        if not isinstance(model.head, nn.Linear):
            raise TypeError('Unexpected NVIDIA FasterViT-0 classification head')
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    raise ValueError('Unknown comparator: ' + name)

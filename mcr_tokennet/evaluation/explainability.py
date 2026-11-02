"""Grad-CAM and perturbation-based attribution evaluation."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F

def grad_cam(model: torch.nn.Module, images: torch.Tensor, layer: torch.nn.Module, targets: torch.Tensor | None=None) -> torch.Tensor:
    """Return normalized [B,H,W] Grad-CAM maps for target logits."""
    activations = []
    gradients = []
    handle = layer.register_forward_hook(lambda _m, _x, out: activations.append(out))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(images)
        if targets is None:
            targets = logits.argmax(dim=1)
        if not activations:
            raise RuntimeError('The selected layer did not execute')
        handle_grad = activations[-1].register_hook(lambda g: gradients.append(g))
        try:
            logits.gather(1, targets[:, None]).sum().backward()
        finally:
            handle_grad.remove()
        fmap = activations[-1]
        weight = gradients[-1].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weight * fmap).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
        return (cam / (cam.flatten(1).amax(dim=1)[:, None, None] + 1e-08)).detach()
    finally:
        handle.remove()

@torch.no_grad()
def insertion_deletion_auc(model: torch.nn.Module, image: torch.Tensor, saliency: torch.Tensor, target: int, baseline: torch.Tensor, step: float=0.05) -> tuple[float, float]:
    """Normalized area of target probability over inserted/removed saliency-ranked pixels."""
    if image.shape[0] != 1 or image.shape[-2:] != saliency.shape[-2:]:
        raise ValueError('Expected one image and matching map')
    if baseline.shape != image.shape:
        raise ValueError('Baseline must match input shape')
    height, width = image.shape[-2:]
    count = height * width
    idx = saliency.flatten().argsort(descending=True)
    fractions = np.linspace(0, 1, int(round(1 / step)) + 1)
    inserted = []
    removed = []
    for fraction in fractions:
        k = min(count, int(round(fraction * count)))
        mask = torch.zeros(count, device=image.device, dtype=torch.bool)
        mask[idx[:k]] = True
        mask = mask.reshape(1, 1, height, width)
        xins = torch.where(mask, image, baseline)
        xdel = torch.where(mask, baseline, image)
        inserted.append(float(model(xins).softmax(1)[0, target]))
        removed.append(float(model(xdel).softmax(1)[0, target]))
    return (float(np.trapezoid(inserted, fractions)), float(np.trapezoid(removed, fractions)))

def lime_superpixels(model: torch.nn.Module, image_rgb: np.ndarray, normalization: dict, *, num_samples: int=1000, seed: int=42):
    """Optional LIME explanation for a single [H,W,3] RGB image."""
    try:
        from lime import lime_image
    except ImportError as exc:
        raise ImportError('Install the optional lime package for LIME explanations') from exc
    from PIL import Image
    from mcr_tokennet.data import ImagePipeline
    device = next(model.parameters()).device
    transformer = ImagePipeline(False, normalization['mean'], normalization['std'])

    def predict(images):
        batch = torch.stack([transformer(Image.fromarray(np.asarray(img, dtype=np.uint8))) for img in images]).to(device)
        with torch.no_grad():
            return model(batch).softmax(1).cpu().numpy()
    explainer = lime_image.LimeImageExplainer(random_state=seed)
    explanation = explainer.explain_instance(image_rgb, predict, top_labels=1, num_samples=num_samples)
    return explanation

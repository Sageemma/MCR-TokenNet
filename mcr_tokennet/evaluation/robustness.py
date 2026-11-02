"""Deterministic controlled RGB corruptions before channel normalization."""
from __future__ import annotations
import hashlib, io
import numpy as np
from PIL import Image
import cv2
BLUR_SIGMA = (0.5, 1.0, 1.5)
BLUR_KERNEL = (5, 7, 11)
NOISE_SIGMA = (5, 10, 15)
PHOTOMETRIC = (0.1, 0.2, 0.3)
JPEG_QUALITY = (90, 70, 50)
OCCLUSION = (50, 71, 100)
KINDS = ('gaussian_blur', 'gaussian_noise', 'brightness_contrast', 'jpeg', 'square_occlusion')

def corrupt(rgb: np.ndarray, kind: str, severity: int, seed: int, mean_rgb=None) -> np.ndarray:
    if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected RGB uint8 array [224,224,3]')
    if severity not in (1, 2, 3):
        raise ValueError('Severity must be 1, 2, or 3')
    i = severity - 1
    rng = np.random.default_rng(seed)
    if kind == 'gaussian_blur':
        return cv2.GaussianBlur(rgb, (BLUR_KERNEL[i],) * 2, BLUR_SIGMA[i], borderType=cv2.BORDER_REFLECT_101)
    if kind == 'gaussian_noise':
        return np.clip(rgb.astype(float) + rng.normal(0, NOISE_SIGMA[i], rgb.shape), 0, 255).round().astype(np.uint8)
    if kind == 'brightness_contrast':
        brightness = rng.uniform(1 - PHOTOMETRIC[i], 1 + PHOTOMETRIC[i])
        contrast = rng.uniform(1 - PHOTOMETRIC[i], 1 + PHOTOMETRIC[i])
        pixels = rgb.astype(float)
        mean = pixels.mean((0, 1), keepdims=True)
        return np.clip(((pixels - mean) * contrast + mean) * brightness, 0, 255).round().astype(np.uint8)
    if kind == 'jpeg':
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format='JPEG', quality=JPEG_QUALITY[i], subsampling=0)
        return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'))
    if kind == 'square_occlusion':
        if mean_rgb is None:
            raise ValueError('Training RGB mean required for occlusion')
        width = OCCLUSION[i]
        row = int(rng.integers(0, 225 - width))
        col = int(rng.integers(0, 225 - width))
        out = rgb.copy()
        out[row:row + width, col:col + width] = np.rint(mean_rgb).astype(np.uint8)
        return out
    raise ValueError(f'Unknown corruption: {kind}')

def per_image_seed(sample_id: str, kind: str, severity: int, realization: int) -> int:
    payload = f'{sample_id}|{kind}|{severity}|{realization}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')

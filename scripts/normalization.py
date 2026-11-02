#!/usr/bin/env python3
"""Estimate channel statistics using the training images only."""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
from mcr_tokennet.data import ImagePipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    with open(a.manifest, encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == 'train']
    if not rows:
        raise ValueError('No TRAIN images')
    total = 0
    sum1 = np.zeros(3, dtype=np.float64)
    sum2 = np.zeros(3, dtype=np.float64)
    root = Path(a.image_root).resolve()
    for row in rows:
        path = Path(row['relative_path'])
        file = (root / path).resolve()
        if path.is_absolute() or '..' in path.parts or (not file.is_relative_to(root)):
            raise ValueError('Unsafe image path')
        with Image.open(file) as raw:
            im = ImagePipeline.resize(raw.convert('RGB'))
        w, h = im.size
        arr = np.asarray(im.crop(((w - 224) // 2, (h - 224) // 2, (w - 224) // 2 + 224, (h - 224) // 2 + 224)), dtype=np.float64) / 255.0
        sum1 += arr.sum(axis=(0, 1))
        sum2 += (arr ** 2).sum(axis=(0, 1))
        total += arr.shape[0] * arr.shape[1]
    mean = sum1 / total
    std = np.sqrt(np.maximum(0, sum2 / total - mean ** 2))
    if (std <= 0).any():
        raise ValueError('Zero channel std')
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({'mean': mean.tolist(), 'std': std.tolist(), 'method': 'train-only deterministic resize and center crop'}, indent=2))
    print('Wrote', dest, 'from', len(rows), 'TRAIN observations')
if __name__ == '__main__':
    main()

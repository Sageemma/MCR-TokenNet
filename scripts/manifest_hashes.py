#!/usr/bin/env python3
"""Compute SHA-256 and perceptual hashes for manifest observations."""
import argparse, csv, hashlib
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
HEADER = ['dataset', 'sample_id', 'relative_path', 'class_label', 'split', 'group_id', 'magnification', 'source_sha256', 'decoded_sha256', 'phash']

def phash(im):
    g = cv2.cvtColor(np.asarray(im.convert('RGB')), cv2.COLOR_RGB2GRAY)
    small = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    threshold = np.median(dct.ravel()[1:])
    bits = dct >= threshold
    val = 0
    for flag in bits.ravel():
        val = val << 1 | int(flag)
    return f'{val:016x}'

def image_hashes(path):
    source = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as raw:
        image = raw.convert('RGB')
        w, h = image.size
        decoded = hashlib.sha256(f'RGB:{w}:{h}:'.encode() + image.tobytes()).hexdigest()
        perceptual = phash(image)
    return (source, decoded, perceptual)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inventory', required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--dataset', default='')
    a = p.parse_args()
    with open(a.inventory, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        needed = set(HEADER[1:7])
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError('Missing manifest columns: ' + str(sorted(missing)))
        rows = list(reader)
    if not rows:
        raise ValueError('Empty manifest')
    out = []
    root = Path(a.image_root).resolve()
    for row in rows:
        if row['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Invalid split label')
        relative = Path(row['relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe relative_path')
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Path escapes root')
        if not path.is_file():
            raise FileNotFoundError(path)
        s, d, h = image_hashes(path)
        out.append({**{k: row.get(k) or a.dataset if k == 'dataset' else row[k] for k in HEADER[:7]}, 'source_sha256': s, 'decoded_sha256': d, 'phash': h})
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', newline='', encoding='utf8') as f:
        wr = csv.DictWriter(f, HEADER)
        wr.writeheader()
        wr.writerows(out)
    print('Hashed', len(out), 'observations into', dest)
if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Evaluate a selected checkpoint under five controlled image degradations."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from mcr_tokennet.data import ImagePipeline
from mcr_tokennet.evaluation import compute_metrics
from mcr_tokennet.evaluation.robustness import KINDS, corrupt, per_image_seed
from evaluate import load_checkpoint

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    device = torch.device(a.device)
    model, record = load_checkpoint(a.checkpoint, device)
    with open(a.manifest, encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == 'test' and r['class_label'] in record['classes']]
    if not rows:
        raise ValueError('Manifest contains no test records')
    norm = record['normalization']
    pipeline = ImagePipeline(False, norm['mean'], norm['std'])
    classes = record['classes']
    actual = np.array([classes.index(r['class_label']) for r in rows])
    mean_rgb = np.array(norm['mean']) * 255
    scores = []
    with torch.no_grad():
        for severity in range(1, 4):
            for kind in KINDS:
                sampled = []
                for realization in (901, 902, 903) if kind in ('gaussian_noise', 'brightness_contrast', 'square_occlusion') else (901,):
                    probs = []
                    for row in rows:
                        src = (Path(a.image_root) / row['relative_path']).resolve()
                        with Image.open(src) as img:
                            img = ImagePipeline.resize(img.convert('RGB'))
                            w, h = img.size
                            img = img.crop(((w - 224) // 2, (h - 224) // 2, (w + 224) // 2, (h + 224) // 2))
                            pixels = np.asarray(img, dtype=np.uint8)
                        perturbed = corrupt(pixels, kind, severity, per_image_seed(row['sample_id'], kind, severity, realization), mean_rgb)
                        arr = perturbed.astype(np.float32).transpose(2, 0, 1) / 255
                        x = torch.from_numpy((arr - np.asarray(norm['mean'], dtype=np.float32)[:, None, None]) / np.asarray(norm['std'], dtype=np.float32)[:, None, None]).unsqueeze(0).to(device)
                        probs.append(model(x).float().softmax(dim=1)[0].cpu().numpy())
                    metrics = compute_metrics(actual, np.array(probs), classes)
                    sampled.append(metrics['macro_f1'])
                scores.append({'corruption': kind, 'severity': severity, 'macro_f1': float(np.mean(sampled)), 'realizations': len(sampled)})
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'conditions': scores, 'mean_corrupted_macro_f1': float(np.mean([s['macro_f1'] for s in scores]))}, indent=2))
    print(out)
if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Evaluate a saved checkpoint and export per-observation probabilities."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from mcr_tokennet.models import MCRTokenNet, ArchitectureConfig
from mcr_tokennet.models.baselines import create_baseline
from mcr_tokennet.data import ManifestDataset
from mcr_tokennet.evaluation import compute_metrics

def load_checkpoint(path: str, device: torch.device):
    state = torch.load(path, map_location=device, weights_only=True)
    classes = state['classes']
    name = state.get('model_name', 'MCR-TokenNet')
    if name == 'MCR-TokenNet':
        model = MCRTokenNet(len(classes), ArchitectureConfig(**state['architecture']))
    else:
        model = create_baseline(name, len(classes), pretrained=False)
    model.load_state_dict(state['model'], strict=True)
    return (model.to(device).eval(), state)

def evaluate(model, dataset, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    true = []
    probabilities = []
    ids = []
    groups = []
    with torch.no_grad():
        for image, label, sample_id, group_id in loader:
            p = model(image.to(device)).softmax(1).float().cpu().numpy()
            probabilities.extend(p)
            true.extend(label.tolist())
            ids.extend(sample_id)
            groups.extend(group_id)
    return (np.asarray(true), np.asarray(probabilities), ids, groups)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--magnification', choices=['400x', '1000x'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    model, state = load_checkpoint(args.checkpoint, device)
    dataset = ManifestDataset(args.manifest, args.image_root, args.split, state['classes'], state['normalization']['mean'], state['normalization']['std'])
    if args.magnification:
        dataset.filter_rows(magnification=args.magnification)
    y, p, ids, groups = evaluate(model, dataset, device)
    scores = compute_metrics(y, p, state['classes'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    formatted = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in scores.items()}
    (out / 'metrics.json').write_text(json.dumps(formatted, indent=2))
    with (out / 'predictions.csv').open('w', encoding='utf8', newline='') as f:
        cols = ['sample_id', 'group_id', 'y_true', 'y_pred'] + [f'p_{i}' for i in range(len(state['classes']))]
        writer = csv.DictWriter(f, cols)
        writer.writeheader()
        for n, sample_id in enumerate(ids):
            row = {'sample_id': sample_id, 'group_id': groups[n], 'y_true': state['classes'][y[n]], 'y_pred': state['classes'][p[n].argmax()]}
            row.update({f'p_{j}': f'{p[n, j]:.12g}' for j in range(p.shape[1])})
            writer.writerow(row)
    print(json.dumps({'samples': len(y), 'accuracy': scores['accuracy'], 'macro_f1': scores['macro_f1']}, indent=2))
if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Train a model using explicit sample-level manifests and validation selection."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random
from dataclasses import asdict
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
ROOT = Path(__file__).resolve().parents[1]
from mcr_tokennet.models import MCRTokenNet, ArchitectureConfig, parameter_count
from mcr_tokennet.models.baselines import create_baseline, TV, TIMM
from mcr_tokennet.training import ClassBalancedFocalLoss
from mcr_tokennet.data import ManifestDataset, read_dataset_labels
from mcr_tokennet.evaluation import compute_metrics

def stable_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collect(model, loader, device):
    model.eval()
    y = []
    p = []
    ids = []
    groups = []
    with torch.no_grad():
        for x, target, sample_id, group_id in loader:
            probability = model(x.to(device)).float().softmax(1).cpu().numpy()
            p.extend(probability)
            y.extend(target.numpy())
            ids.extend(sample_id)
            groups.extend(group_id)
    return (np.asarray(y, dtype=int), np.asarray(p, dtype=float), ids, groups)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', choices=['NIH Malaria', 'BBBC041', 'Multi-Parasite'], required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--image-root', required=True)
    ap.add_argument('--normalization', required=True, help='JSON produced from TRAINING images ONLY: {mean:[...],std:[...]} values in [0,1]')
    ap.add_argument('--seed', type=int, choices=[42, 123, 3407], required=True)
    ap.add_argument('--model', default='MCR-TokenNet', help='MCR-TokenNet or supported baseline name')
    ap.add_argument('--pretrained', action='store_true', help='Load pretrained weights when supported')
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch-size', type=int)
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--patience', type=int)
    ap.add_argument('--peak-lr', type=float)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--accum', type=int)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--no-balanced-sampler', action='store_true')
    ap.add_argument('--training-config', default=str(ROOT / 'configs/training.yaml'))
    ap.add_argument('--model-config', default=str(ROOT / 'configs/model.yaml'))
    ap.add_argument('--source-magnification', choices=['400x', '1000x'])
    ap.add_argument('--shared-classes', nargs='+', help='Restrict training/evaluation to specified classes')
    args = ap.parse_args()
    tc = yaml.safe_load(Path(args.training_config).read_text())
    mc = yaml.safe_load(Path(args.model_config).read_text())
    for flag, key in [('batch_size', 'batch_size'), ('epochs', 'epochs'), ('patience', 'patience'), ('peak_lr', 'peak_lr'), ('accum', 'accum_steps')]:
        if getattr(args, flag) is None:
            setattr(args, flag, tc[key])
    if args.accum < 1 or args.epochs < 1:
        raise ValueError('Invalid accumulation or epochs')
    stable_seed(args.seed)
    norm = json.loads(Path(args.normalization).read_text(encoding='utf8'))
    classes = args.shared_classes or read_dataset_labels(ROOT / 'configs/expected_class_counts.json', args.dataset)
    for key in ('mean', 'std'):
        if key not in norm or len(norm[key]) != 3:
            raise ValueError('Normalization file needs mean and std RGB triplets')
    if not all((0 <= v <= 1 for v in norm['mean'])):
        raise ValueError('Expected normalized channel means, [0,1]')
    ds = {s: ManifestDataset(args.manifest, args.image_root, s, classes, norm['mean'], norm['std']) for s in ('train', 'val', 'test')}
    if args.source_magnification:
        ds['train'].filter_rows(magnification=args.source_magnification)
        ds['val'].filter_rows(magnification=args.source_magnification)
    if args.shared_classes:
        for dataset in ds.values():
            dataset.filter_rows(classes=args.shared_classes)
    train = ds['train']
    device = torch.device(args.device)
    if str(device) == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested but unavailable')

    def worker_init(worker_id):
        seed = (torch.initial_seed() + worker_id) % 2 ** 32
        random.seed(seed)
        np.random.seed(seed)
    loader_args = dict(num_workers=args.num_workers, worker_init_fn=worker_init, pin_memory=device.type == 'cuda')
    sampler = None if args.no_balanced_sampler else train.balanced_sampler(args.seed)
    trainloader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, **loader_args)
    val = DataLoader(ds['val'], batch_size=32, shuffle=False, **loader_args)
    test = DataLoader(ds['test'], batch_size=32, shuffle=False, **loader_args)
    model_fields = ArchitectureConfig.__dataclass_fields__
    cfg = ArchitectureConfig(**{k: tuple(mc[k]) if isinstance(mc[k], list) else mc[k] for k in model_fields if k in mc})
    if args.model == 'MCR-TokenNet':
        if args.pretrained:
            raise ValueError('MCR-TokenNet was randomly initialized, not pretrained')
        model = MCRTokenNet(len(classes), cfg).to(device)
    else:
        model = create_baseline(args.model, len(classes), pretrained=args.pretrained).to(device)
    loss_fn = ClassBalancedFocalLoss(train.class_counts(), **tc.get('loss', {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=tc['weight_decay'])
    steps = math.ceil(len(trainloader) / args.accum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.peak_lr, epochs=args.epochs, steps_per_epoch=steps, pct_start=tc['pct_start'], div_factor=tc['div_factor'], final_div_factor=tc['final_div_factor'], anneal_strategy='cos')
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0
    best_epoch = -1
    bad = 0
    print('Model parameters:', parameter_count(model))
    (out / 'run_config.json').write_text(json.dumps({'arguments': vars(args), 'model': mc, 'training': tc}, indent=2))
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for idx, (inputs, labels, _, _) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss = loss_fn(model(inputs), labels)
            grad_scale = min(args.accum, len(trainloader) - idx // args.accum * args.accum)
            scaler.scale(loss / grad_scale).backward()
            loss_sum += loss.item() * labels.shape[0]
            if (idx + 1) % args.accum == 0 or idx + 1 == len(trainloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        true, prob, _, _ = collect(model, val, device)
        score = float(f1_score(true, prob.argmax(1), average='macro', zero_division=0))
        hist = {'epoch': epoch + 1, 'val_macro_f1': score, 'train_loss_per_image': loss_sum / len(train)}
        history.append(hist)
        print(json.dumps(hist), flush=True)
        if score > best + 1e-12:
            best = score
            best_epoch = epoch + 1
            bad = 0
            torch.save({'model': model.state_dict(), 'classes': classes, 'architecture': asdict(cfg) if args.model == 'MCR-TokenNet' else {'baseline': args.model}, 'model_name': args.model, 'dataset': args.dataset, 'seed': args.seed, 'epoch': best_epoch, 'val_macro_f1': best, 'normalization': norm}, out / 'best.pt')
        else:
            bad += 1
            if bad >= args.patience:
                break
    (out / 'training_history.json').write_text(json.dumps(history, indent=2))
    state = torch.load(out / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(state['model'])
    y, prob, ids, groups = collect(model, test, device)
    metrics = compute_metrics(y, prob, classes)
    digest = hashlib.sha256((out / 'best.pt').read_bytes()).hexdigest()
    result = {'model': args.model, 'dataset': args.dataset, 'seed': args.seed, 'checkpoint_sha256': digest, 'checkpoint_epoch': best_epoch, 'val_macro_f1': best, 'measured_parameter_count': parameter_count(model), 'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'], 'balanced_accuracy': metrics['balanced_accuracy'], 'macro_auroc': metrics.get('macro_auroc'), 'macro_auprc': metrics.get('macro_auprc')}
    (out / 'test_metrics.json').write_text(json.dumps(result, indent=2))
    with (out / 'test_predictions.csv').open('w', newline='', encoding='utf8') as f:
        fields = ['dataset', 'model', 'seed', 'sample_id', 'group_id', 'y_true', 'y_pred'] + [f'p_{i}' for i in range(len(classes))]
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for i, sid in enumerate(ids):
            row = {'dataset': args.dataset, 'model': args.model, 'seed': args.seed, 'sample_id': sid, 'group_id': groups[i], 'y_true': classes[y[i]], 'y_pred': classes[int(prob[i].argmax())]}
            row.update({f'p_{j}': f'{float(prob[i, j]):.12g}' for j in range(len(classes))})
            writer.writerow(row)
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()

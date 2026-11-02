"""Image transforms and manifest-driven microscopy datasets."""
from __future__ import annotations
from collections import Counter
import csv, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, WeightedRandomSampler

class ImagePipeline:

    def __init__(self, training: bool, mean: list[float], std: list[float]):
        self.training = training
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        if np.any(self.std <= 0):
            raise ValueError('Invalid normalizing standard deviation')

    @staticmethod
    def resize(im: Image.Image) -> Image.Image:
        w, h = im.size
        if min(w, h) <= 0:
            raise ValueError('Empty image')
        factor = 256 / min(w, h)
        return im.resize((round(w * factor), round(h * factor)), Image.Resampling.BILINEAR)

    def __call__(self, im: Image.Image) -> torch.Tensor:
        im = self.resize(im.convert('RGB'))
        w, h = im.size
        if self.training:
            left = random.randint(0, w - 224)
            top = random.randint(0, h - 224)
        else:
            left = (w - 224) // 2
            top = (h - 224) // 2
        im = im.crop((left, top, left + 224, top + 224))
        if self.training:
            if random.random() < 0.5:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.2:
                im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            im = im.rotate(random.uniform(-10, 10), resample=Image.Resampling.BILINEAR)
            im = ImageEnhance.Brightness(im).enhance(random.uniform(0.88, 1.12))
            im = ImageEnhance.Contrast(im).enhance(random.uniform(0.88, 1.12))
            im = ImageEnhance.Color(im).enhance(random.uniform(0.92, 1.08))
        arr = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy((arr - self.mean) / self.std)

class ManifestDataset(Dataset):

    def __init__(self, manifest: str | Path, image_root: str | Path, split: str, classes: list[str], mean: list[float], std: list[float]):
        self.root = Path(image_root).resolve()
        self.classes = list(classes)
        self.mapping = {name: i for i, name in enumerate(classes)}
        with open(manifest, encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not {'sample_id', 'relative_path', 'class_label', 'split'}.issubset(set(reader.fieldnames or [])):
                raise ValueError('Manifest lacks required columns')
            self.rows = [r for r in reader if r['split'] == split and r['class_label'] in self.mapping]
        if not self.rows:
            raise ValueError(f'No records for {split}')
        self.labels = []
        for r in self.rows:
            if r['class_label'] not in self.mapping:
                raise ValueError(f"Unknown class {r['class_label']}")
            self.labels.append(self.mapping[r['class_label']])
            rel = Path(r['relative_path'])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe relative path in manifest')
        self.pipeline = ImagePipeline(split == 'train', mean, std)

    def filter_rows(self, *, magnification: str | None=None, classes: list[str] | None=None) -> None:
        selected = [(row, label) for row, label in zip(self.rows, self.labels) if (magnification is None or row.get('magnification', '').lower().replace('×', 'x') == magnification.lower()) and (classes is None or row['class_label'] in classes)]
        if not selected:
            raise ValueError('No observations after filtering; check manifest magnification/class labels')
        self.rows = [row for row, _ in selected]
        self.labels = [label for _, label in selected]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        p = (self.root / row['relative_path']).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError('Image path escapes root')
        with Image.open(p) as image:
            x = self.pipeline(image)
        return (x, torch.tensor(self.labels[i], dtype=torch.long), row['sample_id'], row.get('group_id', ''))

    def class_counts(self):
        return [Counter(self.labels).get(i, 0) for i in range(len(self.classes))]

    def balanced_sampler(self, seed: int):
        counts = np.bincount(self.labels, minlength=len(self.classes)).astype(float)
        if (counts == 0).any():
            raise ValueError('Missing training class')
        weights = [1.0 / counts[y] for y in self.labels]
        g = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), replacement=True, generator=g)

def read_dataset_labels(class_counts_path, dataset_name):
    import json
    config = json.loads(Path(class_counts_path).read_text(encoding='utf8'))
    return list(config['counts'][dataset_name].keys())

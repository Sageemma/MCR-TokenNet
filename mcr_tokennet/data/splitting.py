"""Sample manifests, stratified partitions, and group-disjointness checks."""
from __future__ import annotations
import csv
from collections import defaultdict, Counter
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, train_test_split
REQUIRED = ('sample_id', 'relative_path', 'class_label', 'group_id')
SPLITS = ('train', 'val', 'test')

def read_manifest(path: str | Path) -> list[dict]:
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if not set(REQUIRED).issubset(reader.fieldnames or []):
            raise ValueError(f'Inventory requires {REQUIRED}')
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Empty inventory or nonunique sample IDs')
    for row in rows:
        rel = Path(row['relative_path'])
        if not row['sample_id'] or rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Missing sample ID or unsafe image path')
    return rows

def validate_splits(rows: list[dict], unit: str) -> dict:
    if any((r.get('split') not in SPLITS for r in rows)):
        raise ValueError('All observations must have train/val/test assignments')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Sample ID appears in multiple partitions')
    if unit in ('patient', 'source_image'):
        groups = defaultdict(set)
        for r in rows:
            if not r.get('group_id'):
                raise ValueError(f'{unit}-aware validation needs group IDs')
            groups[r['group_id']].add(r['split'])
        if any((len(splits) > 1 for splits in groups.values())):
            raise ValueError('Group leakage: an acquisition unit crosses partitions')
    return {split: dict(Counter((r['class_label'] for r in rows if r['split'] == split))) for split in SPLITS}

def generate_new_split(rows: list[dict], unit: str, seed: int=42) -> list[dict]:
    """Generate a seeded partition respecting the selected grouping unit."""
    if unit in ('patient', 'source_image'):
        groups = [r['group_id'] for r in rows]
        if any((not g for g in groups)):
            raise ValueError('Nonempty acquisition-group IDs are required')
        splitter = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)
        train, rest = next(splitter.split(rows, groups=groups))
        other = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
        a, b = next(other.split(rest, groups=[groups[i] for i in rest]))
        val, test = (rest[a], rest[b])
    elif unit == 'image':
        indices = list(range(len(rows)))
        labels = [r['class_label'] for r in rows]
        train, rest = train_test_split(indices, train_size=0.7, random_state=seed, stratify=labels)
        val, test = train_test_split(rest, train_size=0.5, random_state=seed + 1, stratify=[labels[i] for i in rest])
    else:
        raise ValueError('partition unit must be patient, source_image or image')
    result = [dict(row) for row in rows]
    for split, indices in [('train', train), ('val', val), ('test', test)]:
        for index in indices:
            result[int(index)]['split'] = split
    validate_splits(result, unit)
    return result

def write_manifest(rows: list[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys((key for row in rows for key in row)))
    with target.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

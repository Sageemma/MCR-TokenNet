#!/usr/bin/env python3
"""Validate the published manifest counts and metadata (not source image identity)."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((ROOT / 'configs/expected_class_counts.json').read_text(encoding='utf-8'))
NAMES = {'NIH Malaria':'nih_malaria.csv','BBBC041':'bbbc041.csv','Multi-Parasite':'multi_parasite.csv'}
UNITS = {'NIH Malaria':'patient','BBBC041':'source_image','Multi-Parasite':'image'}


def main():
    error_count = 0
    for dataset, filename in NAMES.items():
        with (ROOT / 'splits' / filename).open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            required = {'sample_id', 'relative_path', 'class_label', 'split', 'group_id'}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f'{filename}: missing {required - set(reader.fieldnames or [])}')
            rows = list(reader)
        observed = Counter((r['class_label'], r['split']) for r in rows)
        expected = {(cls, split): int(value) for cls, counts in EXPECTED['counts'][dataset].items() for split, value in zip(EXPECTED['split_order'], counts)}
        if observed != expected:
            error_count += 1
            print(f'FAIL {filename}: class-wise counts differ from configs/expected_class_counts.json')
        sample_ids = [r['sample_id'] for r in rows]
        relpaths = [r['relative_path'] for r in rows]
        if len(set(sample_ids)) != len(rows) or len(set(relpaths)) != len(rows):
            error_count += 1
            print(f'FAIL {filename}: duplicate sample ID or relative path')
        groups = defaultdict(set)
        for r in rows:
            if r['split'] not in {'train', 'val', 'test'} or not r['group_id'] or not r['relative_path']:
                error_count += 1
                print(f'FAIL {filename}: missing or invalid split/group/path')
                break
            groups[r['group_id']].add(r['split'])
        if UNITS[dataset] in {'patient','source_image'} and any(len(v)>1 for v in groups.values()):
            error_count += 1
            print(f'FAIL {filename}: group crosses partitions')
        totals = Counter(r['split'] for r in rows)
        print(f'{filename}: {len(rows):,} records; train {totals["train"]:,}, val {totals["val"]:,}, test {totals["test"]:,}; groups {len(groups):,}')
    if error_count:
        raise SystemExit(f'{error_count} validation checks failed')
    print('PASS: counts, IDs, and declared group separation. Source-image mapping and image-content leakage were NOT verified.')


if __name__ == '__main__':
    main()

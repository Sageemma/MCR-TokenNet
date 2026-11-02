#!/usr/bin/env python3
"""Paired seed-aligned comparison of prediction exports with Holm adjustment."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from mcr_tokennet.evaluation.statistics import paired_macro_f1, holm

def load_csv(path):
    with open(path, encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', nargs='+', required=True, help='Predictions CSV by seed')
    parser.add_argument('--comparators', nargs='+', required=True, help='CSV per seed for one or more comparisons')
    parser.add_argument('--units', choices=['group', 'image'], default='image')
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--bootstraps', type=int, default=2000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reference = [load_csv(p) for p in args.reference]
    comparators = [load_csv(p) for p in args.comparators]
    if len(reference) != len(comparators):
        raise ValueError('Use one comparator CSV per reference seed')
    sample_ids = [row['sample_id'] for row in reference[0]]
    labels = list(dict.fromkeys((row['y_true'] for row in reference[0])))
    for x in reference + comparators:
        if [row['sample_id'] for row in x] != sample_ids:
            raise ValueError('Prediction CSV observation order differs')
        if [row['y_true'] for row in x] != [row['y_true'] for row in reference[0]]:
            raise ValueError('Ground truth or observation order differs')

    def paired(rows):
        return (np.array([[labels.index(row['y_true']) for row in x] for x in rows]), np.array([[labels.index(row['y_pred']) for row in x] for x in rows]))
    units = [r['group_id'] if args.units == 'group' else r['sample_id'] for r in reference[0]]
    if any((not unit for unit in units)):
        raise ValueError('Group-based inference requires group_id for every sample')
    result = paired_macro_f1(paired(reference), paired(comparators), labels, units, bootstraps=args.bootstraps, permutations=args.iterations)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Cross-partition exact-hash and perceptual-hash screening."""
import argparse, csv, json, collections
from pathlib import Path

class BK:

    def __init__(self):
        self.root = None

    def add(self, value, key):
        if self.root is None:
            self.root = [value, [key], {}]
            return
        node = self.root
        while True:
            d = (node[0] ^ value).bit_count()
            if d == 0:
                node[1].append(key)
                return
            if d not in node[2]:
                node[2][d] = [value, [key], {}]
                return
            node = node[2][d]

    def query(self, value, max_dist):
        if self.root is None:
            return []
        q = [self.root]
        res = []
        while q:
            node = q.pop()
            d = (node[0] ^ value).bit_count()
            if d <= max_dist:
                res.extend(((d, key) for key in node[1]))
            q.extend((child for edge, child in node[2].items() if d - max_dist <= edge <= d + max_dist))
        return res

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--threshold', type=int, default=6)
    a = p.parse_args()
    with open(a.manifest, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    for key in ('sample_id', 'split', 'source_sha256', 'decoded_sha256', 'phash', 'relative_path'):
        if any((not x.get(key) for x in rows)):
            raise ValueError('Missing essential field ' + key)
    if not 0 <= a.threshold <= 64:
        raise ValueError('Hamming threshold must be within [0,64]')
    output = Path(a.out)
    output.mkdir(parents=True, exist_ok=True)
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Nonunique sample IDs')
    pairs = {}
    by_sample = {r['sample_id']: r for r in rows}
    for field in ('source_sha256', 'decoded_sha256'):
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        for values in groups.values():
            for i, x in enumerate(values):
                for y in values[i + 1:]:
                    if x['split'] != y['split']:
                        key = tuple(sorted((x['sample_id'], y['sample_id'])))
                        pairs.setdefault(key, {'sample_a': key[0], 'sample_b': key[1], 'hamming': '', 'exact_file': False, 'exact_decoded': False, 'human_review': 'unreviewed'})
                        pairs[key]['exact_file' if field == 'source_sha256' else 'exact_decoded'] = True
    tree = BK()
    for row in sorted(rows, key=lambda r: r['sample_id']):
        v = int(row['phash'], 16)
        for distance, other in tree.query(v, a.threshold):
            x = by_sample[other]
            if x['split'] == row['split']:
                continue
            key = tuple(sorted((other, row['sample_id'])))
            pair = pairs.setdefault(key, {'sample_a': key[0], 'sample_b': key[1], 'hamming': '', 'exact_file': False, 'exact_decoded': False, 'human_review': 'unreviewed'})
            pair['hamming'] = distance
        tree.add(v, row['sample_id'])
    with (output / 'candidate_pairs.csv').open('w', newline='', encoding='utf8') as f:
        wr = csv.DictWriter(f, ['sample_a', 'sample_b', 'hamming', 'exact_file', 'exact_decoded', 'human_review'])
        wr.writeheader()
        wr.writerows(pairs.values())
    summary = {'screened_images': len(rows), 'candidate_pairs': len(pairs), 'exact_match_pairs': sum(bool(p['exact_file'] or p['exact_decoded']) for p in pairs.values()), 'phash_hamming_threshold': a.threshold}
    (output / 'screening_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()

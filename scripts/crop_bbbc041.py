#!/usr/bin/env python3
"""Extract BBBC041 cell crops from bounding-box annotations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

FIELDS = (
    'annotation_id', 'source_image', 'source_image_id', 'class_label',
    'difficult', 'x', 'y', 'w', 'h',
)


def extract_crop(image: np.ndarray, annotation: dict) -> tuple[np.ndarray | None, str]:
    height, width = image.shape[:2]
    x, y, w, h = (float(annotation[key]) for key in ('x', 'y', 'w', 'h'))
    if w <= 0 or h <= 0 or not all(math.isfinite(v) for v in (x, y, w, h)):
        return None, 'invalid_box'
    x0 = max(0.0, x)
    y0 = max(0.0, y)
    x1 = min(float(width), x + w)
    y1 = min(float(height), y + h)
    valid_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if valid_area < 0.75 * w * h:
        return None, 'out_of_bounds'

    left = math.floor(x - 0.1 * w)
    top = math.floor(y - 0.1 * h)
    right = math.ceil(x + 1.1 * w)
    bottom = math.ceil(y + 1.1 * h)
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)

    padded = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )
    crop = padded[top + pad_top:bottom + pad_top, left + pad_left:right + pad_left]
    if crop.size == 0:
        return None, 'invalid_crop'
    return crop, 'kept'


def process_annotations(annotations: Path, image_root: Path, output: Path) -> Counter:
    image_root = image_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    rows = []
    counts = Counter()

    with annotations.open(encoding='utf-8') as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            ann = json.loads(line)
            missing = set(FIELDS) - set(ann)
            if missing:
                raise ValueError(f'Annotation {number} missing {sorted(missing)}')
            annotation_id = str(ann['annotation_id'])
            if annotation_id in seen:
                raise ValueError(f'Duplicate annotation_id: {annotation_id}')
            seen.add(annotation_id)
            if not isinstance(ann['difficult'], bool):
                raise ValueError('difficult must be a boolean')

            record = {key: ann[key] for key in FIELDS[:4]}
            record['crop_relative_path'] = ''
            if ann['difficult']:
                status = 'difficult'
            else:
                relative = Path(ann['source_image'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError(f'Invalid source_image: {relative}')
                source_path = (image_root / relative).resolve()
                if not source_path.is_relative_to(image_root):
                    raise ValueError(f'Source image outside image root: {relative}')
                with Image.open(source_path) as image:
                    pixels = np.asarray(image.convert('RGB'))
                crop, status = extract_crop(pixels, ann)
                if crop is not None:
                    filename = hashlib.sha256(
                        f'{annotation_id}|{relative.as_posix()}'.encode()
                    ).hexdigest()[:24] + '.png'
                    Image.fromarray(crop, 'RGB').save(output / filename)
                    record['crop_relative_path'] = filename
            record['status'] = status
            rows.append(record)
            counts[status] += 1

    fields = [*FIELDS[:4], 'crop_relative_path', 'status']
    with (output / 'crop_records.csv').open('w', encoding='utf-8', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return counts



def crop_from_manifest(manifest: Path, image_root: Path, output: Path,
                       source_map: Path | None = None) -> Counter:
    """Extract BBBC041 crops using fixed manifest boxes and output paths."""
    root = image_root.resolve()
    target_root = output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Source image root is missing: {root}')
    lookup = {}
    if source_map is not None:
        with source_map.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not {'group_id', 'provider_relative_path'}.issubset(reader.fieldnames or []):
                raise ValueError('Source-map CSV requires group_id,provider_relative_path')
            for record in reader:
                group = record['group_id'].strip()
                rel = record['provider_relative_path'].strip()
                if not group or not rel or group in lookup:
                    raise ValueError(f'Blank/duplicate source mapping for {group!r}')
                lookup[group] = rel
    with manifest.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = {'sample_id', 'relative_path', 'group_id', 'source_relative_path',
                  'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'split'}
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError(f'Manifest missing: {sorted(fields-set(reader.fieldnames or []))}')
        records = list(reader)
    if not records:
        raise ValueError('BBBC041 manifest contains no crop records')
    group_to_path = {}
    seen_samples = set()
    seen_targets = set()
    # Check sample and output path uniqueness before extraction.
    prepared = []
    for record in records:
        sid = record['sample_id'].strip()
        gid = record['group_id'].strip()
        if not sid or sid in seen_samples:
            raise ValueError(f'Duplicate/missing sample ID: {sid!r}')
        seen_samples.add(sid)
        if source_map is not None and gid not in lookup:
            raise ValueError(f'Missing original source mapping for {gid}')
        rel_source = (lookup[gid] if source_map is not None else record['source_relative_path'])
        src_rel = Path(rel_source)
        crop_rel = Path(record['relative_path'])
        for rel in (src_rel, crop_rel):
            if rel.is_absolute() or '..' in rel.parts or not str(rel).strip():
                raise ValueError(f'Unsafe manifest path: {rel}')
        src_path = (root / src_rel).resolve()
        crop_path = (target_root / crop_rel).resolve()
        if not src_path.is_relative_to(root) or not crop_path.is_relative_to(target_root):
            raise ValueError('Image path escapes its root')
        if not src_path.is_file():
            raise FileNotFoundError(f'Source microscopy image not found: {src_path}')
        if gid in group_to_path and src_path != group_to_path[gid]:
            raise ValueError(f'Inconsistent source path for group {gid}')
        group_to_path[gid] = src_path
        if crop_path in seen_targets:
            raise ValueError(f'Duplicate manifest crop path: {crop_path}')
        seen_targets.add(crop_path)
        box = {key: record[src] for key, src in (
            ('x','bbox_x'),('y','bbox_y'),('w','bbox_width'),('h','bbox_height'))}
        prepared.append((src_path,crop_path,box))
    # Reuse decoded source images when consecutive crops share a field.
    last_source = None
    pixels = None
    output_count = Counter()
    for src_path, crop_path, box in prepared:
        if src_path != last_source:
            with Image.open(src_path) as im:
                pixels = np.asarray(im.convert('RGB'))
            last_source = src_path
        crop, status = extract_crop(pixels, box)
        if crop is None:
            raise ValueError(f'Manifest contains non-reproducible bounding box: {crop_path} ({status})')
        if crop_path.exists():
            raise FileExistsError(f'Crop already exists (refusing overwrite): {crop_path}')
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop).save(crop_path)
        output_count['kept'] += 1
    return output_count

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--annotations', type=Path,
                      help='Original annotation JSONL for annotation-based cropping')
    mode.add_argument('--manifest', type=Path,
                      help='Existing BBBC041 split CSV with bbox columns and exact crop paths')
    parser.add_argument('--source-map', type=Path,
                        help='Optional source image lookup: group_id,provider_relative_path')
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.manifest is not None:
        counts = crop_from_manifest(args.manifest, args.image_root,
                                    args.out_dir, args.source_map)
    else:
        if args.source_map is not None:
            parser.error('--source-map is only valid with --manifest')
        counts = process_annotations(args.annotations, args.image_root, args.out_dir)
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()

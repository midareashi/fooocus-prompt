import argparse
import ast
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inventory a balanced Fooocus wildprompt benchmark, score Luna face likeness, and make contact sheets.'
    )
    parser.add_argument('--date', required=True, help='Output date folder in YYYY-MM-DD format.')
    parser.add_argument('--after', default='00:00', help='Local inclusive cutoff time, such as 17:30.')
    parser.add_argument('--person-dir', default='input/people/Luna', help='Reference-person directory.')
    parser.add_argument('--report-dir', default='', help='Report directory; defaults beneath reports/.')
    parser.add_argument('--batch-size', type=int, default=16, help='PhotoMaker vision embedding batch size.')
    return parser.parse_args()


def parse_list(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or value.strip() == '':
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(value)
            return result if isinstance(result, list) else []
        except Exception:
            pass
    return []


def read_metadata(path):
    with Image.open(path) as image:
        raw = image.info.get('parameters')
    if isinstance(raw, str) and raw.lstrip().startswith('{'):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def load_wildprompt_rows():
    rows = {}
    root = ROOT / 'wildprompts'
    for path in sorted(root.rglob('*.txt')):
        name = path.relative_to(root).with_suffix('').as_posix()
        rows[name] = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return rows


def reconstruct_resolved_wildprompts(metadata, prompt_rows):
    resolved = metadata.get('resolved_wildprompts')
    if isinstance(resolved, list):
        normalized = []
        for item in resolved:
            if isinstance(item, dict) and item.get('name') and item.get('prompt'):
                normalized.append({'name': str(item['name']), 'prompt': str(item['prompt'])})
        if normalized:
            return normalized, 'metadata'

    prompt = str(metadata.get('prompt', '') or '')
    normalized = []
    for name in parse_list(metadata.get('wildprompts', [])):
        matches = [row for row in prompt_rows.get(str(name), []) if row in prompt]
        if len(matches) == 1:
            normalized.append({'name': str(name), 'prompt': matches[0]})
        elif len(matches) > 1:
            normalized.append({'name': str(name), 'prompt': max(matches, key=len)})
        else:
            normalized.append({'name': str(name), 'prompt': ''})
    return normalized, 'reconstructed'


def inventory_images(date_value, after_value):
    output_dir = ROOT / 'outputs' / date_value
    cutoff = datetime.fromisoformat(f'{date_value} {after_value}').timestamp()
    prompt_rows = load_wildprompt_rows()
    records = []
    for path in sorted(output_dir.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS or path.stat().st_mtime < cutoff:
            continue
        metadata = read_metadata(path)
        resolved, resolved_source = reconstruct_resolved_wildprompts(metadata, prompt_rows)
        shot = next((item for item in resolved if item['name'].startswith('Shots/')), {'name': '', 'prompt': ''})
        location = next((item for item in resolved if item['name'].startswith('Locations/')), {'name': '', 'prompt': ''})
        records.append({
            'path': str(path.resolve()),
            'filename': path.name,
            'created_local': datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec='seconds'),
            'checkpoint': str(metadata.get('base_model', '') or ''),
            'seed': str(metadata.get('seed', '') or ''),
            'prompt': str(metadata.get('prompt', '') or ''),
            'shot_name': shot['name'],
            'shot_prompt': shot['prompt'],
            'location_name': location['name'],
            'location_prompt': location['prompt'],
            'resolved_source': resolved_source,
            'resolved_wildprompts': resolved,
        })
    return records


def person_reference_paths(person_dir):
    person_dir = (ROOT / person_dir).resolve()
    metadata_path = person_dir / 'person.json'
    filenames = []
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            filenames = metadata.get('image_files', []) if isinstance(metadata, dict) else []
        except Exception:
            filenames = []
    paths = [person_dir / name for name in filenames if (person_dir / name).is_file()]
    if not paths:
        paths = sorted(path for path in person_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    return paths


def crop_faces(paths):
    from extras.face_crop import crop_image

    crops = []
    detections = []
    for index, path in enumerate(paths, start=1):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('RGB'))
        crop = crop_image(rgb)
        detected = crop.shape[:2] != rgb.shape[:2]
        crop = np.asarray(Image.fromarray(crop).resize((224, 224), Image.Resampling.LANCZOS))
        crops.append(crop)
        detections.append(detected)
        print(f'[Face crop] {index}/{len(paths)} {Path(path).name}: {"detected" if detected else "fallback"}')
    return np.stack(crops), detections


def load_photomaker_encoder():
    import ldm_patched.modules.clip_vision
    import ldm_patched.modules.utils
    from ldm_patched.contrib.external_photomaker import PhotoMakerIDEncoder

    model_path = ROOT / 'models' / 'photomaker' / 'photomaker-v1.bin'
    model = PhotoMakerIDEncoder()
    state = ldm_patched.modules.utils.load_torch_file(str(model_path), safe_load=True)
    if 'id_encoder' in state:
        state = state['id_encoder']
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(model.load_device)
    return model, ldm_patched.modules.clip_vision.clip_preprocess


@torch.inference_mode()
def embed_crops(model, preprocess, crops, batch_size):
    vectors = []
    for start in range(0, len(crops), batch_size):
        batch = torch.from_numpy(crops[start:start + batch_size].astype(np.float32) / 255.0)
        batch = preprocess(batch.to(model.load_device)).float()
        pooled = model.vision_model(batch)[2].float()
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        vectors.append(pooled.cpu().numpy())
        print(f'[Face embedding] {min(start + batch_size, len(crops))}/{len(crops)}')
    return np.concatenate(vectors, axis=0)


def score_faces(records, reference_paths, batch_size):
    all_paths = [str(path) for path in reference_paths] + [record['path'] for record in records]
    crops, detections = crop_faces(all_paths)
    model, preprocess = load_photomaker_encoder()
    vectors = embed_crops(model, preprocess, crops, batch_size)
    reference_count = len(reference_paths)
    reference_vectors = vectors[:reference_count]
    generated_vectors = vectors[reference_count:]
    centroid = reference_vectors.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    similarities = generated_vectors @ reference_vectors.T
    centroid_scores = generated_vectors @ centroid
    for index, record in enumerate(records):
        record['face_detected'] = bool(detections[reference_count + index])
        record['face_similarity_centroid'] = float(centroid_scores[index])
        record['face_similarity_reference_mean'] = float(similarities[index].mean())
        record['face_similarity_reference_max'] = float(similarities[index].max())
    return {
        'reference_paths': [str(path.resolve()) for path in reference_paths],
        'reference_faces_detected': sum(detections[:reference_count]),
    }


def checkpoint_abbreviation(name):
    lowered = name.casefold()
    if 'cyberrealistic' in lowered:
        return 'CYBER'
    if 'juggernaut' in lowered:
        return 'JUGG'
    if 'lifeisgood' in lowered:
        return 'LIFE'
    if 'lustify' in lowered:
        return 'LUST'
    return name[:8].upper()


def safe_slug(value):
    return ''.join(char.lower() if char.isalnum() else '-' for char in value).strip('-').replace('--', '-')


def make_contact_sheets(records, report_dir):
    sheet_dir = report_dir / 'contact_sheets'
    sheet_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for record in records:
        grouped[(record['shot_name'], record['shot_prompt'])].append(record)

    manifest = []
    for group_index, ((shot_name, shot_prompt), group) in enumerate(sorted(grouped.items()), start=1):
        group.sort(key=lambda item: (item['checkpoint'].casefold(), item['filename']))
        thumb_width, thumb_height = 224, 288
        label_height, prompt_height = 34, 92
        columns = 4
        rows = (len(group) + columns - 1) // columns
        canvas = Image.new('RGB', (columns * thumb_width, prompt_height + rows * (thumb_height + label_height)), 'white')
        draw = ImageDraw.Draw(canvas)
        title = f'{shot_name} | {group_index:02d} | {shot_prompt}'
        wrapped = []
        current = ''
        for word in title.split():
            candidate = f'{current} {word}'.strip()
            if len(candidate) > 115:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        if current:
            wrapped.append(current)
        for line_index, line in enumerate(wrapped[:4]):
            draw.text((8, 6 + line_index * 18), line, fill='black')

        for index, record in enumerate(group):
            column = index % columns
            row = index // columns
            x = column * thumb_width
            y = prompt_height + row * (thumb_height + label_height)
            with Image.open(record['path']) as source:
                source = source.convert('RGB')
                source.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
                tile = Image.new('RGB', (thumb_width, thumb_height), '#222222')
                tile.paste(source, ((thumb_width - source.width) // 2, (thumb_height - source.height) // 2))
            canvas.paste(tile, (x, y))
            score = record.get('face_similarity_centroid')
            score_text = f'{score:.3f}' if isinstance(score, float) else 'n/a'
            label = f'{index + 1}: {checkpoint_abbreviation(record["checkpoint"])} | face {score_text}'
            draw.rectangle((x, y + thumb_height, x + thumb_width, y + thumb_height + label_height), fill='white')
            draw.text((x + 5, y + thumb_height + 8), label, fill='black')

        category = 'upskirt' if 'upskirt' in shot_name.casefold() else 'feet'
        filename = f'{category}_{group_index:02d}_{safe_slug(shot_prompt)[:60]}.jpg'
        path = sheet_dir / filename
        canvas.save(path, quality=92)
        manifest.append({
            'sheet': str(path.relative_to(report_dir)),
            'shot_name': shot_name,
            'shot_prompt': shot_prompt,
            'images': [
                {
                    'position': index + 1,
                    'filename': record['filename'],
                    'checkpoint': record['checkpoint'],
                    'face_similarity_centroid': record.get('face_similarity_centroid'),
                }
                for index, record in enumerate(group)
            ],
        })
    (report_dir / 'contact_sheet_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return manifest


def write_csv(records, report_dir):
    fields = [
        'filename', 'created_local', 'checkpoint', 'seed', 'shot_name', 'shot_prompt',
        'location_name', 'location_prompt', 'resolved_source', 'face_detected',
        'face_similarity_centroid', 'face_similarity_reference_mean',
        'face_similarity_reference_max', 'path', 'prompt',
    ]
    with (report_dir / 'image_scores.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def aggregate(records, key):
    grouped = defaultdict(list)
    for record in records:
        grouped[record[key]].append(record)
    output = []
    for name, group in grouped.items():
        scores = [
            item['face_similarity_centroid']
            for item in group
            if item.get('face_detected') and 'face_similarity_centroid' in item
        ]
        output.append({
            key: name,
            'image_count': len(group),
            'face_detected_count': sum(bool(item.get('face_detected')) for item in group),
            'face_similarity_mean': statistics.fmean(scores) if scores else None,
            'face_similarity_median': statistics.median(scores) if scores else None,
            'face_similarity_min': min(scores) if scores else None,
            'face_similarity_max': max(scores) if scores else None,
        })
    return sorted(output, key=lambda item: item['face_similarity_mean'] or -1, reverse=True)


def main():
    args = parse_args()
    sys.argv = sys.argv[:1]
    report_dir = Path(args.report_dir).resolve() if args.report_dir else \
        ROOT / 'reports' / f'wildprompt-benchmark-{args.date}'
    report_dir.mkdir(parents=True, exist_ok=True)

    records = inventory_images(args.date, args.after)
    if not records:
        raise SystemExit('No matching images found.')
    reference_paths = person_reference_paths(args.person_dir)
    if not reference_paths:
        raise SystemExit('No reference images found.')

    reference_summary = score_faces(records, reference_paths, args.batch_size)
    manifest = make_contact_sheets(records, report_dir)
    write_csv(records, report_dir)
    summary = {
        'date': args.date,
        'after': args.after,
        'image_count': len(records),
        'checkpoint_count': len({record['checkpoint'] for record in records}),
        'shot_prompt_count': len({record['shot_prompt'] for record in records}),
        'reference_summary': reference_summary,
        'checkpoint_face_rankings': aggregate(records, 'checkpoint'),
        'shot_prompt_face_rankings': aggregate(records, 'shot_prompt'),
        'contact_sheet_count': len(manifest),
    }
    (report_dir / 'automated_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

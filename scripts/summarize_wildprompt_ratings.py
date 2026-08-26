import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate manual 0-5 ratings for benchmark contact sheets.')
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--ratings', default='manual_prompt_ratings.json')
    return parser.parse_args()


def aggregate(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key_values, group in grouped.items():
        scores = [row['adherence_score'] for row in group]
        item = {key: value for key, value in zip(keys, key_values)}
        item.update({
            'image_count': len(group),
            'adherence_mean': statistics.fmean(scores),
            'adherence_median': statistics.median(scores),
            'strong_follow_count': sum(score >= 4 for score in scores),
            'strong_follow_rate': sum(score >= 4 for score in scores) / len(scores),
            'failure_count': sum(score <= 1 for score in scores),
            'failure_rate': sum(score <= 1 for score in scores) / len(scores),
        })
        output.append(item)
    return sorted(
        output,
        key=lambda item: (item['adherence_mean'], item['strong_follow_rate'], -item['failure_rate']),
        reverse=True,
    )


def main():
    args = parse_args()
    report_dir = Path(args.report_dir).resolve()
    manifest = json.loads((report_dir / 'contact_sheet_manifest.json').read_text(encoding='utf-8'))
    rating_data = json.loads((report_dir / args.ratings).read_text(encoding='utf-8'))
    rating_by_sheet = {item['sheet'].replace('\\', '/'): item for item in rating_data['ratings']}

    rows = []
    for sheet in manifest:
        sheet_key = sheet['sheet'].replace('\\', '/')
        rating = rating_by_sheet.get(sheet_key)
        if rating is None:
            raise ValueError(f'Missing ratings for {sheet_key}')
        scores = rating.get('scores', [])
        if len(scores) != len(sheet['images']):
            raise ValueError(f'{sheet_key} has {len(sheet["images"])} images but {len(scores)} scores')
        category = 'Upskirt' if 'upskirt' in sheet['shot_name'].casefold() else 'Feet'
        for image, score in zip(sheet['images'], scores):
            if not isinstance(score, int) or not 0 <= score <= 5:
                raise ValueError(f'Invalid score {score!r} for {image["filename"]}')
            rows.append({
                'filename': image['filename'],
                'checkpoint': image['checkpoint'],
                'category': category,
                'shot_name': sheet['shot_name'],
                'shot_prompt': sheet['shot_prompt'],
                'sheet': sheet_key,
                'position': image['position'],
                'adherence_score': score,
                'review_note': rating.get('note', ''),
            })

    with (report_dir / 'manual_adherence_scores.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'scale': rating_data.get('scale', {}),
        'image_count': len(rows),
        'prompt_rankings': {
            'Upskirt': aggregate([row for row in rows if row['category'] == 'Upskirt'], ['shot_prompt']),
            'Feet': aggregate([row for row in rows if row['category'] == 'Feet'], ['shot_prompt']),
        },
        'checkpoint_rankings_overall': aggregate(rows, ['checkpoint']),
        'checkpoint_rankings_by_category': aggregate(rows, ['category', 'checkpoint']),
    }
    (report_dir / 'manual_adherence_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

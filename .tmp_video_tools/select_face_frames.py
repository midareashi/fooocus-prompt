import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


sample_dir = Path(sys.argv[1])
report_dir = Path(sys.argv[2])
report_dir.mkdir(parents=True, exist_ok=True)

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
records = []

for path in sorted(sample_dir.glob("*.jpg")):
    frame = cv2.imread(str(path))
    if frame is None:
        continue
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    faces = cascade.detectMultiScale(
        detect, scaleFactor=1.08, minNeighbors=5, minSize=(45, 45)
    )
    if len(faces) == 0:
        continue
    x, y, fw, fh = max(faces, key=lambda box: box[2] * box[3])
    x, y, fw, fh = (int(v * 2) for v in (x, y, fw, fh))
    pad = int(0.12 * max(fw, fh))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + fw + pad), min(h, y + fh + pad)
    roi = gray[y0:y1, x0:x1]
    sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    brightness = float(np.mean(roi))
    exposure = max(0.0, 1.0 - abs(brightness - 135.0) / 135.0)
    face_ratio = (fw * fh) / (w * h)
    center_x, center_y = x + fw / 2, y + fh / 2
    center_dist = math.hypot(center_x / w - 0.5, center_y / h - 0.5)
    score = (
        2.2 * math.log1p(sharpness)
        + 7.0 * math.sqrt(face_ratio)
        + 1.5 * exposure
        - 0.6 * center_dist
    )
    thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    records.append(
        {
            "path": path,
            "frame": path.stem,
            "x": x,
            "y": y,
            "w": fw,
            "h": fh,
            "sharpness": sharpness,
            "brightness": brightness,
            "face_ratio": face_ratio,
            "score": score,
            "thumb": thumb,
        }
    )

records.sort(key=lambda row: row["score"], reverse=True)

# Retain quality while suppressing consecutive and visually near-identical frames.
chosen = []
for row in records:
    index = int(row["frame"].split("_")[-1])
    duplicate = False
    for kept in chosen:
        kept_index = int(kept["frame"].split("_")[-1])
        visual_delta = float(np.mean(np.abs(row["thumb"].astype(float) - kept["thumb"])))
        if abs(index - kept_index) < 4 or visual_delta < 6.0:
            duplicate = True
            break
    if not duplicate:
        chosen.append(row)
    if len(chosen) >= 30:
        break

with (report_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["rank", "frame", "timestamp_s", "score", "sharpness", "brightness", "face_ratio", "x", "y", "w", "h"],
    )
    writer.writeheader()
    for rank, row in enumerate(chosen, 1):
        writer.writerow(
            {
                "rank": rank,
                "frame": row["frame"],
                "timestamp_s": f"{(int(row['frame'].split('_')[-1]) - 1) / 6:.3f}",
                "score": f"{row['score']:.3f}",
                "sharpness": f"{row['sharpness']:.2f}",
                "brightness": f"{row['brightness']:.2f}",
                "face_ratio": f"{row['face_ratio']:.5f}",
                "x": row["x"], "y": row["y"], "w": row["w"], "h": row["h"],
            }
        )

tile_w, tile_h = 384, 250
cols = 5
rows = math.ceil(len(chosen) / cols)
sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "#111111")
draw = ImageDraw.Draw(sheet)
font = ImageFont.load_default(size=18)
for rank, row in enumerate(chosen, 1):
    image = Image.open(row["path"]).convert("RGB")
    image.thumbnail((tile_w, tile_h - 28), Image.Resampling.LANCZOS)
    col, line = (rank - 1) % cols, (rank - 1) // cols
    ox = col * tile_w + (tile_w - image.width) // 2
    oy = line * tile_h
    sheet.paste(image, (ox, oy))
    label = f"#{rank} {row['frame']}  {(int(row['frame'].split('_')[-1])-1)/6:.2f}s"
    draw.text((col * tile_w + 8, line * tile_h + tile_h - 24), label, fill="white", font=font)

sheet.save(report_dir / "contact_sheet.jpg", quality=92)
print(f"detected={len(records)} chosen={len(chosen)}")

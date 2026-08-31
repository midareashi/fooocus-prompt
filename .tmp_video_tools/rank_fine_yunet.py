import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


sample_dir = Path(sys.argv[1])
report_dir = Path(sys.argv[2])
model_path = Path(sys.argv[3])
fps = 60.0
starts = {"w01": 0.45, "w02": 1.75, "w03": 7.10, "w04": 9.70}
report_dir.mkdir(parents=True, exist_ok=True)
detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.72, 0.3, 5000)
records = []

for path in sorted(sample_dir.glob("*.jpg")):
    window, frame_text = path.stem.split("_")
    source_time = starts[window] + (int(frame_text) - 1) / fps
    frame = cv2.imread(str(path))
    full_h, full_w = frame.shape[:2]
    scale = min(1.0, 1200.0 / max(full_h, full_w))
    detect_w = int(full_w * scale) // 2 * 2
    detect_h = int(full_h * scale) // 2 * 2
    small = cv2.resize(frame, (detect_w, detect_h), interpolation=cv2.INTER_AREA)
    detector.setInputSize((detect_w, detect_h))
    _, faces = detector.detect(small)
    if faces is None:
        continue
    face = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = [int(float(v) / scale) for v in face[:4]]
    confidence = float(face[-1])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[max(0, y):min(full_h, y + fh), max(0, x):min(full_w, x + fw)]
    sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    brightness = float(np.mean(roi))
    exposure = max(0.0, 1.0 - abs(brightness - 135.0) / 135.0)
    face_ratio = (fw * fh) / (full_w * full_h)
    score = 2.7 * math.log1p(sharpness) + 10.0 * math.sqrt(face_ratio) + 1.5 * exposure + confidence
    face_thumb = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
    records.append({"path": path, "time": source_time, "window": window, "x": x, "y": y,
                    "w": fw, "h": fh, "sharpness": sharpness, "brightness": brightness,
                    "confidence": confidence, "score": score, "thumb": face_thumb})

records.sort(key=lambda row: row["score"], reverse=True)
chosen = []
for row in records:
    duplicate = False
    for kept in chosen:
        visual_delta = float(np.mean(np.abs(row["thumb"].astype(float) - kept["thumb"].astype(float))))
        if (row["window"] == kept["window"] and abs(row["time"] - kept["time"]) < 0.18) or visual_delta < 3.0:
            duplicate = True
            break
    if not duplicate:
        chosen.append(row)
    if len(chosen) >= 20:
        break

with (report_dir / "fine_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["rank", "file", "timestamp_s", "score", "sharpness", "confidence", "x", "y", "w", "h"])
    for rank, row in enumerate(chosen, 1):
        writer.writerow([rank, row["path"].name, f"{row['time']:.6f}", f"{row['score']:.3f}",
                         f"{row['sharpness']:.2f}", f"{row['confidence']:.4f}", row["x"], row["y"], row["w"], row["h"]])

tile_w, tile_h, cols = 300, 340, 5
sheet = Image.new("RGB", (tile_w * cols, tile_h * math.ceil(len(chosen) / cols)), "#111")
draw = ImageDraw.Draw(sheet)
font = ImageFont.load_default(size=16)
for rank, row in enumerate(chosen, 1):
    image = Image.open(row["path"]).convert("RGB")
    x, y, fw, fh = row["x"], row["y"], row["w"], row["h"]
    crop = image.crop((max(0, x - fw), max(0, y - fh), min(image.width, x + 2 * fw), min(image.height, y + 2 * fh)))
    crop.thumbnail((tile_w - 8, tile_h - 32), Image.Resampling.LANCZOS)
    col, line = (rank - 1) % cols, (rank - 1) // cols
    ox = col * tile_w + (tile_w - crop.width) // 2
    oy = line * tile_h + (tile_h - 32 - crop.height) // 2
    sheet.paste(crop, (ox, oy))
    draw.text((col * tile_w + 5, line * tile_h + tile_h - 25), f"#{rank} {row['time']:.3f}s", fill="white", font=font)
sheet.save(report_dir / "fine_contact_sheet.jpg", quality=95)
print(f"detected={len(records)} chosen={len(chosen)}")

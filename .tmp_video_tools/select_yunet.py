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
sample_fps = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
report_dir.mkdir(parents=True, exist_ok=True)

detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.72, 0.3, 5000)
records = []

for path in sorted(sample_dir.glob("*.jpg")):
    frame = cv2.imread(str(path))
    if frame is None:
        continue
    full_h, full_w = frame.shape[:2]
    scale = min(1.0, 960.0 / max(full_h, full_w))
    detect_w = max(2, int(full_w * scale) // 2 * 2)
    detect_h = max(2, int(full_h * scale) // 2 * 2)
    detect_frame = cv2.resize(frame, (detect_w, detect_h), interpolation=cv2.INTER_AREA)
    detector.setInputSize((detect_w, detect_h))
    _, faces = detector.detect(detect_frame)
    if faces is None:
        continue
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    face = faces[0]
    x, y, fw, fh = [float(v) / scale for v in face[:4]]
    confidence = float(face[-1])
    x, y, fw, fh = int(x), int(y), int(fw), int(fh)
    x = max(0, x)
    y = max(0, y)
    fw = min(fw, full_w - x)
    fh = min(fh, full_h - y)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[y : y + fh, x : x + fw]
    if roi.size == 0:
        continue
    sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    brightness = float(np.mean(roi))
    exposure = max(0.0, 1.0 - abs(brightness - 135.0) / 135.0)
    face_ratio = (fw * fh) / (full_w * full_h)
    score = 2.4 * math.log1p(sharpness) + 12.0 * math.sqrt(face_ratio) + 2.0 * exposure + confidence
    thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    records.append({
        "path": path, "frame": path.stem, "x": x, "y": y, "w": fw, "h": fh,
        "confidence": confidence, "sharpness": sharpness, "brightness": brightness,
        "face_ratio": face_ratio, "score": score, "thumb": thumb,
    })

records.sort(key=lambda row: row["score"], reverse=True)
chosen = []
for row in records:
    index = int(row["frame"].split("_")[-1])
    if any(abs(index - int(k["frame"].split("_")[-1])) < max(2, int(sample_fps * 0.5)) for k in chosen):
        continue
    chosen.append(row)
    if len(chosen) >= 24:
        break

with (report_dir / "yunet_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
    fields = ["rank", "frame", "timestamp_s", "score", "confidence", "sharpness", "brightness", "face_ratio", "x", "y", "w", "h"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for rank, row in enumerate(chosen, 1):
        idx = int(row["frame"].split("_")[-1])
        writer.writerow({
            "rank": rank, "frame": row["frame"], "timestamp_s": f"{(idx - 1) / sample_fps:.3f}",
            "score": f"{row['score']:.3f}", "confidence": f"{row['confidence']:.4f}",
            "sharpness": f"{row['sharpness']:.2f}", "brightness": f"{row['brightness']:.2f}",
            "face_ratio": f"{row['face_ratio']:.5f}", "x": row["x"], "y": row["y"],
            "w": row["w"], "h": row["h"],
        })

tile_w, tile_h = 260, 300
cols = 6
rows = math.ceil(len(chosen) / cols)
sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "#111111")
draw = ImageDraw.Draw(sheet)
font = ImageFont.load_default(size=16)
for rank, row in enumerate(chosen, 1):
    image = Image.open(row["path"]).convert("RGB")
    x, y, fw, fh = row["x"], row["y"], row["w"], row["h"]
    pad_x, pad_top, pad_bottom = int(fw * 0.75), int(fh * 0.8), int(fh * 1.25)
    crop = image.crop((max(0, x - pad_x), max(0, y - pad_top), min(image.width, x + fw + pad_x), min(image.height, y + fh + pad_bottom)))
    crop.thumbnail((tile_w - 8, tile_h - 32), Image.Resampling.LANCZOS)
    col, line = (rank - 1) % cols, (rank - 1) // cols
    ox = col * tile_w + (tile_w - crop.width) // 2
    oy = line * tile_h + (tile_h - 32 - crop.height) // 2
    sheet.paste(crop, (ox, oy))
    idx = int(row["frame"].split("_")[-1])
    draw.text((col * tile_w + 6, line * tile_h + tile_h - 25), f"#{rank} {row['frame']} {(idx-1)/sample_fps:.2f}s", fill="white", font=font)

sheet.save(report_dir / "yunet_contact_sheet.jpg", quality=94)
print(f"detected={len(records)} chosen={len(chosen)}")

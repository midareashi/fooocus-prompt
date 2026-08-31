import sys
from pathlib import Path

import cv2
from PIL import Image


source_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
model_path = Path(sys.argv[3])
output_dir.mkdir(parents=True, exist_ok=True)
detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.72, 0.3, 5000)


def bounded_window(center, length, limit):
    length = min(int(length), limit)
    start = max(0, min(int(round(center - length / 2)), limit - length))
    return start, start + length


for source in sorted(source_dir.glob("*.png")):
    frame = cv2.imread(str(source))
    full_h, full_w = frame.shape[:2]
    scale = min(1.0, 1400.0 / max(full_h, full_w))
    detect_w = int(full_w * scale) // 2 * 2
    detect_h = int(full_h * scale) // 2 * 2
    small = cv2.resize(frame, (detect_w, detect_h), interpolation=cv2.INTER_AREA)
    detector.setInputSize((detect_w, detect_h))
    _, faces = detector.detect(small)
    if faces is None:
        raise RuntimeError(f"No face detected in {source}")
    face = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = [float(v) / scale for v in face[:4]]

    crop_w = min(full_w, max(1200, int(fw * 2.55)))
    crop_h = min(full_h, int(crop_w * 1.25))
    crop_w = min(crop_w, int(crop_h * 0.8))
    left, right = bounded_window(x + fw / 2, crop_w, full_w)
    desired_top = y - 0.72 * fh
    top, bottom = bounded_window(desired_top + crop_h / 2, crop_h, full_h)

    crop = Image.open(source).convert("RGB").crop((left, top, right, bottom))
    output = output_dir / source.name.replace("native_", "face_hd_")
    crop.save(output, format="PNG", compress_level=3)
    print(f"{source.name}: {full_w}x{full_h} -> {output.name} {crop.width}x{crop.height}")

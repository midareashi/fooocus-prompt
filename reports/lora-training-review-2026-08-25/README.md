# Luna LoRA Training Image Review — 2026-08-25

## Result

- Reviewed: 48 generated image/caption pairs
- Kept in `outputs/2026-08-25`: 29 pairs
  - CyberRealistic XL Desire: 14
  - Juggernaut XL Ragnarok: 15
- Moved to recoverable quarantine: 19 pairs
  - CyberRealistic XL Desire: 10
  - Juggernaut XL Ragnarok: 9
- Every retained PNG still has its matching TXT caption.
- Trainer-ready combined dataset: `datasets/Luna-LoRA-2026-08-25` (43 image/caption pairs: 29 curated generations plus 14 copied likeness references).
- Mean face-likeness score after curation:
  - Desire: 0.8022
  - Juggernaut: 0.7946

The rejected files were moved rather than permanently deleted. They are under `rejected/Desire` and `rejected/Juggernaut`.

## Rejected Desire images

| Image | Reason |
| --- | --- |
| `2026-08-25_23-19-07_3677.png` | Not front-facing as captioned; weaker likeness |
| `2026-08-25_23-19-18_7819.png` | Lowest Desire likeness score; orientation drift |
| `2026-08-25_23-19-44_3929.png` | Failed the precise side-profile prompt |
| `2026-08-25_23-19-55_7771.png` | Hairstyle and clothing do not match the caption |
| `2026-08-25_23-20-17_8626.png` | Noticeable identity drift |
| `2026-08-25_23-21-25_3420.png` | Failed bob haircut and waist-up fashion composition |
| `2026-08-25_23-21-36_3059.png` | Missing denim jacket and intended body pose |
| `2026-08-25_23-22-21_3596.png` | Failed full-body framing and outfit |
| `2026-08-25_23-22-44_5842.png` | Failed rear three-quarter, full-body, and lakeside composition |
| `2026-08-25_23-22-55_3385.png` | Failed perched-on-counter pose and framing |

## Rejected Juggernaut images

| Image | Reason |
| --- | --- |
| `2026-08-25_23-24-19_5247.png` | Weak likeness |
| `2026-08-25_23-24-36_6846.png` | Failed the precise side-profile prompt |
| `2026-08-25_23-24-44_3877.png` | Hairstyle and clothing do not match the caption |
| `2026-08-25_23-24-59_2437.png` | Noticeable identity drift |
| `2026-08-25_23-25-06_3563.png` | Weak likeness plus scene and outfit mismatch |
| `2026-08-25_23-25-14_2408.png` | Missing blazer and intended rooftop styling |
| `2026-08-25_23-25-44_6548.png` | Weak likeness despite correct bob haircut |
| `2026-08-25_23-26-37_5171.png` | Failed rear three-quarter, full-body, and lakeside composition |
| `2026-08-25_23-26-44_3154.png` | Failed perched-on-counter pose |

## Recommended next batch

1. Train a small first-pass LoRA on the curated 29 pairs before generating more. This is enough for a useful identity test and avoids spending time on images that may not be needed.
2. Validate it with a fixed 12-prompt test set that is not present in the training captions: four close-ups, four waist-up images, and four full-body images across neutral studio, daylight, and mixed low light.
3. If likeness is strong in close-ups but weak at distance, generate 8–12 additional clean three-quarter/full-body images. Keep the face at least roughly 300 pixels tall in the source image.
4. For any additional generation, retry only the failed coverage slots: true side profiles, rear three-quarter full-body views, seated/perched poses, and alternate hairstyles. Generate 2–3 seeds per slot and keep one.
5. Use simpler prompts for geometry-heavy poses. The checkpoints followed face, wardrobe, lighting, location, and complex pose inconsistently when all were specified together.
6. Keep Desire and Juggernaut balanced in future additions. The curated likeness averages are close enough that neither checkpoint should dominate the dataset.

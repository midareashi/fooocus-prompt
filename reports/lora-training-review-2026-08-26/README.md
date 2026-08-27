# Luna LoRA Training Image Review — 2026-08-26

## Result

- Reviewed: 96 generated PNGs across 12 composition prompts.
- Checkpoints: 48 CyberRealistic XL Desire and 48 Juggernaut XL Ragnarok.
- Face detection: 96 of 96 images.
- Selected: 10 trainer-ready images, balanced 5 Desire and 5 Juggernaut.
- Selected mean face-likeness score: 0.7955.
- Destination dataset: `datasets/Luna-LoRA-2026-08-25`.
- Every selected PNG has a matching hand-curated TXT caption describing the visible image.

Selection prioritized face likeness and usable composition together. A high face score did not override incorrect framing, pose, or visible artifacts. Captions describe the retained image rather than repeating generation details that were not rendered.

## Selected images

| Image | Checkpoint | Face score | Useful coverage |
| --- | --- | ---: | --- |
| `2026-08-26_18-28-10_6713.png` | Desire | 0.8319 | Seated hotel three-quarter portrait |
| `2026-08-26_18-29-11_3838.png` | Desire | 0.7999 | Waist-up studio portrait, loose waves |
| `2026-08-26_18-31-22_4586.png` | Desire | 0.8195 | Rear three-quarter turn, ponytail |
| `2026-08-26_18-32-39_7989.png` | Desire | 0.7997 | Perched counter pose, high ponytail |
| `2026-08-26_18-34-53_7516.png` | Desire | 0.7930 | Poolside three-quarter portrait |
| `2026-08-26_18-38-20_8956.png` | Juggernaut | 0.7716 | Standing studio three-quarter portrait |
| `2026-08-26_18-39-51_2969.png` | Juggernaut | 0.7937 | Outdoor garden portrait |
| `2026-08-26_18-40-53_4461.png` | Juggernaut | 0.7877 | Seated head-to-knees portrait, braids |
| `2026-08-26_18-41-47_6880.png` | Juggernaut | 0.7962 | Cross-legged floor pose |
| `2026-08-26_18-42-28_3065.png` | Juggernaut | 0.7621 | Bob hairstyle and doorway portrait |

## Exclusions

- The evening full-body set was excluded because none of its eight images preserved full-body framing or the requested standing composition.
- The side-profile set was excluded because none of its eight images produced a true side profile.
- Remaining candidates lost to a stronger face/composition combination within the same coverage slot or introduced redundant framing.

Automated scores and composition contact sheets are retained in this report directory for audit.

## Prompt adjustments

`wildprompts/Training/Luna Face LoRA Dataset v2.txt` was revised after this review:

- Removed the repeated age and demographic prefix from each row so it cannot conflict with the prompt config's identity block.
- Replaced ambiguous `full-body` wording with explicit head-to-toe framing, a pulled-back camera, visible feet, and floor space.
- Made the side-profile request a strict 90-degree profile with only one eye visible.
- Simplified the failed evening scene and made required knees, hands, legs, and counter contact explicit where composition mattered.

## Training configuration review

- Keep the default rank 16 with network alpha 8.
- Keep 8 epochs, saving every epoch. With 54 images, automatic repeats resolve to 5 and the run is approximately 2,160 steps.
- Keep the existing conservative rates: UNet `6e-5`, text encoder `5e-6`, cosine scheduling, and 50 warmup steps.
- Compare epochs 5 through 8 before choosing the installed checkpoint; epoch 6 was strongest in the previous run.
- Expanded epoch previews from 5 generic prompts to a fixed 12-prompt matrix: four close-up, four waist-up, and four full-body tests across studio, daylight, and mixed low light.

# Luna v2 Epoch Evaluation — 2026-08-26

## Verdict

Select **epoch 8**, stored by the trainer as `luna_v2.safetensors`.

Epoch 8 has the highest five-prompt mean face-likeness score, the strongest full-body score, clean anatomy and facial detail, and no visible overtraining. It also outperforms the previous Luna model's selected epoch 6 on the same prompts and seeds.

## Luna v2 epoch ranking

| Rank | Epoch | Mean | Median | Minimum | Maximum |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 0.6814 | 0.7342 | 0.4310 | 0.8049 |
| 2 | 7 | 0.6717 | 0.7001 | 0.4252 | 0.7952 |
| 3 | 6 | 0.6698 | 0.7172 | 0.4084 | 0.7708 |
| 4 | 5 | 0.6600 | 0.6990 | 0.4051 | 0.7979 |
| 5 | 4 | 0.6500 | 0.6794 | 0.4168 | 0.8004 |
| 6 | 2 | 0.6469 | 0.7221 | 0.3937 | 0.7485 |
| 7 | 3 | 0.6343 | 0.6655 | 0.3909 | 0.7821 |
| 8 | 1 | 0.5661 | 0.6260 | 0.2989 | 0.6623 |

All 40 preview faces were detected. The low minimum for every epoch belongs to the full-body prompt, where the face occupies relatively few pixels; it is useful for relative epoch comparison but should not be read as a portrait-quality score.

## Comparison with the previous Luna model

The comparison uses identical prompts, seeds, face detection, reference images, and embedding method.

| Prompt | Luna v1 epoch 6 | Luna v2 epoch 8 | Difference |
| --- | ---: | ---: | ---: |
| Neutral close-up | 0.7946 | 0.7599 | -0.0347 |
| Indoor candid | 0.7194 | 0.7342 | +0.0148 |
| Outdoor portrait | 0.5628 | 0.6772 | +0.1144 |
| Half-body indoor | 0.7969 | 0.8049 | +0.0080 |
| Full-body outdoor | 0.3995 | 0.4310 | +0.0315 |
| **Mean** | **0.6546** | **0.6814** | **+0.0268** |

The v2 dataset achieved its main goal: substantially better identity retention outdoors and modestly better identity at full-body distance, while preserving strong indoor and half-body performance.

## Visual review

- Epochs 1–3 show unstable hair color and weaker identity.
- Epochs 4–7 improve progressively but vary by prompt.
- Epoch 8 is the most consistent overall and leads the indoor candid, half-body, and full-body tests.
- Epoch 6 remains strongest on the outdoor preview alone, but not across the complete suite.
- Epoch 8 shows no obvious texture collapse, facial distortion, anatomy regression, or excessive copying of a training background.

## Limitation and next check

This run used the five preview prompts that were loaded before Fooocus was restarted. The trained weights are unaffected. After installing epoch 8, run the newer fixed 12-prompt suite for broader close-up, waist-up, and full-body validation across studio, daylight, and mixed low light.

Automated results are in `image_scores.csv` and `automated_summary.json`; prompt-by-prompt comparisons are in `contact_sheets/`.

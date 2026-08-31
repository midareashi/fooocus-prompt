# Feet Focus Wildprompt Review

## Scope and settings

- Reviewed batches 903 and 904, image IDs 3374-3517: 9 prompts and 144 images.
- Every image was matched to its prompt through `resolved_wildprompts` metadata.
- Two random-seed images were generated for each prompt/configuration combination.
- Checkpoints: `cyberrealisticXL_desireV30.safetensors` and `juggernautXL_ragnarok.safetensors`.
- LoRA comparisons: no feet LoRA baseline, `Feet_XL.safetensors`, `SeatedSoles_V0.2.safetensors`, and `xuegao-sdxl.safetensors`. Testing LoRAs were applied at the testing-mode weight of 1.0.
- Shared settings: Speed (30 effective steps), 896x1152, guidance 3, `dpmpp_2m_sde_gpu`, Karras, Fooocus V2 + Photograph + Negative, and `StS_age_slider_v1_initial_release.safetensors` at -2.0.
- Earlier batches 894-902 were excluded because they were not part of this adult prompt-adherence run.

Scores use the project 0-3 adherence scale. Each configuration cell is generation A/B. C = CyberRealistic; J = Juggernaut.

| Row | Baseline C | Baseline J | Feet XL C | Feet XL J | SeatedSoles C | SeatedSoles J | xuegao C | xuegao J | Decision |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1. Sideways, one foot foreground | 1/2 | 1/0 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 1/2 | Rewrite |
| 2. Pantyhose feet close-up | 1/1 | 0/0 | 1/1 | 1/1 | 2/1 | 1/1 | 3/3 | 3/3 | Rewrite |
| 3. Feet centered on rug | 0/2 | 0/2 | 0/1 | 0/1 | 2/2 | 2/2 | 2/2 | 2/2 | Rewrite |
| 4. Macro toes on linen | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 2/0 | 2/0 | 1/0 | Rewrite |
| 5. Symmetrical soles nearest lens | 0/0 | 0/0 | 3/3 | 1/3 | 3/2 | 3/3 | 1/3 | 2/3 | Rewrite |
| 6. Full-body backlit tiptoe | 2/1 | 0/0 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | Rewrite |
| 7. Chair, stockinged feet forward | 1/1 | 0/1 | 2/1 | 2/1 | 2/2 | 2/2 | 1/1 | 1/1 | Rewrite |
| 8. Prone with kicked-up soles | 0/2 | 0/2 | 0/0 | 0/0 | 0/0 | 0/0 | 2/2 | 2/2 | Rewrite/retest |
| 9. Armchair, legs over one arm | 1/1 | 0/0 | 1/1 | 1/2 | 2/2 | 2/2 | 1/1 | 1/1 | Remove |

## Findings

- `xuegao-sdxl` was the only configuration that reliably rendered sheer hosiery over the feet. It also handled the prone rear-view concept best.
- `SeatedSoles` was strongest for direct, symmetrical sole compositions and chair poses, but it routinely discarded hosiery and converted unrelated prompts into seated soles-forward portraits.
- `Feet_XL` strongly emphasized feet but frequently collapsed distinct prompts into the same front-facing reclined composition.
- The no-feet-LoRA baseline favored the face and upper body. Explicit framing language is needed to keep feet large and central.
- The macro-on-linen and backlit-tiptoe rows missed their defining compositions almost completely.
- The two chair concepts were redundant, and neither produced the requested armrest drape plus stockings. Row 9 was removed in favor of a stronger, simpler chair composition.

## Revisions

`wildprompts/Shots/Feet Focus.txt` now contains eight rewritten rows. The revisions put the pose and camera relationship first, specify how much of the frame the feet occupy, and use concrete spatial anchors such as sofa, rug, white linen, floor, chair, and mattress.

Both focused Audit files contain exactly these eight revised prompts. They require a fresh two-image-per-row retest before the rewrites are considered proven.

## Review images

The nine labeled comparison sheets are in [contact_sheets](contact_sheets). Each sheet contains all 16 images for one original prompt with image ID, LoRA, checkpoint, and seed labels.

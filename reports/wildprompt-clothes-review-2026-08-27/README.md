# Clothes Wildprompt prompt-adherence review — 2026-08-27

## Result

The tested Clothes prompts were strongly understood: 49 of 51 prompts produced the correct general clothing concept in both generations. Across 102 images, the mean manual adherence score was 2.80/3; 86 images scored 3, 12 scored 2, and 4 scored 1. No image was a complete miss.

Five prompts were revised. Two failed their central visible feature in both images, while three delivered the general concept but missed the feature that made the row distinct.

## Scope and settings

History metadata identified 102 images: two random-seed generations for each of 51 resolved prompt rows.

| Setting | Value |
|---|---|
| Checkpoint | `cyberrealisticXL_desireV30.safetensors` |
| Performance | Speed |
| Steps | 10 |
| Resolution | 896×1152 |
| Sampler / scheduler | `dpmpp_2m_sde_gpu` / Karras |
| Guidance | 3 |
| Styles | Fooocus V2, Fooocus Negative |

The batch covered all 51 rows in the current Clothes library: Babydolls and Lace, Bikinis and Swimwear, Corsets and Bodysuits, Dresses, Emerald Green Qipao, Legwear, Lingerie, School Uniforms, Skirts, Tiny Dresses, and Tops.

Scores measure prompt comprehension only. Anatomy, faces, hands, smudging, and incidental low-step artifacts were ignored.

## Scores

Rows appear in the same order as their Wildprompt files. `3` is strong, `2` is good with missing secondary details, and `1` is weak or confused.

| File | Row scores (A/B) | Result |
|---|---|---|
| Babydolls and Lace | 3/3, 3/2, 3/3, 3/3, 3/3 | Keep all |
| Bikinis and Swimwear | 3/3, 2/2, 3/3, 3/3, 3/3 | Keep all; retro high waist was softened but the concept remained clear |
| Corsets and Bodysuits | 2/3, 3/3, 3/3, 3/3, 3/3 | Keep all; first row leaned toward a separate corset and panties but remained on concept |
| Dresses | 3/3, 3/3, 3/3, 3/3, 3/3 | Keep all |
| Emerald Green Qipao | 3/3 | Keep |
| Legwear | 3/3, 3/3, 3/3, 3/3, **1/1** | Rewrite row 5 |
| Lingerie | 3/3, 3/3, 3/3, 3/3, 3/3 | Keep all |
| School Uniforms | 3/3, 3/3, **2/2**, 3/3, 3/3 | Rewrite row 3 |
| Skirts | 3/3, 3/3, 3/3, **1/1**, **2/2** | Rewrite rows 4 and 5 |
| Tiny Dresses | 3/3, 3/3, 3/3, 3/3, 3/3 | Keep all |
| Tops | 2/2, **2/2**, 3/3, 3/3, 3/3 | Rewrite row 2 |

## Revisions

### Legwear row 5

Both images appeared bare-legged; “barely visible weave” and “sandalfoot” weakened the main pantyhose concept.

Revised to:

> wearing visibly sheer tan pantyhose covering her legs and feet, with a glossy finish and reinforced toes

### School Uniforms row 3

Both images produced the blouse, burgundy tie, and pleated skirt, but neither clearly produced the vest that distinguishes this row.

Revised to:

> wearing an adult academy roleplay uniform with a sleeveless fitted charcoal sweater vest layered over a crisp white blouse, a burgundy tie, and a pleated skirt

### Skirts row 4

Both images put buttons on the shirt while producing an ordinary denim bottom. The prompt did not anchor the button placement strongly enough.

Revised to:

> wearing a faded blue denim miniskirt with five brushed-silver buttons running vertically down the center front and a raw-cut hem

### Skirts row 5

Both images understood black leather and a short high-waisted skirt, but produced a narrow zip-front silhouette rather than a clear A-line shape.

Revised to:

> wearing a short flared black leather A-line skirt with a fitted high waist and a polished finish

### Tops row 2

Both images understood sheer black fabric and visible lingerie, but the blouse shape, sleeves, and button-front construction were inconsistent.

Revised to:

> wearing a long-sleeved sheer black chiffon button-up blouse with a pointed collar, fitted cuffs, and a visible black lace bralette underneath

## Evidence and next step

The side-by-side category sheets are in [contact_sheets](contact_sheets), and the History-derived image-to-prompt mapping is in [manifest.json](manifest.json).

The five revisions require a focused two-image retest before their improvement is considered proven. The repeatable review procedure is documented in [Wildprompt Prompt-Adherence Testing](../WILDPROMPT_PROMPT_TESTING.md).

# Pantyhose LoRA Review — 2026-08-27

## Result

Batch 119 contains 66 images: 11 LoRAs, three prompt variants, and two random seeds per variant. The overall winner is `glossy_pantyhose_SDXL_epoch_15`; it produced unmistakable foot-covering pantyhose in black, nude, and charcoal while preserving the requested dress and composition. `GlossyPantyhose_XL` is the close runner-up and has the strongest glossy editorial character.

The scoring judges the requested general image, with special attention to whether the garment reads as continuous waist-to-foot pantyhose rather than bare legs, stockings, or thigh-highs. Incidental anatomy and face defects were ignored.

## Settings

| Setting | Value |
|---|---|
| Batch | 119 |
| Checkpoint | `cyberrealisticXL_desireV30.safetensors` |
| LoRA weight | 1.0 |
| Performance / steps | Speed / 30 |
| Resolution | 896×1152 |
| Sampler / scheduler | `dpmpp_2m_sde_gpu` / Karras |
| Guidance | 3 |
| Styles | Fooocus Negative |

## Scores

Each cell contains the two seed scores on the 0–3 prompt-adherence scale.

| LoRA | Black | Nude | Charcoal | Total | Verdict |
|---|---:|---:|---:|---:|---|
| `glossy_pantyhose_SDXL_epoch_15` | 3/3 | 3/3 | 3/3 | **18/18** | Best overall; reliable across all colors |
| `GlossyPantyhose_XL` | 3/2 | 3/3 | 3/3 | **17/18** | Close runner-up; excellent shine, one tighter crop |
| `Pantyhose-000009` | 3/3 | 2/2 | 3/3 | **16/18** | Strong dark specialist; nude becomes pale/opaque |
| `nude` | 3/2 | 2/2 | 3/3 | **15/18** | Surprisingly strong dark hosiery; nude is subtle |
| `perfectpantyhose-a` | 3/3 | 1/1 | 3/3 | **14/18** | Strong black/charcoal specialist; nude reads bare |
| `RealPantyhose_XL` | 3/3 | 1/2 | 2/3 | **14/18** | Natural and subtle, but inconsistent on light tones |
| `sheerpantyhose` | 3/2 | 2/2 | 2/3 | **14/18** | Useful sheer specialist; understated rather than glossy |
| `pantyhose_xl_v1` | 3/3 | 0/0 | 3/3 | **12/18** | Excellent dark specialist; nude becomes mismatched stockings/thigh-highs |
| `Pantyhose_1` | 3/3 | 2/2 | 0/0 | **10/18** | Black/nude usable; charcoal consistently becomes bare legs |
| `Lycra_Pantyhose` | 1/1 | 1/0 | 1/1 | **5/18** | Overpowers clothing and exposes toes; reads as bodysuit/stockings |
| `phose_SDXL_v3.TA_trained` | 1/1 | 1/1 | 0/0 | **4/18** | Mostly bare legs; does not reliably activate the garment |

## Prompt winners

- **Black:** `glossy_pantyhose_SDXL_epoch_15`, `Pantyhose-000009`, `Pantyhose_1`, `pantyhose_xl_v1`, `perfectpantyhose-a`, and `RealPantyhose_XL` all delivered two clear results. `GlossyPantyhose_XL` also rendered the garment strongly, but one image tightened the crop instead of giving the requested full-body view.
- **Nude:** `glossy_pantyhose_SDXL_epoch_15` and `GlossyPantyhose_XL` are the only fully convincing two-seed winners. Images 390–391 are especially clear because the sheen and covered feet distinguish pantyhose from bare skin.
- **Charcoal:** `glossy_pantyhose_SDXL_epoch_15`, `GlossyPantyhose_XL`, `nude`, `Pantyhose-000009`, `pantyhose_xl_v1`, and `perfectpantyhose-a` all hit both seeds strongly.

## Recommendations

1. Keep `glossy_pantyhose_SDXL_epoch_15` as the default all-purpose choice.
2. Keep `GlossyPantyhose_XL` when a stronger glossy/editorial finish is wanted.
3. Keep `sheerpantyhose` or `RealPantyhose_XL` as subtler alternatives.
4. Keep `pantyhose_xl_v1`, `perfectpantyhose-a`, and possibly `Pantyhose-000009` as dark-color specialists rather than general-purpose models.
5. Do not use `Lycra_Pantyhose` or `phose_SDXL_v3.TA_trained` for this workflow without a separate trigger/weight investigation.

## Evidence

- [Black prompt contact sheet](contact_sheets/prompt_1.jpg)
- [Nude prompt contact sheet](contact_sheets/prompt_2.jpg)
- [Charcoal prompt contact sheet](contact_sheets/prompt_3.jpg)
- [History-derived manifest](manifest.json)

No Wildprompt wording was changed during this review. The failures followed individual LoRAs while the same prompts succeeded with other LoRAs, so prompt revision would hide the actual comparison rather than improve it.

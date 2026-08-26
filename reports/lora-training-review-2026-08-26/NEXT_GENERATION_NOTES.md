# Luna LoRA V2 generation handoff

## What the first LoRA taught us

- Close-up and normal portrait likeness are good with both CyberRealistic XL Desire and Juggernaut XL Ragnarok.
- Likeness falls off noticeably in head-to-knees and full-body compositions, especially when the camera is several meters away.
- The curated V1 dataset contains 43 pairs, but its strongest identity coverage is concentrated in close-up and waist-up images.
- Vanilla SDXL 1.0 training previews sometimes looked childlike even though the sample prompts specified `adult woman`. The same prompts stayed adult with Desire and Juggernaut, so judge the LoRA on the checkpoints used for production.
- Desire has a stronger cleavage or unintended-nudity bias. Reject outputs that do not follow the requested wardrobe.

## First test: LoRA plus Person Likeness

Use the saved `Luna` Person Likeness profile. Keep the prompt, seed, checkpoint, and resolution fixed while changing only one identity control at a time.

Suggested compact grid:

1. LoRA 0.8, Person Likeness off.
2. LoRA 1.0, Person Likeness off.
3. LoRA 0.8, Identity Strength 0.8, Face Weight 0.0.
4. LoRA 0.8, Identity Strength 1.0, Face Weight 0.0.
5. LoRA 0.8, Identity Strength 1.0, Face Weight 0.5, Face Weight Start At 0.55.

Run the grid on one close-up, one head-to-knees portrait, and one full-body portrait with both Desire and Juggernaut. Tag the batches `Luna LoRA Test`, `Person Likeness`, the checkpoint name, and the LoRA epoch.

## V2 candidate generation

Use `Training/Luna Face LoRA Dataset v2` for the missing distance and body coverage. Generate 2-3 seeds per prompt and retain only one strong candidate from each prompt.

### Aspect-ratio coverage

The current 43-image V1 dataset contains portrait-orientation images only. The trainer has aspect-ratio bucketing enabled with `bucket_no_upscale`, so landscape, portrait, and square source images can be included without forcing everything into a square crop.

For the 8-12 accepted V2 additions, aim for approximately:

- 5-7 portrait images, primarily for head-to-knees and full-body coverage;
- 2-3 square images with strong face or upper-body identity;
- 1-2 landscape images for seated poses or wider environments where Luna still occupies a substantial part of the frame.

Aspect-ratio variety helps prevent the LoRA from associating Luna only with vertical compositions, but it is secondary to face size and clarity. Do not keep a landscape image when distance makes the face tiny or soft. Prefer genuinely different images and poses rather than creating several crops of the same source image.

For generating candidates with the existing LoRA, its current trigger remains `girl_named_luna`. For a clean retraining run, use a neutral unique trigger such as `luna_person` instead.

Keep a candidate only when:

- the subject is unambiguously an adult woman;
- the face matches Luna without relying on hairstyle alone;
- both eyes and the complete face are sharp and unobstructed, except for the intentional side-profile slot;
- the framing, pose, clothing, hands, and location agree with the caption;
- the face remains large enough to judge at the source resolution;
- there are no duplicated poses, near-identical seeds, or strong checkpoint-specific beauty drift.

Reject candidates with a tiny or smeared face, childlike appearance, unintended nudity, warped hands, occluded features, or a caption mismatch. Keep Desire and Juggernaut reasonably balanced.

## Retraining recommendation

Do not replace the curated V1 dataset wholesale. Add 8-12 accepted V2 images to it, emphasizing head-to-knees and full-body coverage. Retrain from the combined curated set, compare epochs with fixed seeds on the actual production checkpoints, and choose the checkpoint that preserves identity at distance without overfitting clothing or hair.

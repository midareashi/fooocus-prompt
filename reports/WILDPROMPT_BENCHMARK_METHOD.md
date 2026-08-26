# Repeatable Luna checkpoint and wildprompt benchmark

This procedure produces a balanced checkpoint comparison, automated Luna face-likeness scores, prompt-by-prompt contact sheets, and reproducible manual adherence rankings.

## What the scripts measure

Two metrics are intentionally kept separate:

1. **Face likeness:** how close the generated face is to the images in `input/people/Luna`.
2. **Prompt adherence:** whether the exact selected shot wildprompt was visibly followed.

Do not merge them too early. A checkpoint can preserve Luna's face while ignoring the pose, or follow the pose while changing her identity.

## 1. Design a fair generation batch

Use the same settings for every checkpoint. Only change the checkpoint itself.

Recommended setup:

- Select all checkpoints under test in Multi-Checkpoint mode.
- Use the same Luna person-likeness references and slider values.
- Select one shot wildprompt file, such as `Shots/Feet Focus` or `Shots/Consensual Upskirt`.
- Mark only the shot file for **Generate All**.
- Keep any location or lighting files in random mode so they do not multiply the queue.
- Use Image Number 2 or more so every prompt/checkpoint pair has repeated seeds.
- Keep resolution, styles, sampler, scheduler, steps, CFG, sharpness, VAE, LoRAs, and negatives fixed.
- Record the local start time or run the benchmark in an otherwise empty output date folder.

For a paired test, each seed must be repeated across every checkpoint. The 2026-08-25 benchmark used 48 unique seeds, each appearing once under all four checkpoints.

Restart Fooocus after modifying Python code. New builds store `resolved_wildprompts` in image/history metadata. The evaluator can reconstruct older selections when the exact wildprompt row still appears verbatim in the final prompt, but direct metadata is safer.

## 2. Run the automated evaluator

From the Fooocus project root:

```powershell
D:\Fooocus\python_embeded\python.exe scripts\evaluate_wildprompt_benchmark.py `
  --date 2026-08-25 `
  --after 17:30 `
  --person-dir input\people\Luna `
  --report-dir reports\wildprompt-benchmark-2026-08-25 `
  --batch-size 24
```

Change `--date`, `--after`, and `--report-dir` for a new run. The cutoff is inclusive and uses local file modification time.

The evaluator:

1. inventories images in `outputs/<date>` at or after the cutoff;
2. reads checkpoint, seed, prompt, and wildprompt metadata;
3. reconstructs an older resolved row by exact substring matching when needed;
4. aligns the largest face with Fooocus's face cropper;
5. embeds faces using the PhotoMaker vision encoder already installed with Fooocus;
6. compares every generated embedding with the centroid of all Luna reference embeddings;
7. writes per-image scores and checkpoint aggregates;
8. creates one eight-image contact sheet per shot prompt.

Generated files:

- `image_scores.csv`
- `automated_summary.json`
- `contact_sheet_manifest.json`
- `contact_sheets/*.jpg`

### Face-score interpretation

The score is cosine similarity, not a percentage or an aesthetic rating. Use it for relative comparison within a controlled batch.

- Report face-detection counts beside every aggregate.
- Exclude undetected faces from the primary likeness mean.
- Keep medians, spread, and low-percentile behavior; a high maximum alone does not make a checkpoint reliable.
- A deliberate foot macro may correctly contain no face. That is a prompt success and a face-likeness non-result, not a bad face.
- Prefer paired comparisons by seed. Bootstrap the per-seed score differences when top checkpoints are close.

## 3. Review prompt adherence

Open every generated contact sheet at full size. Positions are listed in `contact_sheet_manifest.json`; do not assume checkpoint order if the checkpoint set changes.

Score each image against only the selected shot prompt:

| Score | Meaning |
|---:|---|
| 5 | Subject, pose, camera framing, focal detail, and required reveal are all clear |
| 4 | Main intent is strong with one meaningful deviation |
| 3 | Intent is recognizable, but framing or a required detail is weak/missing |
| 2 | Related generic composition; most distinctive requirements are absent |
| 1 | Only a superficial keyword remains |
| 0 | Complete failure or contradiction |

For feet prompts, look specifically at foot prominence, requested perspective, bare/stocking state, toe/sole orientation, and pose. For upskirt prompts, look at camera position, skirt presence, reveal, body orientation, and specified action. Do not raise the adherence score merely because an image is explicit or attractive.

Copy `manual_prompt_ratings.json` from a previous report as a template, update each sheet path, enter one score per listed image position, and record a concise rationale. Keeping the rationales makes later rescoring auditable.

## 4. Aggregate the manual scores

```powershell
D:\Fooocus\python_embeded\python.exe scripts\summarize_wildprompt_ratings.py `
  --report-dir reports\wildprompt-benchmark-2026-08-25
```

This validates that every sheet and image has exactly one integer score from 0 through 5, then creates:

- `manual_adherence_scores.csv`
- `manual_adherence_summary.json`

Rank by mean score, then 4–5 hit rate, then the inverse 0–1 failure rate. When two checkpoint means differ by less than roughly 0.05 on this subjective scale, describe them as effectively tied and explain their hit/failure tradeoff.

## 5. Write the result README

Include:

- image counts and per-checkpoint balance;
- all fixed generation settings;
- face mean, median, spread, detection count, and uncertainty;
- separate prompt-adherence rankings for each category;
- every individual wildprompt ranking with a contact-sheet link;
- best individual examples;
- a practical recommendation for face-only, prompt-only, and balanced use;
- limitations and any exclusions.

The 2026-08-25 result format is available in [the completed benchmark](wildprompt-benchmark-2026-08-25/README.md).

## Common mistakes

- Comparing checkpoints with different seeds or settings.
- Letting Generate All multiply locations, clothes, lighting, and poses at once.
- Treating an embedding score as beauty or image quality.
- Rewarding generic nudity when the requested camera geometry failed.
- Penalizing a correct close-up prompt because it intentionally omits the face.
- Ranking from one cherry-picked image instead of all repeated samples.
- Editing wildprompt files between generation and metadata reconstruction.
- Forgetting to restart Fooocus after metadata-code changes.

## Improving confidence

For a more definitive test, repeat the batch on another day with at least four seeds per prompt/checkpoint, rotate locations, and keep a blinded reviewer unaware of checkpoint labels. Combine batches only after confirming the same settings and prompt text were used.

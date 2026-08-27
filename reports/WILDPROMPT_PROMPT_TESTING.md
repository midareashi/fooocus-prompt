# Wildprompt Prompt-Adherence Testing

Use this procedure whenever the user asks to "test these prompts" or requests a Wildprompt prompt-adherence review.

## Purpose

Determine whether each Wildprompt reliably creates the requested general image. This is a prompt-comprehension test, not an image-quality test.

Ignore incidental low-step defects such as malformed hands, extra fingers, imperfect faces, minor smudging, and background artifacts. Only consider them when they make the requested subject impossible to identify.

## Test batch

- Generate every row in each requested Wildprompt file.
- When testing multiple Wildprompt files together, enable **Test selected files separately** so each row generates independently instead of combining rows from different files.
- Use two generations per row with different random seeds.
- A quick 10-step generation is sufficient for prompt screening.
- Keep the checkpoint, base prompt, resolution, styles, and other settings consistent throughout the batch.
- Read `resolved_wildprompts` from image metadata or History so every image is matched to the row actually used. Do not infer the row from file order alone.

## Review order

Judge these attributes in descending importance:

1. Main subject, garment, pose, or scene type.
2. Overall cut, silhouette, composition, or spatial relationship.
3. Important color.
4. Important material or visual treatment.
5. One or two defining secondary details.

Do not require every adjective or tiny construction detail to appear. Ask whether a viewer would describe the generated image in roughly the same terms as the prompt.

## Scoring

Score both images independently.

| Score | Meaning |
|---:|---|
| 3 | Strong: the intended general image and defining traits are immediately clear. |
| 2 | Good: the central concept is correct, but secondary details are missing or softened. |
| 1 | Weak: the prompt is only partly understood, generic, or confused with another concept. |
| 0 | Miss: the central concept is ignored or contradicted. |

## Decisions

- **Keep:** Both images score at least 2 and the prompt creates a useful, distinct result.
- **Rewrite:** A worthwhile concept is inconsistent, both images miss its defining feature, or either image scores below 2.
- **Remove:** Both images score 0-1, or a stronger prompt already produces the same result.
- **Retest:** The two results strongly disagree, especially a 3 paired with a 0 or 1.

When rewriting, put the main concept first, use concrete visible language, remove low-value details, and make the failed defining feature explicit. Do not change a prompt merely because of anatomy or other low-step rendering defects.

## Focused retest wildcard

After modifying any Wildprompt rows, overwrite both `wildprompts/Tests/Audit.txt` and `wildcards/tests/audit.txt` with only the latest revised prompt lines. Do not append to the previous audit. These files are the reusable focused-retest queue, so every prompt-adherence review must replace their contents even when older audit files already exist.

- Keep both audit files identical, with one complete revised prompt per line.
- Include only prompts changed during the latest review.
- Keep the same row wording used in the source Wildprompt files.
- If the review makes no prompt changes, overwrite both files with empty files.
- Select `Tests/Audit`, enable **Test selected files separately**, generate two random-seed images for every row, and score them with the same rubric before considering the revisions proven.

## Report

For every tested row, record:

| File and row | Generation A | Generation B | Decision | Evidence |
|---|---:|---:|---|---|
| Example | 3 | 2 | Keep | Main concept is clear; one secondary detail is absent. |

The final report must state:

- Test settings and exact scope.
- How many prompts and images were reviewed.
- Which requested files were missing from the batch or otherwise untested.
- Overall adherence results.
- Every rewritten or removed prompt and the visual evidence for the change.
- Where the contact sheets or source images can be reviewed.

After revisions, use `Tests/Audit` to run a focused two-image retest of each changed prompt before treating the rewrite as proven.

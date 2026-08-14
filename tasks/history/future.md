# History System - Future Work

## Search And Review

- Full text search over prompts, negative prompts, notes, LoRA names, checkpoint names, and tags.
- Saved filters/views, such as "best LoRA test runs", "needs review", "favorite portraits", or "missing files".
- Combined filters for checkpoint, LoRA, seed, favorite, review status, and tag.
- Sort modes:
  - newest first
  - oldest first
  - seed
  - checkpoint
  - LoRA
  - rating/favorite
  - similarity group

## Batch Comparison Tools

- Side-by-side LoRA comparison layout for testing mode.
- Matrix view:
  - rows = seed/image index
  - columns = testing LoRA
  - optional checkpoint grouping above columns
- Lock zoom/pan across compared images.
- Mark best image per row, per LoRA, or per batch.
- Promote selected image settings to a new prompt config.

## Data Maintenance

- Missing file scanner that marks DB rows as missing without deleting history.
- Output folder re-query that reconciles the DB with the configured outputs folder by adding new image records and removing records for deleted files, while leaving existing records unchanged.
- "Relink file" workflow for moved images.
- Export selected batches to a portable folder with image files and JSON metadata.
- Import exported batches from another Fooocus install.
- Compact/backup database from UI.
- Automatic periodic backup of `outputs/history.sqlite3`.

## Prompt Config Improvements

- Store prompt configs in SQLite while keeping JSON export/import.
- Version prompt configs when overwritten.
- Track which image or batch a prompt config came from.
- Add prompt-config tags and notes.
- Add diff view between current UI settings and selected history/config entry.

## Quality And Curation

- Batch-level favorite/reject/needs-review flags.
- Batch-level star rating.
- Free-form notes for batches.
- "Winner" marker for LoRA/checkpoint testing batches.
- Optional keyboard shortcuts for review workflows.

## Advanced Metadata

- Store model hashes, LoRA hashes, file sizes, and image dimensions.
- Record generation timing per image and per batch.
- Record VRAM mode/performance information when available.
- Record errors per batch/image so failed generations are visible in history.
- Store prompt expansion and final positive/negative conditioning text separately.

## Optional Integrations

- CLIP/aesthetic embeddings for visual similarity search.
- Perceptual hash to detect duplicates.
- Thumbnail cache for fast history browsing.
- External editor/open-with hooks.

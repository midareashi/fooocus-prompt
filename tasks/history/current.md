# History System - Current Work

## Objective

Replace the fragile generated `log.html` history with a durable local history system backed by SQLite. The new system should track batches, images, prompts, seeds, checkpoints, LoRAs, wildprompts, settings, tags, notes, favorites, deleted/missing files, and reloadable generation configs without depending on metadata embedded inside image files.

## Problems To Solve

- `outputs/YYYY-MM-DD/log.html` breaks or hides entries when image files are deleted.
- History is currently day/file oriented, not batch oriented.
- Reloading configs often requires parsing image metadata or generated HTML instead of querying a stable local record.
- Session gallery state is in-memory and does not survive page reloads.
- Tags, notes, favorites, LoRA comparisons, checkpoint comparisons, and batch grouping are not first-class concepts.
- The app can generate a lot of variants, but later review is weak: filtering is limited, there is no durable comparison workflow, and deleted/moved images are not represented clearly.

## Existing Code Touchpoints

- `modules/private_logger.py`
  - Saves images.
  - Embeds image metadata.
  - Writes `log.html`.
  - Should be reduced to image persistence plus optional legacy HTML compatibility.

- `modules/async_worker.py`
  - Builds generation metadata in `save_and_log`.
  - Calls `log(...)`.
  - Registers in-memory config with `register_generated_image_config`.
  - Best insertion point for durable image and batch records.

- `webui.py`
  - Shows session gallery.
  - Loads config from selected generated images by first checking in-memory config, then parsing image metadata.
  - Has prompt config save/load controls.
  - Needs a new persistent History tab/view and should query SQLite first.

- `modules/meta_parser.py`
  - Parses embedded Fooocus/A1111 metadata.
  - Useful for migration/import from old images, but should not be required for new generated images.

- `modules/prompt_config.py`
  - Stores saved prompt configs as JSON files.
  - Can remain initially, but should eventually sync with or migrate into SQLite.

## Proposed Data Store

Create `modules/history_db.py` backed by SQLite at:

```text
outputs/history.sqlite3
```

Use WAL mode for safer concurrent reads while generation writes:

```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
```

## Initial Schema

```sql
CREATE TABLE batches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_uid TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  completed_at TEXT,
  status TEXT NOT NULL,
  prompt TEXT NOT NULL,
  negative_prompt TEXT NOT NULL,
  image_number INTEGER NOT NULL,
  total_images INTEGER,
  performance TEXT,
  quick_preview INTEGER NOT NULL DEFAULT 0,
  testing_mode INTEGER NOT NULL DEFAULT 0,
  training_mode INTEGER NOT NULL DEFAULT 0,
  config_json TEXT NOT NULL
);

CREATE TABLE images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  image_uid TEXT NOT NULL UNIQUE,
  batch_id INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  filename TEXT NOT NULL,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL,
  file_exists INTEGER NOT NULL DEFAULT 1,
  file_size INTEGER,
  width INTEGER,
  height INTEGER,
  seed INTEGER,
  image_index INTEGER,
  checkpoint TEXT,
  refiner TEXT,
  sampler TEXT,
  scheduler TEXT,
  vae TEXT,
  prompt TEXT NOT NULL,
  negative_prompt TEXT NOT NULL,
  prompt_expansion TEXT,
  metadata_json TEXT NOT NULL,
  config_json TEXT NOT NULL
);

CREATE TABLE image_loras (
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  weight REAL NOT NULL,
  role TEXT NOT NULL DEFAULT 'active',
  position INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (image_id, name, role, position)
);

CREATE TABLE tags (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  color TEXT
);

CREATE TABLE image_tags (
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
  PRIMARY KEY (image_id, tag_id)
);

CREATE TABLE batch_tags (
  batch_id INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
  tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
  PRIMARY KEY (batch_id, tag_id)
);

CREATE TABLE notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  target_type TEXT NOT NULL,
  target_id INTEGER NOT NULL,
  body TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE prompt_configs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  source_batch_id INTEGER REFERENCES batches(id) ON DELETE SET NULL,
  source_image_id INTEGER REFERENCES images(id) ON DELETE SET NULL,
  config_json TEXT NOT NULL
);
```

Indexes:

```sql
CREATE INDEX idx_images_created_at ON images(created_at DESC);
CREATE INDEX idx_images_batch_id ON images(batch_id);
CREATE INDEX idx_images_checkpoint ON images(checkpoint);
CREATE INDEX idx_images_seed ON images(seed);
CREATE INDEX idx_image_loras_name ON image_loras(name);
CREATE INDEX idx_batches_created_at ON batches(created_at DESC);
```

## Implementation Plan

1. Add `modules/history_db.py`
   - `init_db()`
   - `create_batch(config_data) -> batch_id`
   - `finish_batch(batch_id, status)`
   - `record_image(batch_id, image_path, metadata, task_context)`
   - `list_batches(filters)`
   - `list_images(filters)`
   - `get_image_config(image_id_or_path)`
   - `tag_image`, `tag_batch`, `set_note`, `mark_missing_files`

2. Add batch identity to `AsyncTask`
   - Assign `batch_uid` and `history_batch_id` when a task is queued or starts.
   - Store original generation args/config once at batch creation.
   - Include total expected image count and mode flags.

3. Record each generated image immediately after save
   - Insert image row after `log(...)` returns the final path.
   - Insert LoRA rows from the same `loras` list used for generation.
   - Store `testing_lora` with `role='testing'`.
   - Store full reloadable config JSON.

4. Query SQLite before parsing image metadata
   - Update `get_selected_generation_config(...)` in `webui.py`.
   - Lookup by image path in DB first.
   - Fall back to in-memory cache.
   - Fall back to embedded image metadata only for legacy images.

5. Add History UI
   - New tab or panel for persistent history.
   - Batch list with status, created time, checkpoint, LoRA summary, prompt preview, image count.
   - Image grid filtered by batch, checkpoint, LoRA, tag, date, seed, prompt text, favorite/missing.
   - Buttons: Load Full Config, Replace Prompt, Append Prompt, Regenerate, Favorite, Tag, Add Note, Reveal File, Mark Missing.

6. Migration/import
   - Scan `outputs/**` for images.
   - Parse embedded metadata when present.
   - Associate images into inferred batches using nearby timestamps and matching prompt/checkpoint/settings.
   - Record missing metadata images as file-only entries.
   - Never rewrite images during import.

7. Output folder re-query/reconcile
   - Add a "Re-query Outputs Folder" action in the History UI.
   - Scan the current configured output folder for supported image files.
   - Add DB records for image files found on disk that are not already in the database.
   - Remove DB records for image files that no longer exist on disk.
   - Do not modify records for images that already exist in the database and still exist on disk.
   - Report counts for added, removed, unchanged, skipped, and failed files.
   - Run inside a transaction so a failed scan does not leave a partially reconciled database.
   - Keep this separate from richer migration/import so simple folder reconciliation stays predictable.

8. Deprecate `log.html`
   - Keep optional legacy generation behind config flag for now.
   - New default should be DB-backed history.
   - Existing `History Log` link can become `History` tab navigation once UI exists.

## UI Feature Set

- Persistent history survives refresh and browser crashes.
- Batch groups for every generation run.
- Batch detail view shows all images generated together.
- Compare mode for LoRA/checkpoint testing batches.
- Filters:
  - date range
  - batch
  - checkpoint
  - LoRA
  - testing LoRA
  - seed
  - tag
  - favorite
  - missing/deleted file status
  - prompt search
- Actions:
  - reload full config
  - replace prompt
  - append prompt
  - regenerate selected image
  - regenerate entire batch
  - tag image/batch
  - favorite/reject
  - add notes
  - export selected configs
  - reveal/open file

## Current Task Checklist

- [x] Create `tasks/history` tracking folder.
- [x] Document current history weaknesses.
- [x] Draft SQLite-backed architecture.
- [x] Identify integration points.
- [x] Define first-pass schema.
- [x] Implement `modules/history_db.py`.
- [x] Add DB initialization on first history access.
- [x] Create batch records when queueing generation tasks.
- [x] Insert image records after generation.
- [x] Update config reload path to query DB first.
- [x] Add initial persistent History UI.
- [x] Add output folder re-query/reconcile action.
- [x] Add image favorite/tag/note curation actions.
- [x] Add favorite/status/tag filters for history browsing.
- [x] Add import/re-query grouping for existing outputs.
- [x] Add focused test for output re-query grouping and unchanged existing rows.
- [x] Add batch favorite/tag/note curation actions.
- [x] Add first-pass batch comparison table for checkpoint/seed/testing LoRA review.
- [x] Make comparison rows selectable so they load image curation.
- [ ] Add tests for DB schema, image recording, and metadata fallback.

## Implemented First Slice

- SQLite database at `outputs/history.sqlite3`.
- Durable batch rows created when generation tasks are queued.
- Batch status updates for queued, running, completed, and failed.
- Durable image rows inserted after each generated image is saved.
- LoRA rows inserted for active LoRAs and testing LoRA.
- Session image config reload checks SQLite before in-memory cache and embedded image metadata.
- Top-level History tab in the main UI:
  - search history batches
  - refresh batch list
  - re-query the configured outputs folder
  - select a batch
  - preview available images for the batch
  - select a history image
  - load full config
  - replace prompt
  - append prompt
  - mark favorite
  - set rating
  - set review status
  - save comma-separated tags
  - save image note
  - mark batch favorite
  - rate batches
  - set batch review status
  - save batch tags
  - save batch note
  - compare batch images by checkpoint, seed, and testing LoRA
  - click comparison rows to select the matching history image
  - filter batches/images by favorite
  - filter batches/images by review status
  - filter batches/images by tag text

## Output Folder Re-query Semantics

- Scans the configured output folder recursively for supported image files.
- Adds records for files that exist on disk but are not in the database.
- Removes image records whose files no longer exist on disk.
- Leaves records unchanged when the image path already exists in the database and still exists on disk.
- Creates inferred `imported` batches for newly discovered files using parent folder, core generation settings, and nearby timestamps.
- Keeps LoRA/testing LoRA out of the import group key so LoRA comparison runs stay in one imported batch.
- Removes empty imported batches when deleted files leave them empty.
- Reports added, removed, unchanged, imported batch, removed batch, skipped, and failed counts.

## Import/Re-query Grouping

- New files are parsed for embedded metadata when available.
- New files with the same parent folder and core generation settings are grouped together.
- Matching groups are split when timestamps are more than 15 minutes apart.
- Existing image rows are not modified during re-query.
- Plain images with no metadata are still imported as file-only history records.

## Next Implementation Step

Remaining work is polish and hardening: add focused automated tests around SQLite schema migrations, generation image recording, and metadata fallback; then add richer visual comparison tools if needed.

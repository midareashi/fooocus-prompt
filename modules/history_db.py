import json
import os
import re
import sqlite3
import threading
import time
import urllib.parse
import uuid

import modules.config
import modules.meta_parser
from PIL import Image


DB_FILENAME = 'history.sqlite3'
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
INTERNAL_OUTPUT_FOLDERS = {'history_stacks'}
_lock = threading.Lock()
_initialized = False


def _db_path():
    return os.path.abspath(os.path.join(modules.config.path_outputs, DB_FILENAME))


def _utc_now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _json_dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value, default):
    try:
        return json.loads(value)
    except Exception:
        return default


def _connect():
    os.makedirs(modules.config.path_outputs, exist_ok=True)
    conn = sqlite3.connect(_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('PRAGMA journal_mode=WAL')
    return conn


def _ensure_column(conn, table, column, definition):
    columns = [row['name'] for row in conn.execute(f'PRAGMA table_info({table})').fetchall()]
    if column not in columns:
        conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')


def init_db():
    global _initialized
    if _initialized:
        return

    with _lock:
        if _initialized:
            return
        with _connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS batches (
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
                    config_json TEXT NOT NULL,
                    favorite INTEGER NOT NULL DEFAULT 0,
                    rating INTEGER,
                    review_status TEXT NOT NULL DEFAULT '',
                    thumbnail_hidden INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_uid TEXT NOT NULL UNIQUE,
                    batch_id INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
                    path TEXT NOT NULL UNIQUE,
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
                    config_json TEXT NOT NULL,
                    favorite INTEGER NOT NULL DEFAULT 0,
                    rating INTEGER,
                    review_status TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS image_loras (
                    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    weight REAL NOT NULL,
                    role TEXT NOT NULL DEFAULT 'active',
                    position INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (image_id, name, role, position)
                );

                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    color TEXT
                );

                CREATE TABLE IF NOT EXISTS image_tags (
                    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
                    PRIMARY KEY (image_id, tag_id)
                );

                CREATE TABLE IF NOT EXISTS batch_tags (
                    batch_id INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
                    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
                    PRIMARY KEY (batch_id, tag_id)
                );

                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_type TEXT NOT NULL,
                    target_id INTEGER NOT NULL,
                    body TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prompt_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_batch_id INTEGER REFERENCES batches(id) ON DELETE SET NULL,
                    source_image_id INTEGER REFERENCES images(id) ON DELETE SET NULL,
                    config_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_images_created_at ON images(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_images_batch_id ON images(batch_id);
                CREATE INDEX IF NOT EXISTS idx_images_checkpoint ON images(checkpoint);
                CREATE INDEX IF NOT EXISTS idx_images_seed ON images(seed);
                CREATE INDEX IF NOT EXISTS idx_image_loras_name ON image_loras(name);
                CREATE INDEX IF NOT EXISTS idx_batches_created_at ON batches(created_at DESC);
                """
            )
            _ensure_column(conn, 'images', 'favorite', 'INTEGER NOT NULL DEFAULT 0')
            _ensure_column(conn, 'images', 'rating', 'INTEGER')
            _ensure_column(conn, 'images', 'review_status', "TEXT NOT NULL DEFAULT ''")
            _ensure_column(conn, 'images', 'thumbnail_hidden', 'INTEGER NOT NULL DEFAULT 0')
            _ensure_column(conn, 'batches', 'favorite', 'INTEGER NOT NULL DEFAULT 0')
            _ensure_column(conn, 'batches', 'rating', 'INTEGER')
            _ensure_column(conn, 'batches', 'review_status', "TEXT NOT NULL DEFAULT ''")
            conn.execute('CREATE INDEX IF NOT EXISTS idx_images_favorite ON images(favorite)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_images_review_status ON images(review_status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_images_thumbnail_hidden ON images(thumbnail_hidden)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_batches_favorite ON batches(favorite)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_batches_review_status ON batches(review_status)')
        _initialized = True


def task_to_config(task):
    config = {
        'prompt': getattr(task, 'prompt', ''),
        'negative_prompt': getattr(task, 'negative_prompt', ''),
        'styles': str(getattr(task, 'style_selections', []) or []),
        'wildprompts': str(getattr(task, 'wildprompt_selections', []) or []),
        'wildprompt_generate_all': bool(getattr(task, 'wildprompt_generate_all', False)),
        'wildprompt_line_selections': _json_dumps(getattr(task, 'wildprompt_line_selections', {}) or {}),
        'performance': getattr(getattr(task, 'performance_selection', None), 'value', ''),
        'steps': int(getattr(task, 'overwrite_step', 0) or 0),
        'overwrite_switch': getattr(task, 'overwrite_switch', 0),
        'guidance_scale': getattr(task, 'cfg_scale', 0),
        'sharpness': getattr(task, 'sharpness', 0),
        'adm_guidance': str((
            getattr(task, 'adm_scaler_positive', 0),
            getattr(task, 'adm_scaler_negative', 0),
            getattr(task, 'adm_scaler_end', 0)
        )),
        'refiner_swap_method': getattr(task, 'refiner_swap_method', ''),
        'adaptive_cfg': getattr(task, 'adaptive_cfg', 0),
        'clip_skip': int(getattr(task, 'clip_skip', 1) or 1),
        'base_model': getattr(task, 'base_model_name', ''),
        'refiner_model': getattr(task, 'refiner_model_name', ''),
        'refiner_switch': getattr(task, 'refiner_switch', 0),
        'sampler': getattr(task, 'sampler_name', ''),
        'scheduler': getattr(task, 'scheduler_name', ''),
        'vae': getattr(task, 'vae_name', ''),
        'seed': str(getattr(task, 'seed', 0)),
        'resolution': str(_resolution_from_task(task)),
        'quick_preview': bool(getattr(task, 'quick_preview', False)),
        'training_mode': bool(getattr(task, 'training_mode', False)),
        'testing_mode': bool(getattr(task, 'testing_mode', False)),
        'testing_loras': str(getattr(task, 'testing_loras', []) or []),
    }

    for index, (name, weight) in enumerate(getattr(task, 'loras', []) or []):
        if name != 'None':
            config[f'lora_combined_{index + 1}'] = f'{name} : {weight}'

    return config


def _resolution_from_task(task):
    try:
        width, height = str(getattr(task, 'aspect_ratios_selection', '')).replace('×', ' ').split(' ')[:2]
        return int(width), int(height)
    except Exception:
        return None


def _expected_image_count(task):
    image_number = int(getattr(task, 'image_number', 1) or 1)
    checkpoint_count = len(getattr(task, 'multi_checkpoint_model_names', []) or []) or 1
    testing_count = len(getattr(task, 'testing_loras', []) or []) or 1
    return image_number * checkpoint_count * testing_count


def create_batch_from_task(task):
    init_db()
    batch_uid = uuid.uuid4().hex
    config = task_to_config(task)
    now = _utc_now()
    with _lock, _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO batches (
                batch_uid, created_at, status, prompt, negative_prompt, image_number,
                total_images, performance, quick_preview, testing_mode, training_mode, config_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_uid,
                now,
                'queued',
                str(getattr(task, 'prompt', '') or ''),
                str(getattr(task, 'negative_prompt', '') or ''),
                int(getattr(task, 'image_number', 1) or 1),
                _expected_image_count(task),
                getattr(getattr(task, 'performance_selection', None), 'value', ''),
                1 if getattr(task, 'quick_preview', False) else 0,
                1 if getattr(task, 'testing_mode', False) else 0,
                1 if getattr(task, 'training_mode', False) else 0,
                _json_dumps(config)
            )
        )
        return cursor.lastrowid


def update_batch_status(batch_id, status):
    if batch_id is None:
        return
    init_db()
    completed_at = _utc_now() if status in ['completed', 'failed', 'stopped'] else None
    with _lock, _connect() as conn:
        if completed_at is None:
            conn.execute('UPDATE batches SET status = ? WHERE id = ?', (status, batch_id))
        else:
            conn.execute('UPDATE batches SET status = ?, completed_at = ? WHERE id = ?', (status, completed_at, batch_id))


def record_image(batch_id, image_path, metadata, task=None, loras=None, width=None, height=None, image_index=None):
    if batch_id is None or not image_path:
        return None

    init_db()
    metadata_dict = {key: value for _, key, value in metadata}
    config_json = _json_dumps(metadata_dict)
    if task is not None:
        config_data = metadata_dict.copy()
        config_data['positive'] = task.get('positive', [])
        config_data['negative'] = task.get('negative', [])
        config_json = _json_dumps(config_data)

    abs_path = os.path.abspath(image_path)
    filename = os.path.basename(abs_path)
    file_exists = os.path.exists(abs_path)
    file_size = os.path.getsize(abs_path) if file_exists else None
    now = _utc_now()

    with _lock, _connect() as conn:
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO images (
                image_uid, batch_id, path, filename, created_at, status, file_exists, file_size,
                width, height, seed, image_index, checkpoint, refiner, sampler, scheduler, vae,
                prompt, negative_prompt, prompt_expansion, metadata_json, config_json
            )
            VALUES (
                COALESCE((SELECT image_uid FROM images WHERE path = ?), ?),
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                abs_path,
                uuid.uuid4().hex,
                batch_id,
                abs_path,
                filename,
                now,
                'generated',
                1 if file_exists else 0,
                file_size,
                width,
                height,
                _safe_int(metadata_dict.get('seed')),
                image_index,
                str(metadata_dict.get('base_model', '') or ''),
                str(metadata_dict.get('refiner_model', '') or ''),
                str(metadata_dict.get('sampler', '') or ''),
                str(metadata_dict.get('scheduler', '') or ''),
                str(metadata_dict.get('vae', '') or ''),
                str(metadata_dict.get('prompt', '') or ''),
                str(metadata_dict.get('negative_prompt', '') or ''),
                str(metadata_dict.get('prompt_expansion', '') or ''),
                _json_dumps(metadata_dict),
                config_json
            )
        )
        image_id = cursor.lastrowid
        if image_id == 0:
            row = conn.execute('SELECT id FROM images WHERE path = ?', (abs_path,)).fetchone()
            image_id = row['id'] if row else None
        if image_id is not None:
            conn.execute('DELETE FROM image_loras WHERE image_id = ?', (image_id,))
            for position, (name, weight) in enumerate(loras or []):
                if name == 'None':
                    continue
                conn.execute(
                    'INSERT OR IGNORE INTO image_loras (image_id, name, weight, role, position) VALUES (?, ?, ?, ?, ?)',
                    (image_id, str(name), float(weight), 'active', position)
                )
            testing_lora = metadata_dict.get('testing_lora')
            if testing_lora:
                conn.execute(
                    'INSERT OR IGNORE INTO image_loras (image_id, name, weight, role, position) VALUES (?, ?, ?, ?, ?)',
                    (image_id, str(testing_lora), 1.0, 'testing', 0)
                )
        return image_id


def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value, default=1.0):
    try:
        return float(value)
    except Exception:
        return default


def _parse_lora_value(value):
    if not isinstance(value, str):
        return None
    parts = value.split(' : ')
    try:
        if len(parts) == 3:
            enabled, name, weight = parts
            if str(enabled).casefold() not in ['true', '1', 'yes', 'on']:
                return None
            return name, _safe_float(weight)
        if len(parts) == 2:
            name, weight = parts
            return name, _safe_float(weight)
    except Exception:
        return None
    return None


def _metadata_from_image(path):
    config = {}
    width = None
    height = None
    try:
        with Image.open(path) as image:
            width, height = image.size
            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(image)
            if parameters is not None and metadata_scheme is not None:
                parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                config = parser.to_json(parameters)
    except Exception:
        config = {}

    if not isinstance(config, dict):
        config = {}

    return config, width, height


def _is_missing_config_value(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ''
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def _merge_missing_config(primary, fallback):
    merged = primary.copy() if isinstance(primary, dict) else {}
    if not isinstance(fallback, dict):
        return merged
    for key, value in fallback.items():
        if _is_missing_config_value(value):
            continue
        if key not in merged or _is_missing_config_value(merged.get(key)):
            merged[key] = value
    return merged


def _parse_log_html_configs(log_path):
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return {}

    configs = {}
    blocks = re.findall(
        r'<div[^>]+class=["\']image-container["\'][\s\S]*?(?=<div[^>]+class=["\']image-container["\']|<!--fooocus-log-split-->|</body>|</html>)',
        content,
        flags=re.IGNORECASE
    )
    for block in blocks:
        filename = None
        for pattern in [
            r'<a[^>]+href=["\']([^"\']+)["\']',
            r'<img[^>]+src=["\']([^"\']+)["\']',
        ]:
            match = re.search(pattern, block, flags=re.IGNORECASE)
            if match:
                filename = os.path.basename(urllib.parse.unquote(match.group(1)))
                break
        if not filename:
            continue

        encoded = None
        button_match = re.search(
            r'to_clipboard\(["\']([^"\']+)["\']\)',
            block,
            flags=re.IGNORECASE
        )
        if button_match:
            encoded = button_match.group(1)
        if encoded is not None:
            try:
                parsed = json.loads(urllib.parse.unquote(encoded))
                if isinstance(parsed, dict):
                    configs[filename] = parsed
                    continue
            except Exception:
                pass

        table_config = {}
        for label, value in re.findall(
            r"<tr>\s*<td[^>]*class=['\"]label['\"][^>]*>([\s\S]*?)</td>\s*"
            r"<td[^>]*class=['\"]value['\"][^>]*>([\s\S]*?)</td>\s*</tr>",
            block,
            flags=re.IGNORECASE
        ):
            key = _log_label_to_config_key(label)
            if key:
                table_config[key] = _clean_log_html_value(value)
        if table_config:
            configs[filename] = table_config
    return configs


def _clean_log_html_value(value):
    text = re.sub(r'<\s*/?br\s*/?\s*>', '\n', str(value), flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    return urllib.parse.unquote(text).replace(' </br> ', '\n').strip()


def _log_label_to_config_key(label):
    normalized = _clean_log_html_value(label).casefold()
    mapping = {
        'prompt': 'prompt',
        'negative prompt': 'negative_prompt',
        'seed': 'seed',
        'base model': 'base_model',
        'refiner model': 'refiner_model',
        'sampler': 'sampler',
        'scheduler': 'scheduler',
        'vae': 'vae',
        'performance': 'performance',
        'steps': 'steps',
        'guidance scale': 'guidance_scale',
        'sharpness': 'sharpness',
        'adm guidance': 'adm_guidance',
        'styles': 'styles',
        'metadata scheme': 'metadata_scheme',
        'prompt expansion': 'prompt_expansion',
    }
    if normalized.startswith('lora '):
        suffix = re.sub(r'[^0-9]', '', normalized)
        if suffix:
            return f'lora_combined_{suffix}'
    return mapping.get(normalized)


def _log_config_for_image(path, cache):
    directory = os.path.abspath(os.path.dirname(path))
    if directory not in cache:
        log_path = os.path.join(directory, 'log.html')
        cache[directory] = _parse_log_html_configs(log_path) if os.path.exists(log_path) else {}
    return cache[directory].get(os.path.basename(path), {})


def _config_has_core_metadata(config):
    config = config if isinstance(config, dict) else {}
    for key in ['prompt', 'negative_prompt', 'base_model', 'seed', 'sampler', 'scheduler']:
        if not _is_missing_config_value(config.get(key)):
            return True
    return False


def _update_image_metadata_row(conn, image_id, config):
    config = config.copy() if isinstance(config, dict) else {}
    metadata_json = _json_dumps(config)
    conn.execute(
        """
        UPDATE images
        SET seed = ?, checkpoint = ?, refiner = ?, sampler = ?, scheduler = ?, vae = ?,
            prompt = ?, negative_prompt = ?, prompt_expansion = ?,
            metadata_json = ?, config_json = ?
        WHERE id = ?
        """,
        (
            _safe_int(config.get('seed')),
            str(config.get('base_model', '') or ''),
            str(config.get('refiner_model', '') or ''),
            str(config.get('sampler', '') or ''),
            str(config.get('scheduler', '') or ''),
            str(config.get('vae', '') or ''),
            str(config.get('prompt', '') or ''),
            str(config.get('negative_prompt', '') or ''),
            str(config.get('prompt_expansion', '') or ''),
            metadata_json,
            metadata_json,
            image_id
        )
    )
    conn.execute('DELETE FROM image_loras WHERE image_id = ?', (image_id,))
    for key, value in config.items():
        if not str(key).startswith('lora_combined_'):
            continue
        parsed_lora = _parse_lora_value(value)
        if parsed_lora is None:
            continue
        name, weight = parsed_lora
        if name == 'None':
            continue
        position = _safe_int(str(key).replace('lora_combined_', '')) or 0
        conn.execute(
            'INSERT OR IGNORE INTO image_loras (image_id, name, weight, role, position) VALUES (?, ?, ?, ?, ?)',
            (image_id, str(name), float(weight), 'active', position)
        )


def _insert_image_row(conn, batch_id, image_path, config, width=None, height=None, image_index=None):
    abs_path = os.path.abspath(image_path)
    filename = os.path.basename(abs_path)
    file_exists = os.path.exists(abs_path)
    file_size = os.path.getsize(abs_path) if file_exists else None
    now = _utc_now()
    config = config.copy() if isinstance(config, dict) else {}
    metadata_json = _json_dumps(config)
    cursor = conn.execute(
        """
        INSERT INTO images (
            image_uid, batch_id, path, filename, created_at, status, file_exists, file_size,
            width, height, seed, image_index, checkpoint, refiner, sampler, scheduler, vae,
            prompt, negative_prompt, prompt_expansion, metadata_json, config_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uuid.uuid4().hex,
            batch_id,
            abs_path,
            filename,
            now,
            'imported',
            1 if file_exists else 0,
            file_size,
            width,
            height,
            _safe_int(config.get('seed')),
            image_index,
            str(config.get('base_model', '') or ''),
            str(config.get('refiner_model', '') or ''),
            str(config.get('sampler', '') or ''),
            str(config.get('scheduler', '') or ''),
            str(config.get('vae', '') or ''),
            str(config.get('prompt', '') or ''),
            str(config.get('negative_prompt', '') or ''),
            str(config.get('prompt_expansion', '') or ''),
            metadata_json,
            metadata_json
        )
    )
    image_id = cursor.lastrowid

    for key, value in config.items():
        if not str(key).startswith('lora_combined_'):
            continue
        parsed_lora = _parse_lora_value(value)
        if parsed_lora is None:
            continue
        name, weight = parsed_lora
        if name == 'None':
            continue
        position = _safe_int(str(key).replace('lora_combined_', '')) or 0
        conn.execute(
            'INSERT OR IGNORE INTO image_loras (image_id, name, weight, role, position) VALUES (?, ?, ?, ?, ?)',
            (image_id, str(name), float(weight), 'active', position)
        )

    testing_lora = config.get('testing_lora')
    if testing_lora:
        conn.execute(
            'INSERT OR IGNORE INTO image_loras (image_id, name, weight, role, position) VALUES (?, ?, ?, ?, ?)',
            (image_id, str(testing_lora), 1.0, 'testing', 0)
        )

    return image_id


def _import_config_key(path, config):
    config = config if isinstance(config, dict) else {}
    parent = os.path.abspath(os.path.dirname(path))
    key_fields = [
        'prompt',
        'negative_prompt',
        'base_model',
        'refiner_model',
        'sampler',
        'scheduler',
        'vae',
        'performance',
        'resolution',
        'guidance_scale',
        'sharpness',
        'adm_guidance',
        'styles',
    ]
    return tuple([parent] + [str(config.get(key, '') or '') for key in key_fields])


def _group_import_images(parsed_images, max_gap_seconds=15 * 60):
    groups = []
    open_groups = {}

    for item in sorted(parsed_images, key=lambda row: (row['key'], row['mtime'], row['path'])):
        key = item['key']
        current = open_groups.get(key)
        if current is None or item['mtime'] - current['last_mtime'] > max_gap_seconds:
            current = {
                'key': key,
                'items': [],
                'first_mtime': item['mtime'],
                'last_mtime': item['mtime'],
            }
            groups.append(current)
            open_groups[key] = current
        current['items'].append(item)
        current['last_mtime'] = item['mtime']

    return groups


def _create_import_batch(conn, output_folder, group):
    items = group['items']
    first = items[0] if len(items) > 0 else {}
    config = first.get('config') if isinstance(first.get('config'), dict) else {}
    created_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(group.get('first_mtime') or time.time()))
    completed_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(group.get('last_mtime') or time.time()))
    prompt = str(config.get('prompt', '') or '')
    negative_prompt = str(config.get('negative_prompt', '') or '')
    batch_config = config.copy()
    batch_config.update({
        'source': 'output_folder_requery',
        'output_folder': output_folder,
        'imported_image_count': len(items),
        'import_group_key': _json_dumps(group.get('key', ())),
    })
    if prompt == '':
        prompt = 'Imported from output folder re-query'

    cursor = conn.execute(
        """
        INSERT INTO batches (
            batch_uid, created_at, completed_at, status, prompt, negative_prompt,
            image_number, total_images, performance, quick_preview, testing_mode,
            training_mode, config_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uuid.uuid4().hex,
            created_at,
            completed_at,
            'imported',
            prompt,
            negative_prompt,
            len(items),
            len(items),
            str(config.get('performance', '') or ''),
            0,
            1 if str(config.get('testing_mode', '') or '').casefold() in ['true', '1', 'yes', 'on'] else 0,
            1 if str(config.get('training_mode', '') or '').casefold() in ['true', '1', 'yes', 'on'] else 0,
            _json_dumps(batch_config)
        )
    )
    return cursor.lastrowid


def get_config_by_path(image_path):
    if not image_path:
        return {}
    init_db()
    abs_path = os.path.abspath(image_path)
    with _connect() as conn:
        row = conn.execute('SELECT config_json FROM images WHERE path = ?', (abs_path,)).fetchone()
    if row is None:
        return {}
    return _json_loads(row['config_json'], {})


def get_image_id_by_path(image_path):
    if not image_path:
        return None
    init_db()
    abs_path = os.path.abspath(image_path)
    with _connect() as conn:
        row = conn.execute('SELECT id FROM images WHERE path = ?', (abs_path,)).fetchone()
    return row['id'] if row else None


def _normalize_tag_names(tags):
    if isinstance(tags, str):
        tags = tags.split(',')
    if not isinstance(tags, list):
        return []
    normalized = []
    seen = set()
    for tag in tags:
        tag = str(tag or '').strip()
        if tag == '':
            continue
        tag_key = tag.casefold()
        if tag_key in seen:
            continue
        normalized.append(tag[:80])
        seen.add(tag_key)
    return normalized


def _get_or_create_tag(conn, name):
    conn.execute('INSERT OR IGNORE INTO tags (name) VALUES (?)', (name,))
    row = conn.execute('SELECT id FROM tags WHERE name = ?', (name,)).fetchone()
    return row['id'] if row else None


def get_image_curation(image_id):
    init_db()
    try:
        image_id = int(image_id)
    except Exception:
        return {}
    with _connect() as conn:
        image = conn.execute(
            'SELECT id, favorite, rating, review_status FROM images WHERE id = ?',
            (image_id,)
        ).fetchone()
        if image is None:
            return {}
        tag_rows = conn.execute(
            """
            SELECT t.name
            FROM tags t
            JOIN image_tags it ON it.tag_id = t.id
            WHERE it.image_id = ?
            ORDER BY t.name
            """,
            (image_id,)
        ).fetchall()
        note = conn.execute(
            """
            SELECT body
            FROM notes
            WHERE target_type = 'image' AND target_id = ?
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (image_id,)
        ).fetchone()
    return {
        'favorite': bool(image['favorite']),
        'rating': image['rating'] if image['rating'] is not None else 0,
        'review_status': image['review_status'] or '',
        'tags': ', '.join([row['name'] for row in tag_rows]),
        'note': note['body'] if note else ''
    }


def update_image_curation(image_id, favorite=False, rating=0, review_status='', tags='', note=''):
    init_db()
    try:
        image_id = int(image_id)
    except Exception:
        return False
    rating = _safe_int(rating)
    if rating is not None:
        rating = max(0, min(5, rating))
    review_status = str(review_status or '').strip()[:40]
    tag_names = _normalize_tag_names(tags)
    note = str(note or '').strip()
    now = _utc_now()
    with _lock, _connect() as conn:
        row = conn.execute('SELECT id FROM images WHERE id = ?', (image_id,)).fetchone()
        if row is None:
            return False
        conn.execute(
            'UPDATE images SET favorite = ?, rating = ?, review_status = ? WHERE id = ?',
            (1 if favorite else 0, rating, review_status, image_id)
        )
        conn.execute('DELETE FROM image_tags WHERE image_id = ?', (image_id,))
        for tag_name in tag_names:
            tag_id = _get_or_create_tag(conn, tag_name)
            if tag_id is not None:
                conn.execute(
                    'INSERT OR IGNORE INTO image_tags (image_id, tag_id) VALUES (?, ?)',
                    (image_id, tag_id)
                )
        conn.execute("DELETE FROM notes WHERE target_type = 'image' AND target_id = ?", (image_id,))
        if note != '':
            conn.execute(
                """
                INSERT INTO notes (target_type, target_id, body, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                ('image', image_id, note, now, now)
            )
    return True


def set_image_thumbnail_hidden(image_id, hidden=True):
    init_db()
    try:
        image_id = int(image_id)
    except Exception:
        return False
    with _lock, _connect() as conn:
        cursor = conn.execute(
            'UPDATE images SET thumbnail_hidden = ? WHERE id = ?',
            (1 if hidden else 0, image_id)
        )
    return cursor.rowcount is not None and cursor.rowcount > 0


def get_batch_curation(batch_id):
    init_db()
    try:
        batch_id = int(batch_id)
    except Exception:
        return {}
    with _connect() as conn:
        batch = conn.execute(
            'SELECT id, favorite, rating, review_status FROM batches WHERE id = ?',
            (batch_id,)
        ).fetchone()
        if batch is None:
            return {}
        tag_rows = conn.execute(
            """
            SELECT t.name
            FROM tags t
            JOIN batch_tags bt ON bt.tag_id = t.id
            WHERE bt.batch_id = ?
            ORDER BY t.name
            """,
            (batch_id,)
        ).fetchall()
        note = conn.execute(
            """
            SELECT body
            FROM notes
            WHERE target_type = 'batch' AND target_id = ?
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (batch_id,)
        ).fetchone()
    return {
        'favorite': bool(batch['favorite']),
        'rating': batch['rating'] if batch['rating'] is not None else 0,
        'review_status': batch['review_status'] or '',
        'tags': ', '.join([row['name'] for row in tag_rows]),
        'note': note['body'] if note else ''
    }


def update_batch_curation(batch_id, favorite=False, rating=0, review_status='', tags='', note=''):
    init_db()
    try:
        batch_id = int(batch_id)
    except Exception:
        return False
    rating = _safe_int(rating)
    if rating is not None:
        rating = max(0, min(5, rating))
    review_status = str(review_status or '').strip()[:40]
    tag_names = _normalize_tag_names(tags)
    note = str(note or '').strip()
    now = _utc_now()
    with _lock, _connect() as conn:
        row = conn.execute('SELECT id FROM batches WHERE id = ?', (batch_id,)).fetchone()
        if row is None:
            return False
        conn.execute(
            'UPDATE batches SET favorite = ?, rating = ?, review_status = ? WHERE id = ?',
            (1 if favorite else 0, rating, review_status, batch_id)
        )
        conn.execute('DELETE FROM batch_tags WHERE batch_id = ?', (batch_id,))
        for tag_name in tag_names:
            tag_id = _get_or_create_tag(conn, tag_name)
            if tag_id is not None:
                conn.execute(
                    'INSERT OR IGNORE INTO batch_tags (batch_id, tag_id) VALUES (?, ?)',
                    (batch_id, tag_id)
                )
        conn.execute("DELETE FROM notes WHERE target_type = 'batch' AND target_id = ?", (batch_id,))
        if note != '':
            conn.execute(
                """
                INSERT INTO notes (target_type, target_id, body, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                ('batch', batch_id, note, now, now)
            )
    return True


def reconcile_outputs_folder(output_folder=None):
    init_db()
    output_folder = os.path.abspath(output_folder or modules.config.path_outputs)

    def is_inside_output_folder(path):
        try:
            return os.path.commonpath([output_folder, os.path.abspath(path)]) == output_folder
        except Exception:
            return False

    disk_paths = set()
    skipped = 0
    failed = 0

    for root, dirnames, filenames in os.walk(output_folder):
        dirnames[:] = [dirname for dirname in dirnames if dirname not in INTERNAL_OUTPUT_FOLDERS]
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                skipped += 1
                continue
            path = os.path.abspath(os.path.join(root, filename))
            if _is_internal_output_path(path, output_folder):
                skipped += 1
                continue
            disk_paths.add(path)

    with _connect() as conn:
        rows = conn.execute('SELECT id, path, config_json FROM images').fetchall()
    existing_paths = {
        os.path.abspath(row['path']): {
            'id': row['id'],
            'config': _json_loads(row['config_json'], {})
        }
        for row in rows
        if is_inside_output_folder(row['path'])
    }
    missing_paths = sorted([
        path for path in existing_paths
        if not os.path.exists(path) or _is_internal_output_path(path, output_folder)
    ])
    new_paths = sorted([path for path in disk_paths if path not in existing_paths])
    unchanged = len(disk_paths) - len(new_paths)
    log_config_cache = {}

    parsed_new_images = []
    for path in new_paths:
        try:
            config, width, height = _metadata_from_image(path)
            config = _merge_missing_config(config, _log_config_for_image(path, log_config_cache))
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                mtime = time.time()
            parsed_new_images.append({
                'path': path,
                'config': config,
                'width': width,
                'height': height,
                'mtime': mtime,
                'key': _import_config_key(path, config),
            })
        except Exception:
            failed += 1

    added = 0
    updated = 0
    removed = 0
    imported_batches = 0
    removed_batches = 0
    import_groups = _group_import_images(parsed_new_images)
    with _lock, _connect() as conn:
        conn.execute('BEGIN')
        try:
            if len(missing_paths) > 0:
                conn.executemany('DELETE FROM images WHERE path = ?', [(path,) for path in missing_paths])
                removed = len(missing_paths)
                cursor = conn.execute(
                    """
                    DELETE FROM batches
                    WHERE status = 'imported'
                      AND NOT EXISTS (SELECT 1 FROM images WHERE images.batch_id = batches.id)
                    """
                )
                removed_batches = cursor.rowcount if cursor.rowcount is not None else 0

            for path, existing in sorted(existing_paths.items()):
                if path in missing_paths:
                    continue
                current_config = existing.get('config', {})
                log_config = _log_config_for_image(path, log_config_cache)
                merged_config = _merge_missing_config(current_config, log_config)
                if merged_config != current_config and (
                    not _config_has_core_metadata(current_config) or _config_has_core_metadata(log_config)
                ):
                    _update_image_metadata_row(conn, existing['id'], merged_config)
                    updated += 1

            for group in import_groups:
                batch_id = _create_import_batch(conn, output_folder, group)
                imported_batches += 1
                for image_index, item in enumerate(sorted(group['items'], key=lambda row: (row['mtime'], row['path']))):
                    _insert_image_row(
                        conn,
                        batch_id,
                        item['path'],
                        item['config'],
                        item['width'],
                        item['height'],
                        image_index
                    )
                    added += 1

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return {
        'output_folder': output_folder,
        'added': added,
        'updated': updated,
        'removed': removed,
        'unchanged': unchanged,
        'skipped': skipped,
        'failed': failed,
        'imported_batches': imported_batches,
        'removed_batches': removed_batches
    }


def list_batches(limit=100, search='', favorite_only=False, review_status='', tag=''):
    init_db()
    search = str(search or '').strip()
    review_status = str(review_status or '').strip()
    tag = str(tag or '').strip()
    params = []
    where_clauses = []
    if search:
        where_clauses.append('(b.prompt LIKE ? OR b.negative_prompt LIKE ? OR b.performance LIKE ?)')
        like = f'%{search}%'
        params += [like, like, like]
    if favorite_only:
        where_clauses.append(
            '(b.favorite = 1 OR EXISTS (SELECT 1 FROM images fi WHERE fi.batch_id = b.id AND fi.favorite = 1))'
        )
    if review_status:
        where_clauses.append(
            '(b.review_status = ? OR EXISTS (SELECT 1 FROM images si WHERE si.batch_id = b.id AND si.review_status = ?))'
        )
        params += [review_status, review_status]
    if tag:
        where_clauses.append(
            """
            (
            EXISTS (
                SELECT 1
                FROM batch_tags btt
                JOIN tags btag ON btag.id = btt.tag_id
                WHERE btt.batch_id = b.id AND btag.name LIKE ?
            )
            OR EXISTS (
                SELECT 1
                FROM images ti
                JOIN image_tags tit ON tit.image_id = ti.id
                JOIN tags tt ON tt.id = tit.tag_id
                WHERE ti.batch_id = b.id AND tt.name LIKE ?
            )
            )
            """
        )
        params += [f'%{tag}%', f'%{tag}%']
    where = f"WHERE {' AND '.join(where_clauses)}" if len(where_clauses) > 0 else ''
    params.append(int(limit))
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT
                b.id, b.created_at, b.status, b.total_images, b.performance,
                b.testing_mode, b.prompt, b.favorite, b.rating, b.review_status,
                (
                    SELECT GROUP_CONCAT(t.name, ', ')
                    FROM tags t
                    JOIN batch_tags bt ON bt.tag_id = t.id
                    WHERE bt.batch_id = b.id
                ) AS tags,
                COUNT(i.id) AS generated_images
            FROM batches b
            LEFT JOIN images i ON i.batch_id = b.id
            {where}
            GROUP BY b.id
            ORDER BY b.created_at DESC
            LIMIT ?
            """,
            params
        ).fetchall()
    return [dict(row) for row in rows]


def list_batch_images(batch_id, favorite_only=False, review_status='', tag='', show_preview_images=False,
                      thumbnail_visibility='visible'):
    init_db()
    try:
        batch_id = int(batch_id)
    except Exception:
        return []
    review_status = str(review_status or '').strip()
    tag = str(tag or '').strip()
    where_clauses = ['batch_id = ?']
    params = [batch_id]
    if favorite_only:
        where_clauses.append('favorite = 1')
    if review_status:
        where_clauses.append('review_status = ?')
        params.append(review_status)
    if tag:
        where_clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM image_tags fit
                JOIN tags ft ON ft.id = fit.tag_id
                WHERE fit.image_id = images.id AND ft.name LIKE ?
            )
            """
        )
        params.append(f'%{tag}%')
    if not show_preview_images:
        where_clauses.append(_preview_image_filter_clause())
    visibility_clause = _thumbnail_visibility_clause(thumbnail_visibility)
    if visibility_clause:
        where_clauses.append(visibility_clause)
    where = ' AND '.join(where_clauses)
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, path, filename, created_at, status, file_exists, seed, image_index,
                   checkpoint, sampler, scheduler, prompt, favorite, rating, review_status, thumbnail_hidden,
                   (
                       SELECT GROUP_CONCAT(t.name, ', ')
                       FROM tags t
                       JOIN image_tags it ON it.tag_id = t.id
                       WHERE it.image_id = images.id
                   ) AS tags
            FROM images
            WHERE {where}
            ORDER BY COALESCE(image_index, id), id
            """,
            params
        ).fetchall()
    images = []
    for row in rows:
        item = dict(row)
        exists = os.path.exists(item['path'])
        if exists != bool(item['file_exists']):
            mark_image_file_exists(item['id'], exists)
            item['file_exists'] = 1 if exists else 0
        images.append(item)
    return images


def list_output_days():
    init_db()
    output_folder = os.path.abspath(modules.config.path_outputs)
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT path
            FROM images
            WHERE file_exists = 1
            ORDER BY created_at DESC
            """
        ).fetchall()
    days = []
    seen = set()
    for row in rows:
        if _is_internal_output_path(row['path'], output_folder):
            continue
        day = _output_day_from_path(row['path'], output_folder)
        if day == '' or day in seen:
            continue
        seen.add(day)
        days.append(day)
    return sorted(days, key=_output_day_sort_key, reverse=True)


def list_output_day_counts():
    init_db()
    output_folder = os.path.abspath(modules.config.path_outputs)
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT path
            FROM images
            WHERE file_exists = 1
            ORDER BY created_at DESC
            """
        ).fetchall()
    counts = {}
    for row in rows:
        if _is_internal_output_path(row['path'], output_folder):
            continue
        day = _output_day_from_path(row['path'], output_folder)
        if day == '':
            continue
        counts[day] = counts.get(day, 0) + 1
    return counts


def _output_day_sort_key(day):
    try:
        return time.strptime(str(day), '%Y-%m-%d')
    except Exception:
        return time.strptime('1900-01-01', '%Y-%m-%d')


def _output_day_from_path(path, output_folder):
    path = os.path.abspath(path)
    try:
        relative = os.path.relpath(path, output_folder)
    except Exception:
        relative = path
    first_part = relative.split(os.sep, 1)[0]
    if first_part == '' or first_part == os.curdir or first_part == os.pardir:
        first_part = os.path.basename(os.path.dirname(path))
    return first_part


def _is_internal_output_path(path, output_folder=None):
    output_folder = os.path.abspath(output_folder or modules.config.path_outputs)
    try:
        relative = os.path.relpath(os.path.abspath(path), output_folder)
    except Exception:
        return False
    first_part = relative.split(os.sep, 1)[0]
    return first_part in INTERNAL_OUTPUT_FOLDERS


def list_filter_values():
    init_db()
    with _connect() as conn:
        checkpoint_rows = conn.execute(
            """
            SELECT DISTINCT checkpoint
            FROM images
            WHERE checkpoint IS NOT NULL AND checkpoint != ''
            ORDER BY checkpoint
            """
        ).fetchall()
        lora_rows = conn.execute(
            """
            SELECT DISTINCT name
            FROM image_loras
            WHERE name IS NOT NULL AND name != '' AND name != 'None'
            ORDER BY name
            """
        ).fetchall()
    return {
        'checkpoints': [row['checkpoint'] for row in checkpoint_rows],
        'loras': [row['name'] for row in lora_rows]
    }


def _normalize_filter_list(values):
    if isinstance(values, str):
        values = [values] if values.strip() != '' else []
    if not isinstance(values, list):
        return []
    normalized = []
    seen = set()
    for value in values:
        value = str(value or '').strip()
        if value == '' or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def _thumbnail_visibility_clause(thumbnail_visibility):
    mode = str(thumbnail_visibility or 'visible').strip().casefold()
    if mode == 'hidden':
        return 'images.thumbnail_hidden = 1'
    if mode == 'all':
        return ''
    return 'images.thumbnail_hidden = 0'


def _image_filter_where(search='', favorite_only=False, review_status='', tag='', days=None, batch_id=None,
                        checkpoints=None, loras=None, show_preview_images=False,
                        thumbnail_visibility='visible'):
    search = str(search or '').strip()
    review_status = str(review_status or '').strip()
    tag = str(tag or '').strip()
    days = [str(day) for day in (days or []) if str(day or '').strip() != '']
    checkpoints = _normalize_filter_list(checkpoints)
    loras = _normalize_filter_list(loras)
    params = []
    where_clauses = []
    if batch_id is not None:
        where_clauses.append('images.batch_id = ?')
        params.append(int(batch_id))
    output_folder = os.path.abspath(modules.config.path_outputs)
    for internal_folder in INTERNAL_OUTPUT_FOLDERS:
        internal_path = os.path.abspath(os.path.join(output_folder, internal_folder))
        where_clauses.append('NOT (images.path = ? OR images.path LIKE ?)')
        params += [internal_path, internal_path + os.sep + '%']
    if search:
        where_clauses.append('images.prompt LIKE ?')
        like = f'%{search}%'
        params.append(like)
    if favorite_only:
        where_clauses.append('images.favorite = 1')
    if review_status:
        where_clauses.append('images.review_status = ?')
        params.append(review_status)
    if tag:
        where_clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM image_tags fit
                JOIN tags ft ON ft.id = fit.tag_id
                WHERE fit.image_id = images.id AND ft.name LIKE ?
            )
            """
        )
        params.append(f'%{tag}%')
    if len(days) > 0:
        day_clauses = []
        if '__all__' not in days:
            for day in days:
                day_path = os.path.abspath(os.path.join(output_folder, day))
                day_clauses.append('images.path = ? OR images.path LIKE ?')
                params += [day_path, day_path + os.sep + '%']
            if len(day_clauses) > 0:
                where_clauses.append('(' + ' OR '.join(day_clauses) + ')')
    if len(checkpoints) > 0:
        where_clauses.append('images.checkpoint IN (' + ', '.join(['?'] * len(checkpoints)) + ')')
        params += checkpoints
    if len(loras) > 0:
        where_clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM image_loras lora_filter
                WHERE lora_filter.image_id = images.id
                  AND lora_filter.name IN (
            """ + ', '.join(['?'] * len(loras)) + """
                  )
            )
            """
        )
        params += loras
    if not show_preview_images:
        where_clauses.append(_preview_image_filter_clause())
    visibility_clause = _thumbnail_visibility_clause(thumbnail_visibility)
    if visibility_clause:
        where_clauses.append(visibility_clause)
    where = ' AND '.join(where_clauses) if len(where_clauses) > 0 else '1 = 1'
    return where, params


def _preview_image_filter_clause():
    return """
    images.config_json NOT LIKE '%"steps": 10%'
    AND images.config_json NOT LIKE '%"steps":10%'
    AND images.config_json NOT LIKE '%"quick_preview": true%'
    AND images.config_json NOT LIKE '%"quick_preview":true%'
    AND NOT EXISTS (
        SELECT 1
        FROM batches preview_batch
        WHERE preview_batch.id = images.batch_id
          AND preview_batch.quick_preview = 1
    )
    """


def list_images(search='', favorite_only=False, review_status='', tag='', days=None,
                checkpoints=None, loras=None, show_preview_images=False,
                thumbnail_visibility='visible', limit=500):
    init_db()
    where, params = _image_filter_where(search, favorite_only, review_status, tag, days,
                                        checkpoints=checkpoints, loras=loras,
                                        show_preview_images=show_preview_images,
                                        thumbnail_visibility=thumbnail_visibility)
    params.append(int(limit))
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, path, filename, created_at, status, file_exists, seed, image_index,
                   checkpoint, sampler, scheduler, prompt, favorite, rating, review_status, thumbnail_hidden,
                   (
                       SELECT GROUP_CONCAT(t.name, ', ')
                       FROM tags t
                       JOIN image_tags it ON it.tag_id = t.id
                       WHERE it.image_id = images.id
                   ) AS tags
            FROM images
            WHERE {where}
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            params
        ).fetchall()
    images = []
    for row in rows:
        item = dict(row)
        exists = os.path.exists(item['path'])
        if exists != bool(item['file_exists']):
            mark_image_file_exists(item['id'], exists)
            item['file_exists'] = 1 if exists else 0
        images.append(item)
    return images


def list_seed_stacks(search='', favorite_only=False, review_status='', tag='', days=None,
                     checkpoints=None, loras=None, show_preview_images=False,
                     thumbnail_visibility='visible', limit=200):
    init_db()
    where, params = _image_filter_where(search, favorite_only, review_status, tag, days,
                                        checkpoints=checkpoints, loras=loras,
                                        show_preview_images=show_preview_images,
                                        thumbnail_visibility=thumbnail_visibility)
    params.append(int(limit))
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT
                MIN(images.id) AS id,
                images.seed,
                images.prompt,
                (
                    SELECT pi.path
                    FROM images pi
                    WHERE pi.seed = images.seed
                      AND pi.prompt = images.prompt
                      AND pi.file_exists = 1
                    ORDER BY pi.created_at DESC, pi.id DESC
                    LIMIT 1
                ) AS preview_path,
                COUNT(images.id) AS image_count,
                COUNT(DISTINCT images.checkpoint) AS checkpoint_count,
                (
                    SELECT COUNT(DISTINCT il.name)
                    FROM images li
                    JOIN image_loras il ON il.image_id = li.id
                    WHERE li.seed = images.seed
                      AND li.prompt = images.prompt
                      AND il.name IS NOT NULL
                      AND il.name != ''
                ) AS lora_count
            FROM images
            WHERE {where}
              AND images.seed IS NOT NULL
              AND images.prompt IS NOT NULL
              AND images.prompt != ''
            GROUP BY images.seed, images.prompt
            HAVING COUNT(images.id) > 1
            ORDER BY MAX(images.created_at) DESC, MAX(images.id) DESC
            LIMIT ?
            """,
            params
        ).fetchall()
    return [dict(row) for row in rows]


def list_seed_stack_images(seed, prompt, search='', favorite_only=False, review_status='', tag='', days=None,
                           checkpoints=None, loras=None, show_preview_images=False,
                           thumbnail_visibility='visible', limit=100):
    init_db()
    seed = _safe_int(seed)
    prompt = str(prompt or '')
    if seed is None or prompt == '':
        return []
    where, params = _image_filter_where(search, favorite_only, review_status, tag, days,
                                        checkpoints=checkpoints, loras=loras,
                                        show_preview_images=show_preview_images,
                                        thumbnail_visibility=thumbnail_visibility)
    where = f'({where}) AND images.seed = ? AND images.prompt = ?'
    params += [seed, prompt, int(limit)]
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, path, filename, created_at, status, file_exists, seed, image_index,
                   checkpoint, sampler, scheduler, prompt, favorite, rating, review_status, thumbnail_hidden,
                   (
                       SELECT GROUP_CONCAT(t.name, ', ')
                       FROM tags t
                       JOIN image_tags it ON it.tag_id = t.id
                       WHERE it.image_id = images.id
                   ) AS tags,
                   (
                       SELECT GROUP_CONCAT(il.name, ', ')
                       FROM image_loras il
                       WHERE il.image_id = images.id
                   ) AS loras
            FROM images
            WHERE {where}
            ORDER BY checkpoint, loras, id
            LIMIT ?
            """,
            params
        ).fetchall()
    images = []
    for row in rows:
        item = dict(row)
        exists = os.path.exists(item['path'])
        if exists != bool(item['file_exists']):
            mark_image_file_exists(item['id'], exists)
            item['file_exists'] = 1 if exists else 0
        images.append(item)
    return images


def get_seed_stack_key(stack_id):
    init_db()
    try:
        stack_id = int(stack_id)
    except Exception:
        return None, ''
    with _connect() as conn:
        row = conn.execute('SELECT seed, prompt FROM images WHERE id = ?', (stack_id,)).fetchone()
    if row is None:
        return None, ''
    return row['seed'], row['prompt'] or ''


def get_image_summary(image_id):
    init_db()
    try:
        image_id = int(image_id)
    except Exception:
        return {}
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, path, filename, created_at, status, file_exists, seed, image_index,
                   checkpoint, sampler, scheduler, prompt, favorite, rating, review_status,
                   thumbnail_hidden,
                   (
                       SELECT GROUP_CONCAT(t.name, ', ')
                       FROM tags t
                       JOIN image_tags it ON it.tag_id = t.id
                       WHERE it.image_id = images.id
                   ) AS tags
            FROM images
            WHERE id = ?
            """,
            (image_id,)
        ).fetchone()
    if row is None:
        return {}
    item = dict(row)
    exists = os.path.exists(item['path'])
    if exists != bool(item['file_exists']):
        mark_image_file_exists(item['id'], exists)
        item['file_exists'] = 1 if exists else 0
    return item


def get_image_loras(image_id):
    init_db()
    try:
        image_id = int(image_id)
    except Exception:
        return []
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT name, weight, role, position
            FROM image_loras
            WHERE image_id = ?
            ORDER BY
                CASE role WHEN 'active' THEN 0 WHEN 'testing' THEN 1 ELSE 2 END,
                position,
                name
            """,
            (image_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def list_batch_comparison_rows(batch_id):
    init_db()
    try:
        batch_id = int(batch_id)
    except Exception:
        return []
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                i.id, i.path, i.filename, i.file_exists, i.seed, i.image_index,
                i.checkpoint, i.favorite, i.rating, i.review_status,
                (
                    SELECT il.name
                    FROM image_loras il
                    WHERE il.image_id = i.id AND il.role = 'testing'
                    ORDER BY il.position, il.name
                    LIMIT 1
                ) AS testing_lora,
                (
                    SELECT GROUP_CONCAT(il.name || ' (' || il.weight || ')', ', ')
                    FROM image_loras il
                    WHERE il.image_id = i.id AND il.role = 'active'
                ) AS active_loras,
                (
                    SELECT GROUP_CONCAT(t.name, ', ')
                    FROM tags t
                    JOIN image_tags it ON it.tag_id = t.id
                    WHERE it.image_id = i.id
                ) AS tags
            FROM images i
            WHERE i.batch_id = ?
            ORDER BY i.checkpoint, COALESCE(i.seed, i.image_index, i.id),
                     COALESCE(testing_lora, ''), COALESCE(i.image_index, i.id), i.id
            """,
            (batch_id,)
        ).fetchall()
    comparison_rows = []
    for row in rows:
        item = dict(row)
        exists = os.path.exists(item['path'])
        if exists != bool(item['file_exists']):
            mark_image_file_exists(item['id'], exists)
            item['file_exists'] = 1 if exists else 0
        comparison_rows.append(item)
    return comparison_rows


def mark_image_file_exists(image_id, exists):
    with _lock, _connect() as conn:
        conn.execute('UPDATE images SET file_exists = ? WHERE id = ?', (1 if exists else 0, image_id))


def get_image_path(image_id):
    init_db()
    try:
        image_id = int(image_id)
    except Exception:
        return None
    with _connect() as conn:
        row = conn.execute('SELECT path FROM images WHERE id = ?', (image_id,)).fetchone()
    return row['path'] if row else None


def delete_image(image_id, delete_file=True):
    init_db()
    try:
        image_id = int(image_id)
    except Exception:
        return False, None
    with _lock, _connect() as conn:
        row = conn.execute('SELECT path FROM images WHERE id = ?', (image_id,)).fetchone()
        if row is None:
            return False, None
        path = row['path']
        if delete_file and path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                return False, path
        conn.execute('DELETE FROM images WHERE id = ?', (image_id,))
    return True, path


def get_config_by_image_id(image_id):
    init_db()
    try:
        image_id = int(image_id)
    except Exception:
        return {}
    with _connect() as conn:
        row = conn.execute('SELECT config_json FROM images WHERE id = ?', (image_id,)).fetchone()
    if row is None:
        return {}
    return _json_loads(row['config_json'], {})

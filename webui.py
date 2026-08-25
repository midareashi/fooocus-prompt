import gradio as gr
import random
import os
import json
import html as html_lib
import ast
import time
import re
import threading
import hashlib
from datetime import date, datetime, timedelta
import shared
import modules.config
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.wildprompt_sorter as wildprompt_sorter
import modules.sdxl_styles
import modules.meta_parser
import modules.prompt_config
import modules.lora_notes
import modules.history_db
import args_manager
import copy
import launch
from extras.inpaint_mask import SAMOptions

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
from modules.util import is_json


people_dir = os.path.abspath(os.path.join('input', 'people'))
legacy_people_dir = os.path.abspath('input')
history_debug_enabled = bool(
    getattr(args_manager.args, 'history_debug', False)
    or str(os.getenv('FOOOCUS_HISTORY_DEBUG') or '').strip().lower() in ['1', 'true', 'yes', 'on']
)
print(
    '[HistoryDebug] status=',
    'enabled' if history_debug_enabled else 'disabled',
    'source=',
    'arg' if getattr(args_manager.args, 'history_debug', False) else (
        'env' if str(os.getenv('FOOOCUS_HISTORY_DEBUG') or '').strip().lower() in ['1', 'true', 'yes', 'on'] else 'off'
    ),
    'tip=use --history-debug or FOOOCUS_HISTORY_DEBUG=1',
    flush=True
)


def history_debug(*parts):
    if not history_debug_enabled:
        return
    print('[HistoryDebug]', *parts, flush=True)


def sanitize_person_name(name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(name or '').strip())
    name = re.sub(r'\s+', ' ', name).strip(' .')
    return name[:80]


def get_uploaded_file_path(file):
    if file is None:
        return None
    if isinstance(file, str):
        return file
    if isinstance(file, dict):
        return file.get('name') or file.get('path')
    return getattr(file, 'name', None)


def flatten_person_likeness_files(files):
    if files is None:
        return []
    if isinstance(files, str) and files.strip().startswith('['):
        try:
            files = json.loads(files)
        except Exception:
            files = [files]
    if isinstance(files, (str, dict)) or hasattr(files, 'name'):
        files = [files]
    return [file for file in files if get_uploaded_file_path(file) is not None]


def parse_person_likeness_paths(paths_json):
    try:
        paths = json.loads(paths_json or '[]')
    except Exception:
        paths = []
    return [
        path for path in paths
        if isinstance(path, str) and os.path.exists(path)
    ]


def encode_person_likeness_paths(paths):
    deduped = []
    seen = set()
    for path in paths:
        path = os.path.abspath(path)
        key = os.path.normcase(path)
        if os.path.exists(path) and key not in seen:
            deduped.append(path)
            seen.add(key)
    return json.dumps(deduped)


def preview_person_likeness_paths(paths_json):
    return parse_person_likeness_paths(paths_json)


def append_person_likeness_files(files, paths_json):
    paths = parse_person_likeness_paths(paths_json)
    paths += [get_uploaded_file_path(file) for file in flatten_person_likeness_files(files)]
    encoded = encode_person_likeness_paths(paths)
    return encoded, preview_person_likeness_paths(encoded), gr.update(value=None)


def clamp_float(value, default, minimum, maximum):
    try:
        value = float(value)
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


PERSON_LIKENESS_PRESETS = {
    'Baseline': (1.00, 1.00, 0.30),
    'ID+': (1.15, 1.00, 0.25),
    'Face+': (1.00, 1.15, 0.25),
    'Early Lock': (1.05, 1.05, 0.15),
    'Strong Match': (1.20, 1.15, 0.15),
    'Aggressive Match': (1.30, 1.25, 0.10),
    'Flexible': (0.90, 0.90, 0.35),
    'Dataset Candidate': (1.15, 1.10, 0.20),
}


def apply_person_likeness_preset(name):
    return PERSON_LIKENESS_PRESETS.get(name, PERSON_LIKENESS_PRESETS['Baseline'])


def clear_person_likeness_settings():
    strength, face_weight, face_start = apply_person_likeness_preset('Baseline')
    return True, 'person', strength, face_weight, face_start, 'Baseline', 'Person likeness settings reset to Baseline.'


def list_saved_people():
    saved_people = set()
    for base_dir in [people_dir, legacy_people_dir]:
        if not os.path.exists(base_dir):
            continue
        for name in os.listdir(base_dir):
            person_dir = os.path.join(base_dir, name)
            if os.path.isdir(person_dir) and os.path.exists(os.path.join(person_dir, 'person.json')):
                saved_people.add(name)
    return sorted(saved_people, key=lambda x: x.lower())


def resolve_saved_person_dir(person_name):
    for base_dir in [people_dir, legacy_people_dir]:
        person_dir = os.path.abspath(os.path.join(base_dir, person_name))
        if os.path.exists(os.path.join(person_dir, 'person.json')) and os.path.commonpath([base_dir, person_dir]) == base_dir:
            return person_dir, base_dir
    return os.path.abspath(os.path.join(people_dir, person_name)), people_dir


def save_person_likeness(name, enabled, subject, strength, face_weight, face_start, files):
    from PIL import Image

    person_name = sanitize_person_name(name)
    if person_name == '':
        return gr.update(), 'Enter a name before saving.', gr.update(), gr.update()

    valid_files = flatten_person_likeness_files(files)
    if len(valid_files) == 0:
        return gr.update(), 'Add at least one photo before saving.', gr.update(), gr.update()

    os.makedirs(people_dir, exist_ok=True)
    person_dir = os.path.abspath(os.path.join(people_dir, person_name))
    if os.path.commonpath([people_dir, person_dir]) != people_dir:
        return gr.update(), 'Invalid person name.', gr.update(), gr.update()
    if os.path.exists(person_dir) and os.listdir(person_dir) and not os.path.exists(os.path.join(person_dir, 'person.json')):
        return gr.update(), f'Cannot save: input folder already exists and is not a saved person: {person_name}', gr.update(), gr.update()

    os.makedirs(person_dir, exist_ok=True)

    saved_count = 0
    image_files = []
    save_stamp = time.strftime('%Y%m%d_%H%M%S')
    for file in valid_files:
        image_path = get_uploaded_file_path(file)
        source_path = os.path.abspath(image_path)
        if os.path.exists(source_path) and os.path.commonpath([person_dir, source_path]) == person_dir:
            image_files.append(os.path.basename(source_path))
            saved_count += 1
            continue

        try:
            filename = f'{save_stamp}_{saved_count + 1:02d}.png'
            Image.open(image_path).convert('RGB').save(os.path.join(person_dir, filename))
            image_files.append(filename)
            saved_count += 1
        except Exception:
            pass

    if saved_count == 0:
        return gr.update(), 'No valid image files were found.', gr.update(), gr.update()

    person_config = {
        'name': person_name,
        'enabled': bool(enabled),
        'subject': subject if subject in flags.person_likeness_classes else 'person',
        'identity_strength': clamp_float(strength, 1.0, 0.0, modules.config.default_person_likeness_strength_max),
        'face_weight': clamp_float(face_weight, modules.config.default_person_likeness_face_weight, 0.0,
                                   modules.config.default_person_likeness_face_weight_max),
        'face_weight_start': clamp_float(face_start, modules.config.default_person_likeness_face_start, 0.0, 1.0),
        'image_count': saved_count,
        'image_files': image_files
    }
    with open(os.path.join(person_dir, 'person.json'), 'w', encoding='utf-8') as f:
        json.dump(person_config, f, indent=2)

    image_file_set = set(image_files)
    for filename in os.listdir(person_dir):
        path = os.path.join(person_dir, filename)
        if os.path.isfile(path) and filename not in image_file_set and os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg', '.webp']:
            try:
                os.remove(path)
            except Exception:
                pass

    saved_paths = encode_person_likeness_paths([os.path.join(person_dir, filename) for filename in image_files])
    choices = list_saved_people()
    return gr.update(choices=choices, value=person_name), f'Saved {saved_count} photo(s) for {person_name}.', saved_paths, preview_person_likeness_paths(saved_paths)


def load_person_likeness(name):
    person_name = sanitize_person_name(name)
    if person_name == '':
        return True, 'person', 1.0, modules.config.default_person_likeness_face_weight, \
            modules.config.default_person_likeness_face_start, '[]', 'Choose a saved person to load.'

    person_dir, base_dir = resolve_saved_person_dir(person_name)
    if not os.path.exists(person_dir) or os.path.commonpath([base_dir, person_dir]) != base_dir:
        return True, 'person', 1.0, modules.config.default_person_likeness_face_weight, \
            modules.config.default_person_likeness_face_start, '[]', f'Could not find saved person: {person_name}'

    metadata = {}
    metadata_path = os.path.join(person_dir, 'person.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    subject = metadata.get('subject', 'person')
    if subject not in flags.person_likeness_classes:
        subject = 'person'
    enabled = bool(metadata.get('enabled', True))
    strength = clamp_float(metadata.get('identity_strength', metadata.get('strength', 1.0)), 1.0, 0.0,
                           modules.config.default_person_likeness_strength_max)
    face_weight = clamp_float(metadata.get('face_weight', modules.config.default_person_likeness_face_weight),
                              modules.config.default_person_likeness_face_weight, 0.0,
                              modules.config.default_person_likeness_face_weight_max)
    face_start = clamp_float(metadata.get('face_weight_start', metadata.get('face_start',
                                                                            modules.config.default_person_likeness_face_start)),
                             modules.config.default_person_likeness_face_start, 0.0, 1.0)

    image_files = metadata.get('image_files')
    if isinstance(image_files, list):
        image_paths = [
            os.path.join(person_dir, filename)
            for filename in image_files
            if isinstance(filename, str)
            and os.path.exists(os.path.join(person_dir, filename))
            and os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg', '.webp']
        ]
    else:
        image_paths = sorted([
            os.path.join(person_dir, filename)
            for filename in os.listdir(person_dir)
            if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg', '.webp']
        ])
    return enabled, subject, strength, face_weight, face_start, encode_person_likeness_paths(image_paths), \
        f'Loaded {len(image_paths)} photo(s) for {person_name}.'


def build_prompt_config(prompt, negative_prompt, style_selections, wildprompt_selections, wildprompt_generate_all,
                        wildprompt_line_selections,
                        performance_selection, overwrite_step,
                        overwrite_switch, aspect_ratios_selection, overwrite_width, overwrite_height,
                        guidance_scale, sharpness, adm_scaler_positive, adm_scaler_negative, adm_scaler_end,
                        refiner_swap_method, adaptive_cfg, clip_skip, base_model, refiner_model, refiner_switch,
                        sampler_name, scheduler_name, vae_name, seed_random, image_seed, inpaint_engine,
                        inpaint_mode, person_likeness_enabled, person_likeness_class, person_likeness_strength,
                        person_likeness_face_weight, person_likeness_face_start, person_likeness_paths,
                        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, *lora_values):
    lora_prompt_values = list(lora_values[-modules.config.default_max_lora_number:])
    lora_values = lora_values[:-modules.config.default_max_lora_number]
    resolution_numbers = re.findall(r'\d+', str(aspect_ratios_selection))
    if len(resolution_numbers) >= 2:
        resolution = (int(resolution_numbers[0]), int(resolution_numbers[1]))
    elif int(overwrite_width) > 0 and int(overwrite_height) > 0:
        resolution = (int(overwrite_width), int(overwrite_height))
    else:
        resolution = None

    generate_all_files = modules.sdxl_styles.normalize_wildprompt_generate_all_files(
        wildprompt_selections,
        wildprompt_generate_all,
    )
    config_data = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'styles': str(style_selections or []),
        'wildprompts': str(wildprompt_selections or []),
        'wildprompt_generate_all': len(generate_all_files) > 0,
        'wildprompt_generate_all_files': str(generate_all_files),
        'wildprompt_line_selections': wildprompt_line_selections if isinstance(wildprompt_line_selections, str) else '{}',
        'performance': performance_selection,
        'steps': int(overwrite_step),
        'overwrite_switch': overwrite_switch,
        'guidance_scale': guidance_scale,
        'sharpness': sharpness,
        'adm_guidance': str((adm_scaler_positive, adm_scaler_negative, adm_scaler_end)),
        'refiner_swap_method': refiner_swap_method,
        'adaptive_cfg': adaptive_cfg,
        'clip_skip': int(clip_skip),
        'base_model': base_model,
        'refiner_model': refiner_model,
        'refiner_switch': refiner_switch,
        'sampler': sampler_name,
        'scheduler': scheduler_name,
        'vae': vae_name,
        'inpaint_engine_version': inpaint_engine,
        'inpaint_method': inpaint_mode,
        'person_likeness_enabled': bool(person_likeness_enabled),
        'person_likeness_class': person_likeness_class,
        'person_likeness_strength': person_likeness_strength,
        'person_likeness_face_weight': person_likeness_face_weight,
        'person_likeness_face_start': person_likeness_face_start,
        'person_likeness_paths': person_likeness_paths if isinstance(person_likeness_paths, str) else '[]',
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'version': 'Fooocus v' + fooocus_version.version
    }

    if resolution is not None:
        config_data['resolution'] = str(resolution)

    if not seed_random:
        config_data['seed'] = str(image_seed)

    if freeu_enabled:
        config_data['freeu'] = str((freeu_b1, freeu_b2, freeu_s1, freeu_s2))

    for index in range(0, len(lora_values), 3):
        enabled, filename, weight = lora_values[index:index + 3]
        if filename != 'None':
            config_data[f'lora_combined_{index // 3 + 1}'] = f'{enabled} : {filename} : {weight}'

    for index, lora_prompt in enumerate(lora_prompt_values):
        lora_prompt = str(lora_prompt or '').strip()
        if lora_prompt != '':
            config_data[f'lora_prompt_{index + 1}'] = lora_prompt

    return config_data


def append_lora_note_to_prompt(prompt, lora_note):
    prompt = str(prompt or '').strip()
    lora_note = str(lora_note or '').strip()
    if lora_note == '':
        return prompt
    if prompt == '':
        return lora_note
    return f'{prompt}, {lora_note}'


def get_task(*args):
    args = list(args)
    args.pop(0)

    return worker.AsyncTask(args=args)


def set_quick_preview_mode(enabled):
    return bool(enabled)


def make_queue_panel_html():
    snapshot = worker.get_queue_snapshot()
    active = snapshot.get('active')
    pending = snapshot.get('pending') or []

    if active is None and len(pending) == 0:
        return '<div class="queue-panel queue-panel-empty">Queue is empty.</div>'

    rows = ['<div class="queue-panel">']
    rows.append('<div class="queue-panel-header"><span>Queue</span><span>Images</span><span>Steps</span><span>Action</span></div>')

    def row_html(task, status):
        badges = []
        if task.get('quick_preview'):
            badges.append('<span class="queue-badge">Preview</span>')
        prompt = html_lib.escape(task.get('prompt', '(empty prompt)'))
        performance = html_lib.escape(str(task.get('performance') or ''))
        steps = int(task.get("steps", 0) or 0)
        total_steps = int(task.get("total_steps", 0) or 0)
        badge_html = ''.join(badges)
        if status == 'active':
            action = (
                '<div class="queue-action-group">'
                '<button type="button" class="queue-skip-button">Skip</button>'
                '<button type="button" class="queue-stop-button">Stop</button>'
                '</div>'
            )
        else:
            action = (
                f'<button type="button" class="queue-remove-button" '
                f'data-queue-id="{int(task.get("id", 0))}">Remove from Queue</button>'
            )
        return (
            f'<div class="queue-row queue-row-{status}">'
            f'<div><strong>{status.title()}</strong><span>{prompt}</span>{badge_html}</div>'
            f'<div>{int(task.get("images", 0) or 0)}</div>'
            f'<div>{total_steps}<small>{steps} per image</small><small>{performance}</small></div>'
            f'<div>{action}</div>'
            f'</div>'
        )

    if active is not None:
        rows.append(row_html(active, 'active'))
    for task in pending:
        rows.append(row_html(task, 'pending'))

    rows.append('</div>')
    return ''.join(rows)


def get_generation_tracking_task(task):
    active_task = worker.get_current_task()
    if active_task is not None and not getattr(active_task, 'completed', False):
        return active_task
    return task


def get_active_generation_task(fallback_task=None):
    active_task = worker.get_current_task()
    if active_task is not None and not getattr(active_task, 'completed', False):
        return active_task
    if fallback_task is not None and not getattr(fallback_task, 'completed', False):
        return fallback_task
    return None


def enqueue_generate_task(*args):
    task = get_task(*args)
    should_monitor = False
    tracking_task = task

    if len(task.args) > 0:
        pending_count = worker.append_async_task(task)
        should_monitor = worker.begin_queue_monitor()
        tracking_task = get_generation_tracking_task(task)
        print(f'[Queue] Added generation task. Pending tasks: {pending_count}')

    return tracking_task, should_monitor, \
        gr.update(visible=False, interactive=False), \
        gr.update(visible=False, interactive=False), \
        gr.update(visible=True, interactive=True), \
        gr.update(visible=True, interactive=True), \
        gr.update(), \
        gr.update(value=make_queue_panel_html()), \
        True


def remove_queued_task(queue_id):
    removed = worker.remove_pending_task(queue_id)
    message = 'Removed queued item.' if removed else 'Queued item was already running or missing.'
    return gr.update(value=make_queue_panel_html()), message


def prune_missing_gallery_paths(gallery_items):
    pruned = []
    changed = False
    for item in list(gallery_items or []):
        if isinstance(item, str) and not os.path.exists(item):
            changed = True
            continue
        pruned.append(item)
    return pruned, changed


def monitor_generate_queue(should_monitor, session_history):
    session_history, session_history_pruned = prune_missing_gallery_paths(session_history)

    if not should_monitor:
        yield gr.update(), gr.update(), gr.update(), \
            gr.update(value=session_history) if session_history_pruned else gr.update(), session_history, \
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return

    def get_quick_preview_indices():
        indices = []
        for index, image_item in enumerate(session_history):
            if isinstance(image_item, str):
                config_data = worker.get_generated_image_config(image_item)
                if bool(config_data.get('quick_preview', False)):
                    indices.append(index)
        return json.dumps(indices)

    observed_task = None
    execution_start_time = None

    def get_latest_display_image(image_items):
        image_items = list(image_items or [])
        for image_item in reversed(image_items):
            if isinstance(image_item, str):
                if os.path.exists(image_item):
                    return image_item
                continue
            return image_item
        return None

    def append_task_results_to_session_history(task, image_items=None):
        if image_items is None:
            image_items = getattr(task, 'results', []) or []
        changed = False
        for image_item in list(image_items):
            if isinstance(image_item, str):
                if not os.path.exists(image_item):
                    continue
                if image_item not in session_history:
                    session_history.append(image_item)
                    changed = True
            elif image_item not in session_history:
                session_history.append(image_item)
                changed = True
        return changed

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Waiting for task to start ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False, value=None), \
        gr.update(visible=True), \
        gr.update(), \
        gr.update(visible=True, interactive=True), \
        gr.update(visible=False, interactive=False), \
        gr.update(visible=False, interactive=False), \
        gr.update(value=make_queue_panel_html()), \
        gr.update(value=get_quick_preview_indices()), \
        True

    try:
        while True:
            worker.heartbeat_queue_monitor()
            active_task = worker.get_current_task()
            if active_task is not None and observed_task is not active_task:
                observed_task = active_task
                execution_start_time = time.perf_counter()

            if observed_task is None:
                pending_count = worker.get_pending_task_count()
                if pending_count == 0:
                    break

                yield gr.update(
                    visible=True,
                    value=modules.html.make_progress_html(1, f'Waiting for queued task ... ({pending_count} pending)')
                ), gr.update(), gr.update(), gr.update(), \
                    gr.update(), \
                    gr.update(visible=True, interactive=True), \
                    gr.update(visible=False, interactive=False), \
                    gr.update(visible=False, interactive=False), \
                    gr.update(value=make_queue_panel_html()), \
                    gr.update(value=get_quick_preview_indices()), \
                    True
                time.sleep(0.1)
                continue

            if worker.get_task_yield_count(observed_task) == 0:
                time.sleep(0.01)
                continue

            event = worker.get_latest_display_yield(preferred_task=observed_task, same_task_only=True)
            if event is None:
                time.sleep(0.01)
                continue

            observed_task, flag, product = event
            if flag == 'preview':
                percentage, title, image = product
                session_history_changed = append_task_results_to_session_history(observed_task)
                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(), \
                    gr.update(visible=True, value=session_history) if session_history_changed else gr.update(visible=True), \
                    session_history, \
                    gr.update(visible=True, interactive=True), \
                    gr.update(visible=False, interactive=False), \
                    gr.update(visible=False, interactive=False), \
                    gr.update(value=make_queue_panel_html()), \
                    gr.update(value=get_quick_preview_indices()), \
                    True
            if flag == 'results':
                for image_item in product:
                    if isinstance(image_item, str):
                        if not os.path.exists(image_item):
                            continue
                        if image_item not in session_history:
                            session_history.append(image_item)
                    else:
                        session_history.append(image_item)

                latest_image = get_latest_display_image(product)
                yield gr.update(visible=True), \
                    gr.update(visible=True, value=latest_image) if latest_image is not None else gr.update(visible=True), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value=session_history), \
                    session_history, \
                    gr.update(visible=True, interactive=True), \
                    gr.update(visible=False, interactive=False), \
                    gr.update(visible=False, interactive=False), \
                    gr.update(value=make_queue_panel_html()), \
                    gr.update(value=get_quick_preview_indices()), \
                    True
            if flag == 'finish':
                if not args_manager.args.disable_enhance_output_sorting:
                    product = sort_enhance_images(product, observed_task)

                for image_item in product:
                    if isinstance(image_item, str):
                        if not os.path.exists(image_item):
                            continue
                        if image_item not in session_history:
                            session_history.append(image_item)
                    else:
                        session_history.append(image_item)

                latest_image = get_latest_display_image(product)
                yield gr.update(visible=False), \
                    gr.update(visible=True, value=latest_image) if latest_image is not None else gr.update(visible=True), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value=session_history), \
                    session_history, \
                    gr.update(visible=True, interactive=True), \
                    gr.update(visible=False, interactive=False), \
                    gr.update(visible=False, interactive=False), \
                    gr.update(value=make_queue_panel_html()), \
                    gr.update(value=get_quick_preview_indices()), \
                    True

                if execution_start_time is not None:
                    execution_time = time.perf_counter() - execution_start_time
                    print(f'Total time: {execution_time:.2f} seconds')

                observed_task = None
                execution_start_time = None
    finally:
        worker.end_queue_monitor()

    yield gr.update(), gr.update(), gr.update(), gr.update(), \
        gr.update(), \
        gr.update(visible=True, interactive=True), \
        gr.update(visible=False, interactive=False), \
        gr.update(visible=False, interactive=False), \
        gr.update(value=make_queue_panel_html()), \
        gr.update(value=get_quick_preview_indices()), \
        False


def poll_generate_queue(task, is_generating, session_history):
    session_history, session_history_pruned = prune_missing_gallery_paths(session_history)

    def get_quick_preview_indices():
        indices = []
        for index, image_item in enumerate(session_history):
            if isinstance(image_item, str):
                config_data = worker.get_generated_image_config(image_item)
                if bool(config_data.get('quick_preview', False)):
                    indices.append(index)
        return json.dumps(indices)

    def get_latest_display_image(image_items):
        image_items = list(image_items or [])
        for image_item in reversed(image_items):
            if isinstance(image_item, str):
                if os.path.exists(image_item):
                    return image_item
                continue
            return image_item
        return None

    def append_task_results_to_session_history(task, image_items=None):
        if image_items is None:
            image_items = getattr(task, 'results', []) or []
        changed = False
        for image_item in list(image_items):
            if isinstance(image_item, str):
                if not os.path.exists(image_item):
                    continue
                if image_item not in session_history:
                    session_history.append(image_item)
                    changed = True
            elif image_item not in session_history:
                session_history.append(image_item)
                changed = True
        return changed

    def idle_updates():
        return gr.update(), gr.update(), gr.update(), \
            gr.update(value=session_history) if session_history_pruned else gr.update(), session_history, \
            gr.update(visible=True, interactive=True), \
            gr.update(visible=False, interactive=False), \
            gr.update(visible=False, interactive=False), \
            gr.update(value=make_queue_panel_html()), \
            gr.update(value=get_quick_preview_indices()), \
            False

    active_task = worker.get_current_task()
    pending_count = worker.get_pending_task_count()
    task_yield_count = worker.get_task_yield_count(task)
    if active_task is not None and active_task is not task and not getattr(active_task, 'completed', False) \
            and (task_yield_count == 0 or getattr(task, 'completed', False)):
        task = active_task
        task_yield_count = worker.get_task_yield_count(task)

    active_running = active_task is not None and not getattr(active_task, 'completed', False)
    task_is_active = active_task is task
    task_is_pending = (
        task is not None and hasattr(task, 'yields') and
        not task_is_active and pending_count > 0 and not getattr(task, 'completed', False)
    )

    if not is_generating and not active_running and not task_is_pending and pending_count == 0 \
            and task_yield_count == 0:
        return idle_updates()

    worker.heartbeat_queue_monitor()

    event = worker.get_latest_display_yield(preferred_task=task, same_task_only=True) \
        if task_yield_count > 0 else None
    if event is None:
        active_task = worker.get_current_task()
        pending_count = worker.get_pending_task_count()
        active_running = active_task is not None and not getattr(active_task, 'completed', False)
        if not active_running and pending_count == 0:
            if task is not None and getattr(task, 'completed', False):
                final_product = list(getattr(task, 'results', []) or [])
                if not args_manager.args.disable_enhance_output_sorting:
                    final_product = sort_enhance_images(final_product, task)
                append_task_results_to_session_history(task, final_product)
                latest_image = get_latest_display_image(final_product)
                return gr.update(visible=False), \
                    gr.update(visible=True, value=latest_image) if latest_image is not None else gr.update(visible=True), \
                    gr.update(), \
                    gr.update(visible=True, value=session_history) if final_product else gr.update(visible=True), \
                    session_history, \
                    gr.update(visible=True, interactive=True), \
                    gr.update(visible=False, interactive=False), \
                    gr.update(visible=False, interactive=False), \
                    gr.update(value=make_queue_panel_html()), \
                    gr.update(value=get_quick_preview_indices()), \
                    False
            worker.end_queue_monitor()
            return idle_updates()
        if not active_running and pending_count > 0:
            return gr.update(
                visible=True,
                value=modules.html.make_progress_html(1, f'Waiting for queued task ... ({pending_count} pending)')
            ), gr.update(), gr.update(), gr.update(), session_history, \
                gr.update(visible=True, interactive=True), \
                gr.update(visible=False, interactive=False), \
                gr.update(visible=False, interactive=False), \
                gr.update(value=make_queue_panel_html()), \
                gr.update(value=get_quick_preview_indices()), \
                True
        running_task_id = getattr(active_task, 'queue_id', 0)
        status_html = modules.html.make_progress_html(
            1,
            f'Generation running (task {running_task_id})...'
            if running_task_id
            else 'Generation running...'
        )
        latest_image = get_latest_display_image(session_history)
        return status_html, \
            gr.update(visible=True, value=latest_image) if latest_image is not None else gr.update(), \
            gr.update(), \
            gr.update(visible=True, value=session_history), \
            session_history, \
            gr.update(visible=True, interactive=True), \
            gr.update(visible=False, interactive=False), \
            gr.update(visible=False, interactive=False), \
            gr.update(value=make_queue_panel_html()), \
            gr.update(value=get_quick_preview_indices()), \
            True

    task, flag, product = event
    if flag == 'preview':
        percentage, title, image = product
        session_history_changed = append_task_results_to_session_history(task)
        return gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
            gr.update(visible=True, value=image) if image is not None else gr.update(), \
            gr.update(), \
            gr.update(visible=True, value=session_history) if session_history_changed else gr.update(visible=True), \
            session_history, \
            gr.update(visible=True, interactive=True), \
            gr.update(visible=False, interactive=False), \
            gr.update(visible=False, interactive=False), \
            gr.update(value=make_queue_panel_html()), \
            gr.update(value=get_quick_preview_indices()), \
            True

    if flag in ['results', 'finish']:
        if flag == 'finish' and not args_manager.args.disable_enhance_output_sorting:
            product = sort_enhance_images(product, task)

        if flag == 'results':
            image_items = product
        elif flag == 'finish':
            image_items = list(product)
            if not image_items:
                image_items = list(getattr(task, 'results', []) or [])
                if not args_manager.args.disable_enhance_output_sorting:
                    image_items = sort_enhance_images(image_items, task)

        append_task_results_to_session_history(task, image_items)

        latest_image = get_latest_display_image(image_items)
        active_task = worker.get_current_task()
        active_running = active_task is not None and not getattr(active_task, 'completed', False)
        has_more_work = active_running or worker.get_pending_task_count() > 0
        is_finished = flag == 'finish' and not has_more_work
        if is_finished:
            worker.end_queue_monitor()
        return gr.update(visible=not is_finished), \
            gr.update(visible=True, value=latest_image) if latest_image is not None else gr.update(visible=True), \
            gr.update(visible=False), \
            gr.update(visible=True, value=session_history), \
            session_history, \
            gr.update(visible=True, interactive=True), \
            gr.update(visible=False, interactive=False), \
            gr.update(visible=False, interactive=False), \
            gr.update(value=make_queue_panel_html()), \
            gr.update(value=get_quick_preview_indices()), \
            has_more_work

    return gr.update(), gr.update(), gr.update(), gr.update(), session_history, \
        gr.update(visible=True, interactive=True), \
        gr.update(visible=False, interactive=False), \
        gr.update(visible=False, interactive=False), \
        gr.update(value=make_queue_panel_html()), \
        gr.update(value=get_quick_preview_indices()), \
        True


def reconnect_generate_queue(session_history):
    active_task = worker.get_current_task()
    if active_task is None:
        latest_event = worker.get_latest_display_yield()
        if latest_event is not None and latest_event[1] in ['results', 'finish']:
            task = latest_event[0]
            print(f'[Queue] Reconnected to completed generation task {getattr(task, "queue_id", 0)}.')
            return task, False, \
                gr.update(visible=True, interactive=True), \
                gr.update(visible=False, interactive=False), \
                gr.update(visible=False, interactive=False), \
                gr.update(visible=False, interactive=False), \
                gr.update(value=make_queue_panel_html()), \
                True

        return worker.AsyncTask(args=[]), False, \
            gr.update(visible=True, interactive=True), \
            gr.update(visible=False, interactive=False), \
            gr.update(visible=False, interactive=False), \
            gr.update(visible=False, interactive=False), \
            gr.update(value=make_queue_panel_html()), \
            False

    should_monitor = worker.begin_queue_monitor()
    print(f'[Queue] Reconnected to active generation task {getattr(active_task, "queue_id", 0)}.')
    return active_task, should_monitor, \
        gr.update(visible=True, interactive=True), \
        gr.update(visible=False, interactive=False), \
        gr.update(visible=False, interactive=False), \
        gr.update(visible=True, interactive=True), \
        gr.update(value=make_queue_panel_html()), \
        True


def sort_enhance_images(images, task):
    if not task.should_enhance or len(images) <= task.images_to_enhance_count:
        return images

    sorted_images = []
    walk_index = task.images_to_enhance_count

    for index, enhanced_img in enumerate(images[:task.images_to_enhance_count]):
        sorted_images.append(enhanced_img)
        if index not in task.enhance_stats:
            continue
        target_index = walk_index + task.enhance_stats[index]
        if walk_index < len(images) and target_index <= len(images):
            sorted_images += images[walk_index:target_index]
        walk_index += task.enhance_stats[index]

    return sorted_images


def inpaint_mode_change(mode, inpaint_engine_version):
    assert mode in modules.flags.inpaint_options

    # inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
    # inpaint_disable_initial_latent, inpaint_engine,
    # inpaint_strength, inpaint_respective_field

    if mode == modules.flags.inpaint_option_detail:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=True, samples=modules.config.example_inpaint_prompts),
            False, 'None', 0.5, 0.0
        ]

    if inpaint_engine_version == 'empty':
        inpaint_engine_version = modules.config.default_inpaint_engine_version

    if mode == modules.flags.inpaint_option_modify:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
            True, inpaint_engine_version, 1.0, 0.0
        ]

    return [
        gr.update(visible=False, value=''), gr.update(visible=True),
        gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
        False, inpaint_engine_version, 1.0, 0.618
    ]


reload_javascript()

title = f'Fooocus {fooocus_version.version}'

if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset

shared.gradio_root = gr.Blocks(title=title).queue()

with shared.gradio_root:
    with gr.Tabs(elem_id='generation_mode_tabs', selected='image_generation_tab'):
        with gr.Tab(label='History', id='history_tab'):
            history_visible_image_ids = gr.State([])
            history_selected_image_ids = gr.State([])
            history_selection_mode = gr.Textbox(value='single', elem_id='history_selection_mode',
                                                visible=False)
            history_day_selection_mode = gr.Textbox(value='single', elem_id='history_day_selection_mode',
                                                    visible=False)
            history_selected_image_ids_json = gr.Textbox(value='[]', elem_id='history_selected_image_ids_json',
                                                         visible=False)
            history_select_thumbnail_image_id = gr.Textbox(value='', elem_id='history_select_thumbnail_image_id',
                                                           visible=False)
            history_select_thumbnail_button = gr.Button(value='Select History Thumbnail',
                                                        elem_id='history_select_thumbnail_button',
                                                        visible=False)
            history_selected_days = gr.State([])
            history_remove_selected_image_id = gr.Textbox(value='', elem_id='history_remove_selected_image_id',
                                                          visible=False)
            history_remove_selected_image_button = gr.Button(value='Remove Selected History Image',
                                                             elem_id='history_remove_selected_image_button',
                                                             visible=False)
            history_delete_selected_image_id = gr.Textbox(value='', elem_id='history_delete_selected_image_id',
                                                          visible=False)
            history_delete_selected_image_button = gr.Button(value='Delete Selected History Image',
                                                             elem_id='history_delete_selected_image_button',
                                                             visible=False)
            history_apply_selected_image_id = gr.Textbox(value='', elem_id='history_apply_selected_image_id',
                                                         visible=False)
            history_apply_selected_image_button = gr.Button(value='Apply Selected History Image Config',
                                                            elem_id='history_apply_selected_image_button',
                                                            visible=False)
            history_quality_selected_image_id = gr.Textbox(value='', elem_id='history_quality_selected_image_id',
                                                           visible=False)
            history_quality_selected_image_button = gr.Button(value='Generate History Preview at Quality',
                                                              elem_id='history_quality_selected_image_button',
                                                              visible=False)
            history_toggle_favorite_image_id = gr.Textbox(value='', elem_id='history_toggle_favorite_image_id',
                                                          visible=False)
            history_toggle_favorite_button = gr.Button(value='Toggle History Favorite',
                                                       elem_id='history_toggle_favorite_button',
                                                       visible=False)
            history_hide_thumbnail_image_id = gr.Textbox(value='', elem_id='history_hide_thumbnail_image_id',
                                                         visible=False)
            history_hide_thumbnail_button = gr.Button(value='Hide History Thumbnail',
                                                      elem_id='history_hide_thumbnail_button',
                                                      visible=False)
            history_config_action_image_id = gr.Textbox(value='', elem_id='history_config_action_image_id',
                                                        visible=False)
            history_image_selection = gr.Dropdown(label='Images', choices=[], value=None, visible=False)
            history_favorite = gr.State(False)
            history_rating = gr.State(0)
            history_review_status = gr.State('')
            history_tags = gr.State('')
            history_note = gr.State('')
            history_load_full_button = gr.Button(value='Load Full Config',
                                                 elem_id='history_load_full_button',
                                                 visible=False)
            history_replace_prompt_button = gr.Button(value='Replace Prompt',
                                                      elem_id='history_replace_prompt_button',
                                                      visible=False)
            history_append_prompt_button = gr.Button(value='Append Prompt',
                                                     elem_id='history_append_prompt_button',
                                                     visible=False)
            history_send_to_inpaint_button = gr.Button(value='Send to Inpaint',
                                                       elem_id='history_send_to_inpaint_button',
                                                       visible=False)
            history_save_curation_button = gr.Button(value='Save Curation',
                                                     elem_id='history_save_curation_button',
                                                     visible=False)
            with gr.Row():
                with gr.Column(scale=1, min_width=220):
                    with gr.Row(elem_id='history_thumbnail_bulk_actions'):
                        history_bulk_delete_button = gr.Button(value='🗑', elem_id='history_bulk_delete_button',
                                                                elem_classes='history-thumbnail-bulk-action')
                        history_bulk_favorite_button = gr.Button(value='★', elem_id='history_bulk_favorite_button',
                                                                  elem_classes='history-thumbnail-bulk-action')
                        history_bulk_hide_button = gr.Button(value='◉', elem_id='history_bulk_hide_button',
                                                              elem_classes='history-thumbnail-bulk-action')
                    history_gallery = gr.Gallery(label='Thumbnails', show_label=True, object_fit='cover',
                                                 columns=1, height=820, preview=False, allow_preview=False,
                                                 elem_id='history_thumbnail_gallery',
                                                 elem_classes=['image_gallery'])
                    with gr.Column(elem_id='history_thumbnail_view_controls'):
                        history_thumbnail_layout = gr.Radio(label='Thumbnail Layout',
                                                            choices=['Large (1 column)', 'Small (2 columns)'],
                                                            value='Large (1 column)',
                                                            elem_id='history_thumbnail_layout')
                        history_stack_by_seed = gr.Checkbox(label='Stack Matching Seeds', value=False,
                                                            elem_id='history_stack_by_seed')
                        history_filter_favorites = gr.Checkbox(label='Favorites Only', value=False)
                        history_thumbnail_visibility = gr.Radio(label='Thumbnail View',
                                                                choices=['Visible', 'All', 'Hidden'],
                                                                value='Visible',
                                                                elem_id='history_thumbnail_visibility')
                        history_show_preview_images = gr.Radio(label='Preview Images',
                                                               choices=['Finished only', 'Finished + previews', 'Previews only'],
                                                               value='Finished only',
                                                               elem_id='history_preview_visibility')
                with gr.Column(scale=4):
                    history_selected_gallery = gr.Gallery(label='Selected Images', show_label=True,
                                                          object_fit='contain', columns=2,
                                                          height=820, preview=False, allow_preview=False,
                                                          elem_id='history_selected_gallery',
                                                          elem_classes=['image_gallery'])
                    history_image_details = gr.HTML(elem_id='history_selected_image_details')
            history_status = gr.HTML(visible=False)
            with gr.Group():
                gr.Markdown('Filters')
                with gr.Row():
                    history_search = gr.Textbox(label='Positive Prompt Search', placeholder='Words in the positive prompt')
                    history_refresh_button = gr.Button(value='Refresh', variant='secondary')
                    history_requery_button = gr.Button(value='Re-query Outputs Folder', variant='secondary')
                history_day_selection = gr.CheckboxGroup(label='Output Days', choices=[], value=[],
                                                         elem_id='history_day_selection',
                                                         elem_classes=['history-day-picker'],
                                                         min_width=0)
                with gr.Row():
                    history_filter_checkpoints = gr.Dropdown(label='Checkpoints', choices=[], value=[],
                                                             multiselect=True)
                    history_filter_loras = gr.Dropdown(label='LoRAs', choices=[], value=[],
                                                       multiselect=True)
                history_seed_stack_selection = gr.Dropdown(label='Seed Group', choices=[], value=None,
                                                           visible=False)
                history_seed_stack_prompt = gr.Textbox(value='', visible=False)
                history_filter_tag = gr.State('')
                history_filter_status = gr.State('')
                history_batch_selection = gr.Dropdown(label='Generation Batch', choices=[], value='All Images',
                                                      visible=False)
            with gr.Accordion(label='Batch Details', open=False, visible=False):
                history_batch_rating = gr.State(0)
                with gr.Row():
                    history_batch_favorite = gr.Checkbox(label='Batch Favorite', value=False)
                    history_batch_review_status = gr.Dropdown(label='Batch Status',
                                                              choices=['', 'needs review', 'keeper', 'reject'],
                                                              value='')
                history_batch_tags = gr.Textbox(label='Batch Tags', placeholder='Comma separated tags')
                history_batch_note = gr.Textbox(label='Batch Note', lines=2)
                history_save_batch_curation_button = gr.Button(value='Save Batch', variant='secondary')
            with gr.Accordion(label='Comparison Table', open=False, visible=False):
                history_comparison_table = gr.Dataframe(
                    label='Comparison',
                    headers=['Checkpoint', 'Seed', 'Testing LoRA', 'Image ID', 'File', 'Favorite', 'Status', 'Tags'],
                    datatype=['str', 'str', 'str', 'str', 'str', 'str', 'str', 'str'],
                    value=[],
                    type='array',
                    interactive=False,
                    wrap=True
                )
        with gr.Tab(label='Image Generation', id='image_generation_tab'):
            currentTask = gr.State(worker.AsyncTask(args=[]))
            state_session_gallery = gr.State([])
            state_selected_generation_index = gr.State(None)
            quick_preview_mode = gr.State(False)
            inpaint_engine_state = gr.State('empty')
            with gr.Row(elem_id='generation_main_row'):
                with gr.Column(scale=2):
                    with gr.Accordion(label='Generation Preview & Session History', open=True,
                                      elem_id='generation_preview_history'):
                        with gr.Row(elem_id='generation_image_panel'):
                            with gr.Column(scale=1, elem_id='current_generation_panel'):
                                progress_window = grh.Image(label='Current Image', show_label=True, visible=True, height=640,
                                                            elem_classes=['main_view'])
                                progress_gallery = gr.Gallery(label='Finished Images', show_label=True, object_fit='contain',
                                                              height=640, visible=False, elem_classes=['main_view', 'image_gallery'])
                            with gr.Column(scale=1, elem_id='generation_history_panel'):
                                selected_generation_apply_index = gr.Textbox(value='',
                                                                              elem_id='selected_generation_apply_index',
                                                                              elem_classes='generation_apply_hidden_control')
                                apply_selected_image_config_button = gr.Button(value='Apply Selected Image Config',
                                                                               elem_id='apply_selected_image_config_button',
                                                                               elem_classes='generation_apply_hidden_control')
                                selected_generation_remove_index = gr.Textbox(value='',
                                                                               elem_id='selected_generation_remove_index',
                                                                               elem_classes='generation_apply_hidden_control')
                                remove_selected_image_button = gr.Button(value='Remove Selected Image',
                                                                         elem_id='remove_selected_image_button',
                                                                         elem_classes='generation_apply_hidden_control')
                                selected_generation_delete_index = gr.Textbox(value='',
                                                                               elem_id='selected_generation_delete_index',
                                                                               elem_classes='generation_apply_hidden_control')
                                delete_selected_image_button = gr.Button(value='Delete Selected Image',
                                                                         elem_id='delete_selected_image_button',
                                                                         elem_classes='generation_apply_hidden_control')
                                selected_generation_quality_index = gr.Textbox(value='',
                                                                                elem_id='selected_generation_quality_index',
                                                                                elem_classes='generation_apply_hidden_control')
                                regenerate_selected_quality_button = gr.Button(value='Regenerate Selected Preview at Quality',
                                                                               elem_id='regenerate_selected_quality_button',
                                                                               elem_classes='generation_apply_hidden_control')
                                selected_generation_favorite_index = gr.Textbox(value='',
                                                                                elem_id='selected_generation_favorite_index',
                                                                                elem_classes='generation_apply_hidden_control')
                                favorite_selected_generation_button = gr.Button(value='Favorite Selected Image',
                                                                                elem_id='favorite_selected_generation_button',
                                                                                elem_classes='generation_apply_hidden_control')
                                selected_generation_detail_index = gr.Textbox(value='',
                                                                              elem_id='selected_generation_detail_index',
                                                                              elem_classes='generation_apply_hidden_control')
                                show_selected_generation_detail_button = gr.Button(value='Show Selected Image Details',
                                                                                   elem_id='show_selected_generation_detail_button',
                                                                                   elem_classes='generation_apply_hidden_control')
                                quick_preview_generation_indices = gr.Textbox(value='[]',
                                                                              elem_id='quick_preview_generation_indices',
                                                                              elem_classes='generation_apply_hidden_control')
                                selected_queue_remove_id = gr.Textbox(value='',
                                                                      elem_id='selected_queue_remove_id',
                                                                      elem_classes='generation_apply_hidden_control')
                                remove_queued_task_button = gr.Button(value='Remove Queued Item',
                                                                      elem_id='remove_queued_task_button',
                                                                      elem_classes='generation_apply_hidden_control')
                                stop_queue_button = gr.Button(value='Stop Queue',
                                                              elem_id='stop_queue_button',
                                                              elem_classes='generation_apply_hidden_control')
                                gallery = gr.Gallery(label='Session History', show_label=True, object_fit='contain', visible=True, height=640,
                                                     elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                                                     elem_id='final_gallery')
                                clear_session_history_button = gr.Button(value='Clear Session History', variant='secondary',
                                                                         elem_id='clear_session_history_button')
                                selected_image_status = gr.HTML(elem_id='selected_generation_details', visible=False)
                    progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                            elem_id='progress-bar', elem_classes='progress-bar')
                    queue_status_html = gr.HTML(value=make_queue_panel_html(), elem_id='queue_status_panel')
                    with gr.Row(elem_id='queue_batch_controls'):
                        skip_button = gr.Button(label="Skip", value="Skip", elem_classes='type_row_half', elem_id='skip_button', visible=False)
                        stop_button = gr.Button(label="Stop", value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False)

                    def stop_clicked(currentTask):
                        import ldm_patched.modules.model_management as model_management
                        target_task = get_active_generation_task(currentTask)
                        if target_task is None:
                            return currentTask, gr.update(value=make_queue_panel_html())
                        target_task.last_stop = 'stop'
                        if target_task.processing:
                            model_management.interrupt_current_processing()
                        return target_task, gr.update(value=make_queue_panel_html())

                    def skip_clicked(currentTask):
                        import ldm_patched.modules.model_management as model_management
                        target_task = get_active_generation_task(currentTask)
                        if target_task is None:
                            return currentTask
                        target_task.last_stop = 'skip'
                        if target_task.processing:
                            model_management.interrupt_current_processing()
                        return target_task

                    stop_queue_button.click(stop_clicked, inputs=currentTask, outputs=[currentTask, queue_status_html], queue=False, show_progress=False, _js='cancelGenerateForever')
                    stop_button.click(stop_clicked, inputs=currentTask, outputs=[currentTask, queue_status_html], queue=False, show_progress=False, _js='cancelGenerateForever')
                    skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False)
                    with gr.Row(elem_id='prompt_action_row'):
                        with gr.Column(scale=17):
                            prompt = gr.Textbox(show_label=False, placeholder="Type prompt here or paste parameters.", elem_id='positive_prompt',
                                                autofocus=True, lines=3)
        
                            default_prompt = modules.config.default_prompt
                            if isinstance(default_prompt, str) and default_prompt != '':
                                shared.gradio_root.load(lambda: default_prompt, outputs=prompt)
        
                        with gr.Column(scale=3, min_width=0):
                            generate_button = gr.Button(label="Generate", value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True)
                            quick_preview_button = gr.Button(label="Quick Preview", value="Quick Preview", elem_classes='type_row', elem_id='quick_preview_button', visible=True)
                            load_parameter_button = gr.Button(label="Load Parameters", value="Load Parameters", elem_classes='type_row', elem_id='load_parameter_button', visible=False)
                    with gr.Row(elem_classes='advanced_check_row'):
                        input_image_checkbox = gr.Checkbox(label='Input Image', value=modules.config.default_image_prompt_checkbox, container=False, elem_classes='min_check')
                        enhance_checkbox = gr.Checkbox(label='Enhance', value=modules.config.default_enhance_checkbox, container=False, elem_classes='min_check')
                        advanced_checkbox = gr.Checkbox(label='Advanced', value=modules.config.default_advanced_checkbox, container=False, elem_classes='min_check')
                    with gr.Accordion(label='Prompt Configs', open=False):
                        with gr.Row():
                            prompt_config_name = gr.Textbox(label='Name', placeholder='Optional name for the current prompt config')
                            prompt_config_selection = gr.Dropdown(label='Saved', choices=modules.prompt_config.list_prompt_configs(), value=None)
                        with gr.Row():
                            save_prompt_config_button = gr.Button(value='Save Current Config', variant='secondary')
                            delete_prompt_config_button = gr.Button(value='Delete Selected Config', variant='secondary')
                        with gr.Row():
                            load_full_prompt_config_button = gr.Button(value='Load Full Config', variant='secondary')
                            replace_prompt_config_button = gr.Button(value='Replace Prompt', variant='secondary')
                            append_prompt_config_button = gr.Button(value='Append Prompt', variant='secondary')
                        prompt_config_status = gr.HTML()
                    with gr.Accordion(label='Wildprompt', open=False, elem_classes=['wildprompt_selections_tab']):
                        wildprompt_sorter.try_load_sorted_wildprompts()
                        wildprompt_generation_factors = gr.State(
                            value=wildprompt_sorter.build_generation_factors(
                                modules.config.default_image_number
                            )
                        )

                        wildprompt_folder_chips = gr.CheckboxGroup(
                            label='Folders (clear to show all)',
                            choices=wildprompt_sorter.get_wildprompt_folder_names(),
                            value=[],
                            elem_classes=['wildprompt-folder-chips']
                        )
                        with gr.Row():
                            wildprompt_search_bar = gr.Textbox(
                                label='Search',
                                placeholder='Find a wildprompt ...',
                                value='',
                                scale=4
                            )
                            wildprompt_reset = gr.Button(
                                value='Reset',
                                variant='secondary',
                                scale=1,
                                min_width=100
                            )
                        wildprompt_selections = gr.CheckboxGroup(show_label=False, container=False,
                                                                 choices=copy.deepcopy(wildprompt_sorter.all_wildprompts),
                                                                 value=copy.deepcopy(modules.config.default_wildprompts),
                                                                 label='Selected Wildprompts',
                                                                 elem_classes=['wildprompt_selections'])
                        wildprompt_generate_all = gr.CheckboxGroup(
                            label='Generate all rows for',
                            choices=copy.deepcopy(modules.config.default_wildprompts),
                            value=[],
                            info='Checked files use every selected row. Other applied files choose a random row for each combination.',
                            elem_classes=['wildprompt-generate-all-files']
                        )
                        gradio_receiver_wildprompt_selections = gr.Textbox(
                            elem_id='gradio_receiver_wildprompt_selections',
                            visible=False
                        )
                        wildprompt_line_selection_json = gr.Textbox(
                            value='{}',
                            elem_id='wildprompt_line_selection_json',
                            visible=False
                        )
                        wildprompt_combination_summary = gr.HTML(
                            value=wildprompt_sorter.build_wildprompt_combination_summary(
                                modules.config.default_wildprompts, [], '{}',
                                wildprompt_sorter.build_generation_factors(modules.config.default_image_number)
                            ),
                            elem_id='wildprompt_combination_summary'
                        )
                        wildprompt_line_section_ctrls = []
                        wildprompt_line_name_ctrls = []
                        wildprompt_line_selection_ctrls = []
                        wildprompt_line_all_buttons = []
                        wildprompt_line_none_buttons = []
                        for wildprompt_line_section_index in range(wildprompt_sorter.max_wildprompt_detail_sections):
                            with gr.Accordion(label='Wildprompt Rows', open=False, visible=False) as wildprompt_line_section:
                                wildprompt_line_name = gr.Textbox(value='', visible=False)
                                with gr.Row():
                                    wildprompt_line_all = gr.Button(value='All', variant='secondary')
                                    wildprompt_line_none = gr.Button(value='None', variant='secondary')
                                wildprompt_line_selection = gr.CheckboxGroup(
                                    show_label=False,
                                    container=False,
                                    choices=[],
                                    value=[],
                                    label='Selected Prompt Rows',
                                    elem_classes=['wildprompt_line_selections']
                                )
                            wildprompt_line_section_ctrls.append(wildprompt_line_section)
                            wildprompt_line_name_ctrls.append(wildprompt_line_name)
                            wildprompt_line_selection_ctrls.append(wildprompt_line_selection)
                            wildprompt_line_all_buttons.append(wildprompt_line_all)
                            wildprompt_line_none_buttons.append(wildprompt_line_none)
                        wildprompt_line_section_outputs = sum(
                            [[section, name, selection] for section, name, selection in zip(
                                wildprompt_line_section_ctrls,
                                wildprompt_line_name_ctrls,
                                wildprompt_line_selection_ctrls
                            )],
                            []
                        )

                        for wildprompt_line_name, wildprompt_line_selection, wildprompt_line_all, wildprompt_line_none in zip(
                                wildprompt_line_name_ctrls,
                                wildprompt_line_selection_ctrls,
                                wildprompt_line_all_buttons,
                                wildprompt_line_none_buttons):
                            wildprompt_line_all.click(
                                wildprompt_sorter.select_all_wildprompt_lines,
                                inputs=wildprompt_line_name,
                                outputs=wildprompt_line_selection,
                                queue=False,
                                show_progress=False
                            ).then(
                                wildprompt_sorter.encode_wildprompt_line_selections,
                                inputs=[wildprompt_selections] + wildprompt_line_selection_ctrls,
                                outputs=wildprompt_line_selection_json,
                                queue=False,
                                show_progress=False
                            ).then(
                                wildprompt_sorter.build_wildprompt_combination_summary,
                                inputs=[wildprompt_selections, wildprompt_generate_all,
                                        wildprompt_line_selection_json, wildprompt_generation_factors],
                                outputs=wildprompt_combination_summary,
                                queue=False,
                                show_progress=False
                            )
                            wildprompt_line_none.click(
                                wildprompt_sorter.select_no_wildprompt_lines,
                                outputs=wildprompt_line_selection,
                                queue=False,
                                show_progress=False
                            ).then(
                                wildprompt_sorter.encode_wildprompt_line_selections,
                                inputs=[wildprompt_selections] + wildprompt_line_selection_ctrls,
                                outputs=wildprompt_line_selection_json,
                                queue=False,
                                show_progress=False
                            ).then(
                                wildprompt_sorter.build_wildprompt_combination_summary,
                                inputs=[wildprompt_selections, wildprompt_generate_all,
                                        wildprompt_line_selection_json, wildprompt_generation_factors],
                                outputs=wildprompt_combination_summary,
                                queue=False,
                                show_progress=False
                            )

                        shared.gradio_root.load(
                            wildprompt_sorter.refresh_wildprompt_chip_browser,
                            inputs=[wildprompt_selections, wildprompt_folder_chips, wildprompt_search_bar],
                            outputs=[wildprompt_folder_chips, wildprompt_selections],
                            queue=False,
                            show_progress=False
                        )

                        wildprompt_search_bar.change(
                            wildprompt_sorter.filter_wildprompts_by_folders,
                            inputs=[wildprompt_selections, wildprompt_folder_chips,
                                    wildprompt_search_bar],
                            outputs=wildprompt_selections,
                            queue=False,
                            show_progress=False
                        ).then(
                            lambda: None, _js='()=>{refresh_wildprompt_localization();}')

                        wildprompt_folder_chips.change(
                            wildprompt_sorter.filter_wildprompts_by_folders,
                            inputs=[wildprompt_selections, wildprompt_folder_chips,
                                    wildprompt_search_bar],
                            outputs=wildprompt_selections,
                            queue=False,
                            show_progress=False
                        ).then(
                            lambda: None, _js='()=>{refresh_wildprompt_localization();}')

                        wildprompt_selections.change(
                            wildprompt_sorter.sync_wildprompt_generate_all_files,
                            inputs=[wildprompt_selections, wildprompt_generate_all],
                            outputs=wildprompt_generate_all,
                            queue=False,
                            show_progress=False
                        ).then(
                            wildprompt_sorter.update_wildprompt_line_sections,
                            inputs=[wildprompt_selections, wildprompt_line_selection_json],
                            outputs=wildprompt_line_section_outputs,
                            queue=False,
                            show_progress=False
                        ).then(
                            wildprompt_sorter.encode_wildprompt_line_selections,
                            inputs=[wildprompt_selections] + wildprompt_line_selection_ctrls,
                            outputs=wildprompt_line_selection_json,
                            queue=False,
                            show_progress=False
                        ).then(
                            wildprompt_sorter.build_wildprompt_combination_summary,
                            inputs=[wildprompt_selections, wildprompt_generate_all,
                                    wildprompt_line_selection_json, wildprompt_generation_factors],
                            outputs=wildprompt_combination_summary,
                            queue=False,
                            show_progress=False
                        )

                        for wildprompt_line_selection in wildprompt_line_selection_ctrls:
                            wildprompt_line_selection.change(
                                wildprompt_sorter.encode_wildprompt_line_selections,
                                inputs=[wildprompt_selections] + wildprompt_line_selection_ctrls,
                                outputs=wildprompt_line_selection_json,
                                queue=False,
                                show_progress=False
                            ).then(
                                wildprompt_sorter.build_wildprompt_combination_summary,
                                inputs=[wildprompt_selections, wildprompt_generate_all,
                                        wildprompt_line_selection_json, wildprompt_generation_factors],
                                outputs=wildprompt_combination_summary,
                                queue=False,
                                show_progress=False
                            )

                        wildprompt_generate_all.change(
                            wildprompt_sorter.build_wildprompt_combination_summary,
                            inputs=[wildprompt_selections, wildprompt_generate_all,
                                    wildprompt_line_selection_json, wildprompt_generation_factors],
                            outputs=wildprompt_combination_summary,
                            queue=False,
                            show_progress=False
                        )

                        wildprompt_reset.click(
                            wildprompt_sorter.reset_wildprompt_browser,
                            outputs=[wildprompt_folder_chips, wildprompt_search_bar,
                                     wildprompt_selections],
                            queue=False,
                            show_progress=False
                        ).then(
                            wildprompt_sorter.sync_wildprompt_generate_all_files,
                            inputs=[wildprompt_selections],
                            outputs=wildprompt_generate_all,
                            queue=False,
                            show_progress=False
                        ).then(
                            wildprompt_sorter.update_wildprompt_line_sections,
                            inputs=[wildprompt_selections, wildprompt_line_selection_json],
                            outputs=wildprompt_line_section_outputs,
                            queue=False,
                            show_progress=False
                        ).then(
                            wildprompt_sorter.encode_wildprompt_line_selections,
                            inputs=[wildprompt_selections] + wildprompt_line_selection_ctrls,
                            outputs=wildprompt_line_selection_json,
                            queue=False,
                            show_progress=False
                        ).then(
                            wildprompt_sorter.build_wildprompt_combination_summary,
                            inputs=[wildprompt_selections, wildprompt_generate_all,
                                    wildprompt_line_selection_json, wildprompt_generation_factors],
                            outputs=wildprompt_combination_summary,
                            queue=False,
                            show_progress=False
                        ).then(
                            lambda: None, _js='()=>{refresh_wildprompt_localization();}')

                        gradio_receiver_wildprompt_selections.input(
                            wildprompt_sorter.sort_wildprompts,
                            inputs=wildprompt_selections,
                            outputs=wildprompt_selections,
                            queue=False,
                            show_progress=False
                        ).then(
                            wildprompt_sorter.filter_wildprompts_by_folders,
                            inputs=[wildprompt_selections, wildprompt_folder_chips,
                                    wildprompt_search_bar],
                            outputs=wildprompt_selections,
                            queue=False,
                            show_progress=False
                        ).then(
                            lambda: None, _js='()=>{refresh_wildprompt_localization();}')

                        wildprompt_refresh = gr.Button(label='Refresh', value='Refresh All Wildprompts',
                                                       variant='secondary', elem_classes='refresh_button')

                        wildprompt_refresh.click(
                            wildprompt_sorter.refresh_wildprompt_chip_browser,
                            inputs=[wildprompt_selections, wildprompt_folder_chips, wildprompt_search_bar],
                            outputs=[wildprompt_folder_chips, wildprompt_selections],
                            queue=False,
                            show_progress=False
                        ).then(
                            wildprompt_sorter.sync_wildprompt_generate_all_files,
                            inputs=[wildprompt_selections, wildprompt_generate_all],
                            outputs=wildprompt_generate_all,
                            queue=False,
                            show_progress=False
                        ).then(
                            wildprompt_sorter.update_wildprompt_line_sections,
                            inputs=[wildprompt_selections, wildprompt_line_selection_json],
                            outputs=wildprompt_line_section_outputs,
                            queue=False,
                            show_progress=False
                        ).then(
                            wildprompt_sorter.encode_wildprompt_line_selections,
                            inputs=[wildprompt_selections] + wildprompt_line_selection_ctrls,
                            outputs=wildprompt_line_selection_json,
                            queue=False,
                            show_progress=False
                        ).then(
                            wildprompt_sorter.build_wildprompt_combination_summary,
                            inputs=[wildprompt_selections, wildprompt_generate_all,
                                    wildprompt_line_selection_json, wildprompt_generation_factors],
                            outputs=wildprompt_combination_summary,
                            queue=False,
                            show_progress=False
                        )

                    with gr.Accordion(label='Person Likeness', open=False):
                        person_likeness_ctrls = []
                        with gr.Row():
                            saved_person_name = gr.Textbox(label='Name', placeholder='Person name')
                            saved_person_selection = gr.Dropdown(
                                label='Saved People',
                                choices=list_saved_people(),
                                value=None
                            )
                            save_person_button = gr.Button(value='Save Person', variant='secondary')
                            load_person_button = gr.Button(value='Load Person', variant='secondary')
                        saved_person_status = gr.HTML()
                        person_likeness_files = gr.File(
                            label='Drop Photos',
                            file_count='multiple',
                            file_types=['image'],
                            type='file',
                            elem_id='person_likeness_upload'
                        )
                        person_likeness_paths = gr.Textbox(
                            value='[]',
                            visible=False,
                            elem_id='person_likeness_paths'
                        )
                        person_likeness_gallery = gr.Gallery(
                            label='Selected Photos',
                            show_label=True,
                            elem_id='person_likeness_gallery',
                            columns=6,
                            object_fit='cover',
                            height=330,
                            allow_preview=False,
                            show_download_button=False
                        )
                        person_likeness_refresh_button = gr.Button(
                            value='Refresh Person Thumbnails',
                            visible=False,
                            elem_id='person_likeness_refresh_button'
                        )
                        with gr.Row():
                            person_likeness_enabled = gr.Checkbox(label='Enable Person Likeness', value=True)
                            person_likeness_class = gr.Radio(
                                label='Subject',
                                choices=flags.person_likeness_classes,
                                value='person',
                                container=False
                            )
                            person_likeness_strength = gr.Slider(
                                label='Identity Strength',
                                minimum=0.0,
                                maximum=modules.config.default_person_likeness_strength_max,
                                step=0.001,
                                value=1.0
                            )
                            person_likeness_face_weight = gr.Slider(
                                label='Face Weight',
                                minimum=0.0,
                                maximum=modules.config.default_person_likeness_face_weight_max,
                                step=0.001,
                                value=modules.config.default_person_likeness_face_weight
                            )
                            person_likeness_face_start = gr.Slider(
                                label='Face Weight Start At',
                                minimum=0.0,
                                maximum=1.0,
                                step=0.001,
                                value=modules.config.default_person_likeness_face_start
                            )
                        with gr.Row():
                            person_likeness_preset = gr.Dropdown(
                                label='Preset',
                                choices=list(PERSON_LIKENESS_PRESETS.keys()),
                                value=None
                            )
                            clear_person_likeness_button = gr.Button(
                                value='Clear Settings',
                                variant='secondary'
                            )
                        person_likeness_ctrls = [person_likeness_enabled, person_likeness_class,
                                                 person_likeness_strength,
                                                 person_likeness_face_weight,
                                                 person_likeness_face_start,
                                                 person_likeness_paths]
                        save_person_button.click(
                            save_person_likeness,
                            inputs=[saved_person_name, person_likeness_enabled, person_likeness_class,
                                    person_likeness_strength, person_likeness_face_weight,
                                    person_likeness_face_start, person_likeness_paths],
                            outputs=[saved_person_selection, saved_person_status, person_likeness_paths,
                                     person_likeness_gallery],
                            queue=False,
                            show_progress=False
                        )
                        load_person_button.click(
                            load_person_likeness,
                            inputs=[saved_person_selection],
                            outputs=[person_likeness_enabled, person_likeness_class, person_likeness_strength,
                                     person_likeness_face_weight, person_likeness_face_start,
                                     person_likeness_paths, saved_person_status],
                            queue=False,
                            show_progress=False
                        ).then(
                            preview_person_likeness_paths,
                            inputs=person_likeness_paths,
                            outputs=person_likeness_gallery,
                            queue=False,
                            show_progress=False
                        )
                        person_likeness_preset.change(
                            apply_person_likeness_preset,
                            inputs=person_likeness_preset,
                            outputs=[person_likeness_strength, person_likeness_face_weight,
                                     person_likeness_face_start],
                            queue=False,
                            show_progress=False
                        )
                        clear_person_likeness_button.click(
                            clear_person_likeness_settings,
                            outputs=[person_likeness_enabled, person_likeness_class,
                                     person_likeness_strength, person_likeness_face_weight,
                                     person_likeness_face_start, person_likeness_preset,
                                     saved_person_status],
                            queue=False,
                            show_progress=False
                        )
                        person_likeness_files.change(
                            append_person_likeness_files,
                            inputs=[person_likeness_files, person_likeness_paths],
                            outputs=[person_likeness_paths, person_likeness_gallery, person_likeness_files],
                            queue=False,
                            show_progress=False
                        )
                        person_likeness_refresh_button.click(
                            preview_person_likeness_paths,
                            inputs=person_likeness_paths,
                            outputs=person_likeness_gallery,
                            queue=False,
                            show_progress=False
                        )
                        person_likeness_paths.change(
                            preview_person_likeness_paths,
                            inputs=person_likeness_paths,
                            outputs=person_likeness_gallery,
                            queue=False,
                            show_progress=False
                        )
                        gr.HTML('* Drop clear photos of the same person. Use one trigger phrase like "woman img", "man img", or "person img"; Fooocus will add it when omitted.')

                    with gr.Row(visible=modules.config.default_image_prompt_checkbox) as image_input_panel:
                        with gr.Tabs(selected=modules.config.default_selected_image_input_tab_id) as image_input_tabs:
                            with gr.Tab(label='Upscale or Variation', id='uov_tab') as uov_tab:
                                with gr.Row():
                                    with gr.Column():
                                        uov_input_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False)
                                    with gr.Column():
                                        uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list, value=modules.config.default_uov_method)
                                        gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Documentation</a>')
                            with gr.Tab(label='Image Prompt', id='ip_tab') as ip_tab:
                                with gr.Row():
                                    ip_images = []
                                    ip_types = []
                                    ip_stops = []
                                    ip_weights = []
                                    ip_ctrls = []
                                    ip_ad_cols = []
                                    for image_count in range(modules.config.default_controlnet_image_count):
                                        image_count += 1
                                        with gr.Column():
                                            ip_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False, height=300, value=modules.config.default_ip_images[image_count])
                                            ip_images.append(ip_image)
                                            ip_ctrls.append(ip_image)
                                            with gr.Column(visible=modules.config.default_image_prompt_advanced_checkbox) as ad_col:
                                                with gr.Row():
                                                    ip_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.001, value=modules.config.default_ip_stop_ats[image_count])
                                                    ip_stops.append(ip_stop)
                                                    ip_ctrls.append(ip_stop)
        
                                                    ip_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.001, value=modules.config.default_ip_weights[image_count])
                                                    ip_weights.append(ip_weight)
                                                    ip_ctrls.append(ip_weight)
        
                                                ip_type = gr.Radio(label='Type', choices=flags.ip_list, value=modules.config.default_ip_types[image_count], container=False)
                                                ip_types.append(ip_type)
                                                ip_ctrls.append(ip_type)
        
                                                ip_type.change(lambda x: flags.default_parameters[x], inputs=[ip_type], outputs=[ip_stop, ip_weight], queue=False, show_progress=False)
                                            ip_ad_cols.append(ad_col)
                                ip_advanced = gr.Checkbox(label='Advanced', value=modules.config.default_image_prompt_advanced_checkbox, container=False)
                                gr.HTML('* \"Image Prompt\" is powered by Fooocus Image Mixture Engine (v1.0.1). <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">\U0001F4D4 Documentation</a>')
        
                                def ip_advance_checked(x):
                                    return [gr.update(visible=x)] * len(ip_ad_cols) + \
                                        [flags.default_ip] * len(ip_types) + \
                                        [flags.default_parameters[flags.default_ip][0]] * len(ip_stops) + \
                                        [flags.default_parameters[flags.default_ip][1]] * len(ip_weights)
        
                                ip_advanced.change(ip_advance_checked, inputs=ip_advanced,
                                                   outputs=ip_ad_cols + ip_types + ip_stops + ip_weights,
                                                   queue=False, show_progress=False)
        
                            with gr.Tab(label='Inpaint or Outpaint', id='inpaint_tab') as inpaint_tab:
                                with gr.Row():
                                    with gr.Column():
                                        inpaint_input_image = grh.Image(label='Image', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", elem_id='inpaint_canvas', show_label=False)
                                        inpaint_advanced_masking_checkbox = gr.Checkbox(label='Enable Advanced Masking Features', value=modules.config.default_inpaint_advanced_masking_checkbox)
                                        inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.config.default_inpaint_method, label='Method')
                                        inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=False)
                                        outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='Outpaint Direction')
                                        example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts,
                                                                             label='Additional Prompt Quick List',
                                                                             components=[inpaint_additional_prompt],
                                                                             visible=False)
                                        gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Documentation</a>')
                                        example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)
        
                                    with gr.Column(visible=modules.config.default_inpaint_advanced_masking_checkbox) as inpaint_mask_generation_col:
                                        inpaint_mask_image = grh.Image(label='Mask Upload', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", mask_opacity=1, elem_id='inpaint_mask_canvas')
                                        invert_mask_checkbox = gr.Checkbox(label='Invert Mask When Generating', value=modules.config.default_invert_mask_checkbox)
                                        inpaint_mask_model = gr.Dropdown(label='Mask generation model',
                                                                         choices=flags.inpaint_mask_models,
                                                                         value=modules.config.default_inpaint_mask_model)
                                        inpaint_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                                     choices=flags.inpaint_mask_cloth_category,
                                                                     value=modules.config.default_inpaint_mask_cloth_category,
                                                                     visible=False)
                                        inpaint_mask_dino_prompt_text = gr.Textbox(label='Detection prompt', value='', visible=False, info='Use singular whenever possible', placeholder='Describe what you want to detect.')
                                        example_inpaint_mask_dino_prompt_text = gr.Dataset(
                                            samples=modules.config.example_enhance_detection_prompts,
                                            label='Detection Prompt Quick List',
                                            components=[inpaint_mask_dino_prompt_text],
                                            visible=modules.config.default_inpaint_mask_model == 'sam')
                                        example_inpaint_mask_dino_prompt_text.click(lambda x: x[0],
                                                                                    inputs=example_inpaint_mask_dino_prompt_text,
                                                                                    outputs=inpaint_mask_dino_prompt_text,
                                                                                    show_progress=False, queue=False)
        
                                        with gr.Accordion("Advanced options", visible=False, open=False) as inpaint_mask_advanced_options:
                                            inpaint_mask_sam_model = gr.Dropdown(label='SAM model', choices=flags.inpaint_mask_sam_model, value=modules.config.default_inpaint_mask_sam_model)
                                            inpaint_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                                            inpaint_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05)
                                            inpaint_mask_sam_max_detections = gr.Slider(label="Maximum number of detections", info="Set to 0 to detect all", minimum=0, maximum=10, value=modules.config.default_sam_max_detections, step=1, interactive=True)
                                        generate_mask_button = gr.Button(value='Generate mask from image')
        
                                        def generate_mask(image, mask_model, cloth_category, dino_prompt_text, sam_model, box_threshold, text_threshold, sam_max_detections, dino_erode_or_dilate, dino_debug):
                                            from extras.inpaint_mask import generate_mask_from_image
        
                                            extras = {}
                                            sam_options = None
                                            if mask_model == 'u2net_cloth_seg':
                                                extras['cloth_category'] = cloth_category
                                            elif mask_model == 'sam':
                                                sam_options = SAMOptions(
                                                    dino_prompt=dino_prompt_text,
                                                    dino_box_threshold=box_threshold,
                                                    dino_text_threshold=text_threshold,
                                                    dino_erode_or_dilate=dino_erode_or_dilate,
                                                    dino_debug=dino_debug,
                                                    max_detections=sam_max_detections,
                                                    model_type=sam_model
                                                )
        
                                            mask, _, _, _ = generate_mask_from_image(image, mask_model, extras, sam_options)
        
                                            return mask
        
        
                                        inpaint_mask_model.change(lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                                                            [gr.update(visible=x == 'sam')] * 2 +
                                                                            [gr.Dataset.update(visible=x == 'sam',
                                                                                               samples=modules.config.example_enhance_detection_prompts)],
                                                                  inputs=inpaint_mask_model,
                                                                  outputs=[inpaint_mask_cloth_category,
                                                                           inpaint_mask_dino_prompt_text,
                                                                           inpaint_mask_advanced_options,
                                                                           example_inpaint_mask_dino_prompt_text],
                                                                  queue=False, show_progress=False)
        
                            with gr.Tab(label='Describe', id='describe_tab') as describe_tab:
                                with gr.Row():
                                    with gr.Column():
                                        describe_input_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False)
                                    with gr.Column():
                                        describe_methods = gr.CheckboxGroup(
                                            label='Content Type',
                                            choices=flags.describe_types,
                                            value=modules.config.default_describe_content_type)
                                        describe_apply_styles = gr.Checkbox(label='Apply Styles', value=modules.config.default_describe_apply_prompts_checkbox)
                                        describe_btn = gr.Button(value='Describe this Image into Prompt')
                                        describe_image_size = gr.Textbox(label='Image Size and Recommended Size', elem_id='describe_image_size', visible=False)
                                        gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/1363" target="_blank">\U0001F4D4 Documentation</a>')
        
                                        def trigger_show_image_properties(image):
                                            value = modules.util.get_image_size_info(image, modules.flags.sdxl_aspect_ratios)
                                            return gr.update(value=value, visible=True)
        
                                        describe_input_image.upload(trigger_show_image_properties, inputs=describe_input_image,
                                                                    outputs=describe_image_size, show_progress=False, queue=False)
        
                            with gr.Tab(label='Enhance', id='enhance_tab') as enhance_tab:
                                with gr.Row():
                                    with gr.Column():
                                        enhance_input_image = grh.Image(label='Use with Enhance, skips image generation', source='upload', type='numpy')
                                        gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')
        
                            with gr.Tab(label='Metadata', id='metadata_tab') as metadata_tab:
                                with gr.Column():
                                    metadata_input_image = grh.Image(label='For images created by Fooocus', source='upload', type='pil')
                                    metadata_json = gr.JSON(label='Metadata')
                                    metadata_import_button = gr.Button(value='Apply Metadata')
        
                                def trigger_metadata_preview(file):
                                    parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
        
                                    results = {}
                                    if parameters is not None:
                                        results['parameters'] = parameters
        
                                    if isinstance(metadata_scheme, flags.MetadataScheme):
                                        results['metadata_scheme'] = metadata_scheme.value
        
                                    return results
        
                                metadata_input_image.upload(trigger_metadata_preview, inputs=metadata_input_image,
                                                            outputs=metadata_json, queue=False, show_progress=True)
        
                    with gr.Row(visible=modules.config.default_enhance_checkbox) as enhance_input_panel:
                        with gr.Tabs():
                            with gr.Tab(label='Upscale or Variation'):
                                with gr.Row():
                                    with gr.Column():
                                        enhance_uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list,
                                                                      value=modules.config.default_enhance_uov_method)
                                        enhance_uov_processing_order = gr.Radio(label='Order of Processing',
                                                                                info='Use before to enhance small details and after to enhance large areas.',
                                                                                choices=flags.enhancement_uov_processing_order,
                                                                                value=modules.config.default_enhance_uov_processing_order)
                                        enhance_uov_prompt_type = gr.Radio(label='Prompt',
                                                                           info='Choose which prompt to use for Upscale or Variation.',
                                                                           choices=flags.enhancement_uov_prompt_types,
                                                                           value=modules.config.default_enhance_uov_prompt_type,
                                                                           visible=modules.config.default_enhance_uov_processing_order == flags.enhancement_uov_after)
        
                                        enhance_uov_processing_order.change(lambda x: gr.update(visible=x == flags.enhancement_uov_after),
                                                                            inputs=enhance_uov_processing_order,
                                                                            outputs=enhance_uov_prompt_type,
                                                                            queue=False, show_progress=False)
                                        gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')
                            enhance_ctrls = []
                            enhance_inpaint_mode_ctrls = []
                            enhance_inpaint_engine_ctrls = []
                            enhance_inpaint_update_ctrls = []
                            for index in range(modules.config.default_enhance_tabs):
                                with gr.Tab(label=f'#{index + 1}') as enhance_tab_item:
                                    enhance_enabled = gr.Checkbox(label='Enable', value=False, elem_classes='min_check',
                                                                  container=False)
        
                                    enhance_mask_dino_prompt_text = gr.Textbox(label='Detection prompt',
                                                                               info='Use singular whenever possible',
                                                                               placeholder='Describe what you want to detect.',
                                                                               interactive=True,
                                                                               visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                                    example_enhance_mask_dino_prompt_text = gr.Dataset(
                                        samples=modules.config.example_enhance_detection_prompts,
                                        label='Detection Prompt Quick List',
                                        components=[enhance_mask_dino_prompt_text],
                                        visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                                    example_enhance_mask_dino_prompt_text.click(lambda x: x[0],
                                                                                inputs=example_enhance_mask_dino_prompt_text,
                                                                                outputs=enhance_mask_dino_prompt_text,
                                                                                show_progress=False, queue=False)
        
                                    enhance_prompt = gr.Textbox(label="Enhancement positive prompt",
                                                                placeholder="Uses original prompt instead if empty.",
                                                                elem_id='enhance_prompt')
                                    enhance_negative_prompt = gr.Textbox(label="Enhancement negative prompt",
                                                                         placeholder="Uses original negative prompt instead if empty.",
                                                                         elem_id='enhance_negative_prompt')
        
                                    with gr.Accordion("Detection", open=False):
                                        enhance_mask_model = gr.Dropdown(label='Mask generation model',
                                                                         choices=flags.inpaint_mask_models,
                                                                         value=modules.config.default_enhance_inpaint_mask_model)
                                        enhance_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                                                  choices=flags.inpaint_mask_cloth_category,
                                                                                  value=modules.config.default_inpaint_mask_cloth_category,
                                                                                  visible=modules.config.default_enhance_inpaint_mask_model == 'u2net_cloth_seg',
                                                                                  interactive=True)
        
                                        with gr.Accordion("SAM Options",
                                                          visible=modules.config.default_enhance_inpaint_mask_model == 'sam',
                                                          open=False) as sam_options:
                                            enhance_mask_sam_model = gr.Dropdown(label='SAM model',
                                                                                 choices=flags.inpaint_mask_sam_model,
                                                                                 value=modules.config.default_inpaint_mask_sam_model,
                                                                                 interactive=True)
                                            enhance_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0,
                                                                                   maximum=1.0, value=0.3, step=0.05,
                                                                                   interactive=True)
                                            enhance_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0,
                                                                                    maximum=1.0, value=0.25, step=0.05,
                                                                                    interactive=True)
                                            enhance_mask_sam_max_detections = gr.Slider(label="Maximum number of detections",
                                                                                        info="Set to 0 to detect all",
                                                                                        minimum=0, maximum=10,
                                                                                        value=modules.config.default_sam_max_detections,
                                                                                        step=1, interactive=True)
        
                                    with gr.Accordion("Inpaint", visible=True, open=False):
                                        enhance_inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options,
                                                                           value=modules.config.default_inpaint_method,
                                                                           label='Method', interactive=True)
                                        enhance_inpaint_disable_initial_latent = gr.Checkbox(
                                            label='Disable initial latent in inpaint', value=False)
                                        enhance_inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                                             value=modules.config.default_inpaint_engine_version,
                                                                             choices=flags.inpaint_engine_versions,
                                                                             info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.')
                                        enhance_inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                                             minimum=0.0, maximum=1.0, step=0.001,
                                                                             value=1.0,
                                                                             info='Same as the denoising strength in A1111 inpaint. '
                                                                                  'Only used in inpaint, not used in outpaint. '
                                                                                  '(Outpaint always use 1.0)')
                                        enhance_inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                                                     minimum=0.0, maximum=1.0, step=0.001,
                                                                                     value=0.618,
                                                                                     info='The area to inpaint. '
                                                                                          'Value 0 is same as "Only Masked" in A1111. '
                                                                                          'Value 1 is same as "Whole Image" in A1111. '
                                                                                          'Only used in inpaint, not used in outpaint. '
                                                                                          '(Outpaint always use 1.0)')
                                        enhance_inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                                                    minimum=-64, maximum=64, step=1, value=0,
                                                                                    info='Positive value will make white area in the mask larger, '
                                                                                         'negative value will make white area smaller. '
                                                                                         '(default is 0, always processed before any mask invert)')
                                        enhance_mask_invert = gr.Checkbox(label='Invert Mask', value=False)
        
                                    gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')
        
                                enhance_ctrls += [
                                    enhance_enabled,
                                    enhance_mask_dino_prompt_text,
                                    enhance_prompt,
                                    enhance_negative_prompt,
                                    enhance_mask_model,
                                    enhance_mask_cloth_category,
                                    enhance_mask_sam_model,
                                    enhance_mask_text_threshold,
                                    enhance_mask_box_threshold,
                                    enhance_mask_sam_max_detections,
                                    enhance_inpaint_disable_initial_latent,
                                    enhance_inpaint_engine,
                                    enhance_inpaint_strength,
                                    enhance_inpaint_respective_field,
                                    enhance_inpaint_erode_or_dilate,
                                    enhance_mask_invert
                                ]
        
                                enhance_inpaint_mode_ctrls += [enhance_inpaint_mode]
                                enhance_inpaint_engine_ctrls += [enhance_inpaint_engine]
        
                                enhance_inpaint_update_ctrls += [[
                                    enhance_inpaint_mode, enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                                    enhance_inpaint_strength, enhance_inpaint_respective_field
                                ]]
        
                                enhance_inpaint_mode.change(inpaint_mode_change, inputs=[enhance_inpaint_mode, inpaint_engine_state], outputs=[
                                    inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
                                    enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                                    enhance_inpaint_strength, enhance_inpaint_respective_field
                                ], show_progress=False, queue=False)
        
                                enhance_mask_model.change(
                                    lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                              [gr.update(visible=x == 'sam')] * 2 +
                                              [gr.Dataset.update(visible=x == 'sam',
                                                                 samples=modules.config.example_enhance_detection_prompts)],
                                    inputs=enhance_mask_model,
                                    outputs=[enhance_mask_cloth_category, enhance_mask_dino_prompt_text, sam_options,
                                             example_enhance_mask_dino_prompt_text],
                                    queue=False, show_progress=False)
        
                    switch_js = "(x) => {if(x){viewer_to_bottom(100);viewer_to_bottom(500);}else{viewer_to_top();} return x;}"
                    down_js = "() => {viewer_to_bottom();}"
        
                    input_image_checkbox.change(lambda x: gr.update(visible=x), inputs=input_image_checkbox,
                                                outputs=image_input_panel, queue=False, show_progress=False, _js=switch_js)
                    ip_advanced.change(lambda: None, queue=False, show_progress=False, _js=down_js)
        
                    current_tab = gr.Textbox(value='uov', visible=False)
                    uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
                    inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
                    ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
                    describe_tab.select(lambda: 'desc', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
                    enhance_tab.select(lambda: 'enhance', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
                    metadata_tab.select(lambda: 'metadata', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
                    enhance_checkbox.change(lambda x: gr.update(visible=x), inputs=enhance_checkbox,
                                                outputs=enhance_input_panel, queue=False, show_progress=False, _js=switch_js)
        
                with gr.Column(scale=1, visible=modules.config.default_advanced_checkbox, elem_id='advanced_column') as advanced_column:
                    with gr.Tab(label='Settings'):
                        if not args_manager.args.disable_preset_selection:
                            preset_selection = gr.Dropdown(label='Preset',
                                                           choices=modules.config.available_presets,
                                                           value=args_manager.args.preset if args_manager.args.preset else "initial",
                                                           interactive=True)
        
                        performance_selection = gr.Radio(label='Performance',
                                                         choices=flags.Performance.values(),
                                                         value=modules.config.default_performance,
                                                         elem_classes=['performance_selection'])
        
                        with gr.Accordion(label='Aspect Ratios', open=False, elem_id='aspect_ratios_accordion') as aspect_ratios_accordion:
                            aspect_ratios_selection = gr.Radio(label='Aspect Ratios', show_label=False,
                                                               choices=modules.config.available_aspect_ratios_labels,
                                                               value=modules.config.default_aspect_ratio,
                                                               info='width Ã— height',
                                                               elem_classes='aspect_ratios')
        
                            aspect_ratios_selection.change(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')
                            shared.gradio_root.load(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')
        
                        image_number = gr.Slider(label='Image Number', minimum=1, maximum=modules.config.default_max_image_number, step=1, value=modules.config.default_image_number)
        
                        output_format = gr.Radio(label='Output Format',
                                                 choices=flags.OutputFormat.list(),
                                                 value=modules.config.default_output_format)
        
                        negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True, placeholder="Type prompt here.",
                                                     info='Describing what you do not want to see.', lines=2,
                                                     elem_id='negative_prompt',
                                                     value=modules.config.default_prompt_negative)
                        seed_random = gr.Checkbox(label='Random', value=True)
                        image_seed = gr.Textbox(label='Seed', value=0, max_lines=1, visible=False) # workaround for https://github.com/gradio-app/gradio/issues/5354
                        training_mode = gr.Checkbox(label='Training Mode',
                                                    value=modules.config.default_training_mode,
                                                    info='Creates a LoRA training .txt caption file next to each generated image.')
                        testing_mode = gr.Checkbox(label='Testing Mode',
                                                   value=modules.config.default_testing_mode,
                                                   info='Generates Image Number images for each selected testing LoRA using the same seed.')
                        testing_loras = gr.Dropdown(label='Testing LoRAs',
                                                    choices=modules.config.lora_filenames,
                                                    value=[],
                                                    multiselect=True,
                                                    visible=modules.config.default_testing_mode)
                        history_link = gr.HTML(elem_id='history_link')
        
                        def random_checked(r):
                            return gr.update(visible=not r)
        
                        def refresh_seed(r, seed_string):
                            if r:
                                return random.randint(constants.MIN_SEED, constants.MAX_SEED)
                            else:
                                try:
                                    seed_value = int(seed_string)
                                    if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                                        return seed_value
                                except ValueError:
                                    pass
                                return random.randint(constants.MIN_SEED, constants.MAX_SEED)
        
                        seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed],
                                           queue=False, show_progress=False)
                        testing_mode.change(lambda x: gr.update(visible=x), inputs=testing_mode,
                                            outputs=testing_loras, queue=False, show_progress=False)
        
                        def update_history_link():
                            if args_manager.args.disable_image_log or not modules.config.save_history_log_html:
                                return gr.update(value='')

                            return gr.update(value=f'<a href="file={get_current_html_path(output_format)}" target="_blank">\U0001F4DA History Log</a>')

                        shared.gradio_root.load(update_history_link, outputs=history_link, queue=False, show_progress=False)
        
                    with gr.Tab(label='Styles', elem_classes=['style_selections_tab']):
                        style_sorter.try_load_sorted_styles(
                            style_names=legal_style_names,
                            default_selected=modules.config.default_styles)
        
                        style_search_bar = gr.Textbox(show_label=False, container=False,
                                                      placeholder="\U0001F50E Type here to search styles ...",
                                                      value="",
                                                      label='Search Styles')
                        style_selections = gr.CheckboxGroup(show_label=False, container=False,
                                                            choices=copy.deepcopy(style_sorter.all_styles),
                                                            value=copy.deepcopy(modules.config.default_styles),
                                                            label='Selected Styles',
                                                            elem_classes=['style_selections'])
                        gradio_receiver_style_selections = gr.Textbox(elem_id='gradio_receiver_style_selections', visible=False)
        
                        shared.gradio_root.load(lambda: gr.update(choices=copy.deepcopy(style_sorter.all_styles)),
                                                outputs=style_selections)
        
                        style_search_bar.change(style_sorter.search_styles,
                                                inputs=[style_selections, style_search_bar],
                                                outputs=style_selections,
                                                queue=False,
                                                show_progress=False).then(
                            lambda: None, _js='()=>{refresh_style_localization();}')
        
                        gradio_receiver_style_selections.input(style_sorter.sort_styles,
                                                               inputs=style_selections,
                                                               outputs=style_selections,
                                                               queue=False,
                                                               show_progress=False).then(
                            lambda: None, _js='()=>{refresh_style_localization();}')

                    with gr.Tab(label='Models'):
                        with gr.Group():
                            with gr.Row():
                                base_model = gr.Dropdown(label='Base Model (SDXL only)', choices=modules.config.model_filenames, value=modules.config.default_base_model_name, show_label=True)
                                refiner_model = gr.Dropdown(label='Refiner (SDXL or SD 1.5)', choices=['None'] + modules.config.model_filenames, value=modules.config.default_refiner_model_name, show_label=True)
                            multi_checkpoint_enabled = gr.Checkbox(label='Generate Across Multiple Checkpoints', value=False)
                            multi_checkpoint_models = gr.CheckboxGroup(
                                label='Base Models',
                                choices=modules.config.model_filenames,
                                value=[modules.config.default_base_model_name] if modules.config.default_base_model_name in modules.config.model_filenames else [],
                                visible=False
                            )
        
                            refiner_switch = gr.Slider(label='Refiner Switch At', minimum=0.1, maximum=1.0, step=0.0001,
                                                       info='Use 0.4 for SD1.5 realistic models; '
                                                            'or 0.667 for SD1.5 anime models; '
                                                            'or 0.8 for XL-refiners; '
                                                            'or any value for switching two SDXL models.',
                                                       value=modules.config.default_refiner_switch,
                                                       visible=modules.config.default_refiner_model_name != 'None')
        
                            refiner_model.change(lambda x: gr.update(visible=x != 'None'),
                                                 inputs=refiner_model, outputs=refiner_switch, show_progress=False, queue=False)
                            def multi_checkpoint_checked(enabled, current_base_model, selected_models):
                                if enabled and (not isinstance(selected_models, list) or len(selected_models) == 0):
                                    selected_models = [current_base_model]
                                return [gr.update(visible=not enabled), gr.update(visible=enabled, value=selected_models)]
        
                            multi_checkpoint_enabled.change(
                                multi_checkpoint_checked,
                                inputs=[multi_checkpoint_enabled, base_model, multi_checkpoint_models],
                                outputs=[base_model, multi_checkpoint_models],
                                show_progress=False,
                                queue=False
                            )

                            wildprompt_generation_factor_inputs = [
                                image_number, multi_checkpoint_enabled, multi_checkpoint_models,
                                testing_mode, testing_loras
                            ]
                            for generation_factor_control in [
                                image_number, multi_checkpoint_enabled, multi_checkpoint_models,
                                testing_mode, testing_loras
                            ]:
                                generation_factor_control.change(
                                    wildprompt_sorter.build_generation_factors,
                                    inputs=wildprompt_generation_factor_inputs,
                                    outputs=wildprompt_generation_factors,
                                    queue=False,
                                    show_progress=False
                                ).then(
                                    wildprompt_sorter.build_wildprompt_combination_summary,
                                    inputs=[wildprompt_selections, wildprompt_generate_all,
                                            wildprompt_line_selection_json, wildprompt_generation_factors],
                                    outputs=wildprompt_combination_summary,
                                    queue=False,
                                    show_progress=False
                                )
        
                        with gr.Group():
                            lora_ctrls = []
                            lora_prompt_ctrls = []
                            lora_note_buttons = []
                            lora_note_add_buttons = []
                            lora_note_editor_cols = []
        
                            for i, (enabled, filename, weight) in enumerate(modules.config.default_loras):
                                with gr.Row():
                                    lora_enabled = gr.Checkbox(label='Enable', value=enabled,
                                                               elem_classes=['lora_enable', 'min_check'], scale=1)
                                    lora_model = gr.Dropdown(label=f'LoRA {i + 1}',
                                                             choices=['None'] + modules.config.lora_filenames, value=filename,
                                                             elem_classes='lora_model', scale=5)
                                    lora_weight = gr.Slider(label='Weight', minimum=modules.config.default_loras_min_weight,
                                                            maximum=modules.config.default_loras_max_weight, step=0.01, value=weight,
                                                            elem_classes='lora_weight', scale=5)
                                    lora_note_button = gr.Button(value='\U0001f4dd', variant='secondary',
                                                                 visible=filename != 'None')
                                    saved_lora_note = modules.lora_notes.load_lora_note(filename)
                                    lora_note_add_button = gr.Button(value='+', variant='secondary',
                                                                     visible=filename != 'None' and saved_lora_note != '')
                                    lora_prompt = gr.Textbox(value=saved_lora_note, visible=False)
                                    lora_ctrls += [lora_enabled, lora_model, lora_weight]
                                    lora_prompt_ctrls.append(lora_prompt)
                                    lora_note_buttons.append(lora_note_button)
                                    lora_note_add_buttons.append(lora_note_add_button)
        
                                with gr.Column(visible=False) as lora_note_editor_col:
                                    lora_note_editor = gr.Textbox(label=f'LoRA {i + 1} Note', lines=3,
                                                                  placeholder='Prompt text to add when using this LoRA')
                                    with gr.Row():
                                        lora_note_save_button = gr.Button(value='Save', variant='secondary')
                                        lora_note_cancel_button = gr.Button(value='Cancel', variant='secondary')
                                    lora_note_editor_cols.append(lora_note_editor_col)
        
                                def lora_selection_changed(model_name):
                                    has_model = model_name != 'None'
                                    note = modules.lora_notes.load_lora_note(model_name)
                                    has_note = note != ''
                                    return gr.update(value=note), gr.update(visible=has_model), gr.update(visible=has_model and has_note), gr.update(visible=False)
        
                                def open_lora_note(note):
                                    return gr.update(visible=True), str(note or '')
        
                                def save_lora_note(model_name, note):
                                    note = modules.lora_notes.save_lora_note(model_name, note)
                                    return note, gr.update(visible=False), gr.update(visible=model_name != 'None' and note != '')
        
                                def cancel_lora_note(note):
                                    return gr.update(visible=False), str(note or '')
        
                                lora_model.change(lora_selection_changed, inputs=lora_model,
                                                  outputs=[lora_prompt, lora_note_button, lora_note_add_button, lora_note_editor_col],
                                                  show_progress=False, queue=False)
                                lora_note_button.click(open_lora_note, inputs=lora_prompt,
                                                       outputs=[lora_note_editor_col, lora_note_editor],
                                                       show_progress=False, queue=False)
                                lora_note_save_button.click(save_lora_note, inputs=[lora_model, lora_note_editor],
                                                            outputs=[lora_prompt, lora_note_editor_col, lora_note_add_button],
                                                            show_progress=False, queue=False)
                                lora_note_cancel_button.click(cancel_lora_note, inputs=lora_prompt,
                                                              outputs=[lora_note_editor_col, lora_note_editor],
                                                              show_progress=False, queue=False)
                                lora_note_add_button.click(append_lora_note_to_prompt, inputs=[prompt, lora_prompt],
                                                           outputs=prompt, show_progress=False, queue=False)
        
                        with gr.Row():
                            refresh_files = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button')
                            clear_all_loras = gr.Button(label='Clear All LoRAs', value='Clear All LoRAs', variant='secondary')
                    with gr.Tab(label='Advanced'):
                        reset_button = gr.Button(label="Reconnect", value="Reconnect", variant='secondary',
                                                 elem_id='reset_button', visible=False)
                        poll_generate_button = gr.Button(label="Poll Generation", value="Poll Generation",
                                                         elem_id='poll_generate_button', visible=False)
                        guidance_scale = gr.Slider(label='Guidance Scale', minimum=1.0, maximum=30.0, step=0.01,
                                                   value=modules.config.default_cfg_scale,
                                                   info='Higher value means style is cleaner, vivider, and more artistic.')
                        sharpness = gr.Slider(label='Image Sharpness', minimum=0.0, maximum=30.0, step=0.001,
                                              value=modules.config.default_sample_sharpness,
                                              info='Higher value means image and texture are sharper.')
                        gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/117" target="_blank">\U0001F4D4 Documentation</a>')
                        dev_mode = gr.Checkbox(label='Developer Debug Mode', value=modules.config.default_developer_debug_mode_checkbox, container=False)
        
                        with gr.Column(visible=modules.config.default_developer_debug_mode_checkbox) as dev_tools:
                            with gr.Tab(label='Debug Tools'):
                                adm_scaler_positive = gr.Slider(label='Positive ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                                step=0.001, value=1.5, info='The scaler multiplied to positive ADM (use 1.0 to disable). ')
                                adm_scaler_negative = gr.Slider(label='Negative ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                                step=0.001, value=0.8, info='The scaler multiplied to negative ADM (use 1.0 to disable). ')
                                adm_scaler_end = gr.Slider(label='ADM Guidance End At Step', minimum=0.0, maximum=1.0,
                                                           step=0.001, value=0.3,
                                                           info='When to end the guidance from positive/negative ADM. ')
        
                                refiner_swap_method = gr.Dropdown(label='Refiner swap method', value=flags.refiner_swap_method,
                                                                  choices=['joint', 'separate', 'vae'])
        
                                adaptive_cfg = gr.Slider(label='CFG Mimicking from TSNR', minimum=1.0, maximum=30.0, step=0.01,
                                                         value=modules.config.default_cfg_tsnr,
                                                         info='Enabling Fooocus\'s implementation of CFG mimicking for TSNR '
                                                              '(effective when real CFG > mimicked CFG).')
                                clip_skip = gr.Slider(label='CLIP Skip', minimum=1, maximum=flags.clip_skip_max, step=1,
                                                         value=modules.config.default_clip_skip,
                                                         info='Bypass CLIP layers to avoid overfitting (use 1 to not skip any layers, 2 is recommended).')
                                sampler_name = gr.Dropdown(label='Sampler', choices=flags.sampler_list,
                                                           value=modules.config.default_sampler)
                                scheduler_name = gr.Dropdown(label='Scheduler', choices=flags.scheduler_list,
                                                             value=modules.config.default_scheduler)
                                vae_name = gr.Dropdown(label='VAE', choices=[modules.flags.default_vae] + modules.config.vae_filenames,
                                                             value=modules.config.default_vae, show_label=True)
        
                                generate_image_grid = gr.Checkbox(label='Generate Image Grid for Each Batch',
                                                                  info='(Experimental) This may cause performance problems on some computers and certain internet conditions.',
                                                                  value=False)
        
                                overwrite_step = gr.Slider(label='Forced Overwrite of Sampling Step',
                                                           minimum=-1, maximum=200, step=1,
                                                           value=modules.config.default_overwrite_step,
                                                           info='Set as -1 to disable. For developer debugging.')
                                overwrite_switch = gr.Slider(label='Forced Overwrite of Refiner Switch Step',
                                                             minimum=-1, maximum=200, step=1,
                                                             value=modules.config.default_overwrite_switch,
                                                             info='Set as -1 to disable. For developer debugging.')
                                overwrite_width = gr.Slider(label='Forced Overwrite of Generating Width',
                                                            minimum=-1, maximum=2048, step=1, value=-1,
                                                            info='Set as -1 to disable. For developer debugging. '
                                                                 'Results will be worse for non-standard numbers that SDXL is not trained on.')
                                overwrite_height = gr.Slider(label='Forced Overwrite of Generating Height',
                                                             minimum=-1, maximum=2048, step=1, value=-1,
                                                             info='Set as -1 to disable. For developer debugging. '
                                                                  'Results will be worse for non-standard numbers that SDXL is not trained on.')
                                overwrite_vary_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Vary"',
                                                                    minimum=-1, maximum=1.0, step=0.001, value=-1,
                                                                    info='Set as negative number to disable. For developer debugging.')
                                overwrite_upscale_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Upscale"',
                                                                       minimum=-1, maximum=1.0, step=0.001,
                                                                       value=modules.config.default_overwrite_upscale,
                                                                       info='Set as negative number to disable. For developer debugging.')
        
                                disable_preview = gr.Checkbox(label='Disable Preview', value=modules.config.default_black_out_nsfw,
                                                              interactive=not modules.config.default_black_out_nsfw,
                                                              info='Disable preview during generation.')
                                disable_intermediate_results = gr.Checkbox(label='Disable Intermediate Results',
                                                              value=flags.Performance.has_restricted_features(modules.config.default_performance),
                                                              info='Disable intermediate results during generation, only show final gallery.')
        
                                disable_seed_increment = gr.Checkbox(label='Disable seed increment',
                                                                     info='Disable automatic seed increment when image number is > 1.',
                                                                     value=False)
                                read_wildcards_in_order = gr.Checkbox(label="Read wildcards in order", value=False)
        
                                black_out_nsfw = gr.Checkbox(label='Black Out NSFW', value=modules.config.default_black_out_nsfw,
                                                             interactive=not modules.config.default_black_out_nsfw,
                                                             info='Use black image if NSFW is detected.')
        
                                black_out_nsfw.change(lambda x: gr.update(value=x, interactive=not x),
                                                      inputs=black_out_nsfw, outputs=disable_preview, queue=False,
                                                      show_progress=False)
        
                                if not args_manager.args.disable_image_log:
                                    save_final_enhanced_image_only = gr.Checkbox(label='Save only final enhanced image',
                                                                                 value=modules.config.default_save_only_final_enhanced_image)

                                if not args_manager.args.disable_metadata:
                                    save_metadata_to_images = gr.Checkbox(label='Save Metadata to Images', value=modules.config.default_save_metadata_to_images,
                                                                          info='Adds parameters to generated images allowing manual regeneration.')
                                    metadata_scheme = gr.Radio(label='Metadata Scheme', choices=flags.metadata_scheme, value=modules.config.default_metadata_scheme,
                                                               info='Image Prompt parameters are not included. Use png and a1111 for compatibility with Civitai.',
                                                               visible=modules.config.default_save_metadata_to_images)
        
                                    save_metadata_to_images.change(lambda x: gr.update(visible=x), inputs=[save_metadata_to_images], outputs=[metadata_scheme],
                                                                   queue=False, show_progress=False)
        
                            with gr.Tab(label='Control'):
                                debugging_cn_preprocessor = gr.Checkbox(label='Debug Preprocessors', value=False,
                                                                        info='See the results from preprocessors.')
                                skipping_cn_preprocessor = gr.Checkbox(label='Skip Preprocessors', value=False,
                                                                       info='Do not preprocess images. (Inputs are already canny/depth/cropped-face/etc.)')
        
                                mixing_image_prompt_and_vary_upscale = gr.Checkbox(label='Mixing Image Prompt and Vary/Upscale',
                                                                                   value=False)
                                mixing_image_prompt_and_inpaint = gr.Checkbox(label='Mixing Image Prompt and Inpaint',
                                                                              value=False)
        
                                controlnet_softness = gr.Slider(label='Softness of ControlNet', minimum=0.0, maximum=1.0,
                                                                step=0.001, value=0.25,
                                                                info='Similar to the Control Mode in A1111 (use 0.0 to disable). ')
        
                                with gr.Tab(label='Canny'):
                                    canny_low_threshold = gr.Slider(label='Canny Low Threshold', minimum=1, maximum=255,
                                                                    step=1, value=64)
                                    canny_high_threshold = gr.Slider(label='Canny High Threshold', minimum=1, maximum=255,
                                                                     step=1, value=128)
        
                            with gr.Tab(label='Inpaint'):
                                debugging_inpaint_preprocessor = gr.Checkbox(label='Debug Inpaint Preprocessing', value=False)
                                debugging_enhance_masks_checkbox = gr.Checkbox(label='Debug Enhance Masks', value=False,
                                                                               info='Show enhance masks in preview and final results')
                                debugging_dino = gr.Checkbox(label='Debug GroundingDINO', value=False,
                                                             info='Use GroundingDINO boxes instead of more detailed SAM masks')
                                inpaint_disable_initial_latent = gr.Checkbox(label='Disable initial latent in inpaint', value=False)
                                inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                             value=modules.config.default_inpaint_engine_version,
                                                             choices=flags.inpaint_engine_versions,
                                                             info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.')
                                inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                             minimum=0.0, maximum=1.0, step=0.001, value=1.0,
                                                             info='Same as the denoising strength in A1111 inpaint. '
                                                                  'Only used in inpaint, not used in outpaint. '
                                                                  '(Outpaint always use 1.0)')
                                inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                                     minimum=0.0, maximum=1.0, step=0.001, value=0.618,
                                                                     info='The area to inpaint. '
                                                                          'Value 0 is same as "Only Masked" in A1111. '
                                                                          'Value 1 is same as "Whole Image" in A1111. '
                                                                          'Only used in inpaint, not used in outpaint. '
                                                                          '(Outpaint always use 1.0)')
                                inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                                    minimum=-64, maximum=64, step=1, value=0,
                                                                    info='Positive value will make white area in the mask larger, '
                                                                         'negative value will make white area smaller. '
                                                                         '(default is 0, always processed before any mask invert)')
                                dino_erode_or_dilate = gr.Slider(label='GroundingDINO Box Erode or Dilate',
                                                                 minimum=-64, maximum=64, step=1, value=0,
                                                                 info='Positive value will make white area in the mask larger, '
                                                                      'negative value will make white area smaller. '
                                                                      '(default is 0, processed before SAM)')
        
                                inpaint_mask_color = gr.ColorPicker(label='Inpaint brush color', value='#FFFFFF', elem_id='inpaint_brush_color')
        
                                inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                                                 inpaint_strength, inpaint_respective_field,
                                                 inpaint_advanced_masking_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate]
        
                                inpaint_advanced_masking_checkbox.change(lambda x: [gr.update(visible=x)] * 2,
                                                                         inputs=inpaint_advanced_masking_checkbox,
                                                                         outputs=[inpaint_mask_image, inpaint_mask_generation_col],
                                                                         queue=False, show_progress=False)
        
                                inpaint_mask_color.change(lambda x: gr.update(brush_color=x), inputs=inpaint_mask_color,
                                                          outputs=inpaint_input_image,
                                                          queue=False, show_progress=False)
        
                            with gr.Tab(label='FreeU'):
                                freeu_enabled = gr.Checkbox(label='Enabled', value=False)
                                freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
                                freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
                                freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
                                freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
                                freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]
        
                        def dev_mode_checked(r):
                            return gr.update(visible=r)
        
                        dev_mode.change(dev_mode_checked, inputs=[dev_mode], outputs=[dev_tools],
                                        queue=False, show_progress=False)
        
                        def refresh_files_clicked():
                            modules.config.update_files()
                            results = [gr.update(choices=modules.config.model_filenames)]
                            results += [gr.update(choices=modules.config.model_filenames)]
                            results += [gr.update(choices=['None'] + modules.config.model_filenames)]
                            results += [gr.update(choices=[flags.default_vae] + modules.config.vae_filenames)]
                            results += [gr.update(choices=modules.config.lora_filenames)]
                            if not args_manager.args.disable_preset_selection:
                                results += [gr.update(choices=modules.config.available_presets)]
                            for i in range(modules.config.default_max_lora_number):
                                results += [gr.update(interactive=True),
                                            gr.update(choices=['None'] + modules.config.lora_filenames), gr.update()]
                            return results

                        def clear_all_loras_clicked():
                            lora_updates = []
                            for _ in range(modules.config.default_max_lora_number):
                                lora_updates += [gr.update(), gr.update(value='None'), gr.update(value=1.0)]
                            return lora_updates + \
                                [gr.update(value='') for _ in lora_prompt_ctrls] + \
                                [gr.update(visible=False) for _ in lora_note_buttons] + \
                                [gr.update(visible=False) for _ in lora_note_add_buttons] + \
                                [gr.update(visible=False) for _ in lora_note_editor_cols]
        
                        refresh_files_output = [base_model, multi_checkpoint_models, refiner_model, vae_name, testing_loras]
                        if not args_manager.args.disable_preset_selection:
                            refresh_files_output += [preset_selection]
                        refresh_files.click(refresh_files_clicked, [], refresh_files_output + lora_ctrls,
                                            queue=False, show_progress=False)
                        clear_all_loras.click(clear_all_loras_clicked, [],
                                              lora_ctrls + lora_prompt_ctrls + lora_note_buttons +
                                              lora_note_add_buttons + lora_note_editor_cols,
                                              queue=False, show_progress=False)

                state_is_generating = gr.State(False)
                state_queue_monitor = gr.State(False)
        
                load_data_outputs = [advanced_checkbox, image_number, prompt, negative_prompt, style_selections,
                                     wildprompt_selections, wildprompt_generate_all, wildprompt_line_selection_json,
                                     performance_selection, overwrite_step, overwrite_switch, aspect_ratios_selection,
                                     overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                                     adm_scaler_negative, adm_scaler_end, refiner_swap_method, adaptive_cfg, clip_skip,
                                     base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, vae_name,
                                     seed_random, image_seed, inpaint_engine, inpaint_engine_state,
                                     inpaint_mode] + enhance_inpaint_mode_ctrls + [generate_button,
                                     load_parameter_button] + freeu_ctrls + lora_ctrls
        
                prompt_config_inputs = [
                    prompt, negative_prompt, style_selections, wildprompt_selections, wildprompt_generate_all,
                    wildprompt_line_selection_json,
                    performance_selection, overwrite_step, overwrite_switch,
                    aspect_ratios_selection, overwrite_width, overwrite_height, guidance_scale, sharpness,
                    adm_scaler_positive, adm_scaler_negative, adm_scaler_end, refiner_swap_method, adaptive_cfg, clip_skip,
                    base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, vae_name, seed_random,
                    image_seed, inpaint_engine, inpaint_mode
                ] + person_likeness_ctrls + freeu_ctrls + lora_ctrls + lora_prompt_ctrls
        
                def refresh_prompt_config_dropdown(selected=None):
                    return gr.update(choices=modules.prompt_config.list_prompt_configs(), value=selected)
        
                def save_current_prompt_config(name, *values):
                    config_data = build_prompt_config(*values)
                    selected = modules.prompt_config.save_prompt_config(name, config_data)
                    return refresh_prompt_config_dropdown(selected), f'Saved prompt config: {selected}'
        
                def delete_prompt_config(name):
                    if modules.prompt_config.delete_prompt_config(name):
                        return refresh_prompt_config_dropdown(), f'Deleted prompt config: {name}'
                    return refresh_prompt_config_dropdown(), 'No prompt config was deleted.'

                person_likeness_outputs = person_likeness_ctrls + [person_likeness_gallery]

                def person_likeness_config_to_ui_updates(config_data):
                    person_keys = [
                        'person_likeness_enabled',
                        'person_likeness_class',
                        'person_likeness_strength',
                        'person_likeness_face_weight',
                        'person_likeness_face_start',
                        'person_likeness_paths'
                    ]
                    if not any(key in config_data for key in person_keys):
                        return [gr.update()] * len(person_likeness_outputs)

                    def get_bool_config(key, default):
                        value = config_data.get(key, default)
                        if isinstance(value, str):
                            return value.strip().casefold() in ['true', '1', 'yes', 'on']
                        return bool(value)

                    enabled = get_bool_config('person_likeness_enabled', True)
                    subject = str(config_data.get('person_likeness_class', 'person'))
                    if subject not in flags.person_likeness_classes:
                        subject = 'person'
                    strength = clamp_float(config_data.get('person_likeness_strength', 1.0), 1.0, 0.0,
                                           modules.config.default_person_likeness_strength_max)
                    face_weight = clamp_float(config_data.get('person_likeness_face_weight',
                                                              modules.config.default_person_likeness_face_weight),
                                              modules.config.default_person_likeness_face_weight, 0.0,
                                              modules.config.default_person_likeness_face_weight_max)
                    face_start = clamp_float(config_data.get('person_likeness_face_start',
                                                             modules.config.default_person_likeness_face_start),
                                             modules.config.default_person_likeness_face_start, 0.0, 1.0)
                    if 'person_likeness_paths' in config_data:
                        paths = config_data.get('person_likeness_paths', '[]')
                        paths = paths if isinstance(paths, str) else '[]'
                        path_updates = [paths, preview_person_likeness_paths(paths)]
                    else:
                        path_updates = [gr.update(), gr.update()]
                    return [enabled, subject, strength, face_weight, face_start] + path_updates

                def prompt_config_to_ui_updates(config_data, is_generating, inpaint_mode, status):
                    if len(config_data) == 0:
                        return [gr.update()] * (len(load_data_outputs) + len(person_likeness_outputs) +
                                                len(lora_prompt_ctrls) + len(lora_note_buttons) +
                                                len(lora_note_add_buttons) + len(lora_note_editor_cols)) + [status]
        
                    lora_prompts = []
                    lora_note_button_updates = []
                    lora_note_add_button_updates = []
                    lora_note_editor_updates = []
                    for i in range(len(lora_prompt_ctrls)):
                        lora_config = str(config_data.get(f'lora_combined_{i + 1}', 'None'))
                        lora_parts = lora_config.split(' : ')
                        lora_model_name = lora_parts[1] if len(lora_parts) == 3 else lora_parts[0]
                        lora_prompt = modules.lora_notes.load_lora_note(lora_model_name)
                        if lora_prompt == '':
                            lora_prompt = config_data.get(f'lora_prompt_{i + 1}', '')
                        has_lora = lora_model_name != 'None'
                        has_note = str(lora_prompt or '').strip() != ''
                        lora_prompts.append(gr.update(value=lora_prompt))
                        lora_note_button_updates.append(gr.update(visible=has_lora))
                        lora_note_add_button_updates.append(gr.update(visible=has_lora and has_note))
                        lora_note_editor_updates.append(gr.update(visible=False))
                    return modules.meta_parser.load_parameter_button_click(config_data, is_generating, inpaint_mode) + \
                        person_likeness_config_to_ui_updates(config_data) + lora_prompts + lora_note_button_updates + \
                        lora_note_add_button_updates + lora_note_editor_updates + [status]

                def prompt_only_config_to_ui_updates(config_data, current_prompt, mode, status):
                    update_count = len(load_data_outputs) + len(person_likeness_outputs) + len(lora_prompt_ctrls) + \
                        len(lora_note_buttons) + len(lora_note_add_buttons) + len(lora_note_editor_cols)
                    updates = [gr.update()] * update_count

                    saved_prompt = str(config_data.get('prompt', config_data.get('Prompt', '')) or '')
                    current_prompt = str(current_prompt or '')

                    if mode == 'Append Prompt' and current_prompt.strip() != '' and saved_prompt.strip() != '':
                        next_prompt = f'{current_prompt}, {saved_prompt}'
                    elif mode == 'Append Prompt' and saved_prompt.strip() == '':
                        next_prompt = current_prompt
                    else:
                        next_prompt = saved_prompt

                    # load_data_outputs order: advanced checkbox, image number, prompt, ...
                    updates[2] = next_prompt
                    return updates + [status]

                def load_prompt_config(name, mode, current_prompt, is_generating, inpaint_mode):
                    config_data = modules.prompt_config.load_prompt_config(name)
                    if len(config_data) == 0:
                        return prompt_config_to_ui_updates(config_data, is_generating, inpaint_mode, 'Select a saved prompt config to load.')
                    if mode in ['Replace Prompt', 'Append Prompt']:
                        action = 'Replaced prompt from' if mode == 'Replace Prompt' else 'Appended prompt from'
                        return prompt_only_config_to_ui_updates(config_data, current_prompt, mode, f'{action}: {name}')
                    return prompt_config_to_ui_updates(config_data, is_generating, inpaint_mode, f'Loaded prompt config: {name}')

                def normalize_generation_index(selected_index):
                    try:
                        return int(selected_index)
                    except Exception:
                        return None

                def get_selected_generation_image_path(selected_index, session_history):
                    session_history = list(session_history or [])
                    selected_index = normalize_generation_index(selected_index)
                    if selected_index is None:
                        return None
                    if selected_index < 0 or selected_index >= len(session_history):
                        return None
                    image_path = session_history[selected_index]
                    return image_path if isinstance(image_path, str) else None

                def get_selected_generation_config(selected_index, session_history):
                    image_path = get_selected_generation_image_path(selected_index, session_history)
                    if image_path is None:
                        return {}, None

                    config_data = modules.history_db.get_config_by_path(image_path)
                    if len(config_data) > 0:
                        return config_data, image_path

                    config_data = worker.get_generated_image_config(image_path)
                    if len(config_data) > 0:
                        return config_data, image_path

                    parameters, metadata_scheme = modules.meta_parser.read_info_from_image(image_path)
                    if parameters is None or metadata_scheme is None:
                        return {}, image_path

                    metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                    return metadata_parser.to_json(parameters), image_path

                def is_empty_generation_detail_value(value):
                    if value is None:
                        return True
                    if isinstance(value, str):
                        return value == ''
                    if isinstance(value, (list, tuple, set, dict)):
                        return len(value) == 0
                    return False

                def selected_generation_config_value(config_data, summary, *keys):
                    for key in keys:
                        if isinstance(config_data, dict):
                            value = config_data.get(key)
                            if not is_empty_generation_detail_value(value):
                                return value
                        if isinstance(summary, dict):
                            value = summary.get(key)
                            if not is_empty_generation_detail_value(value):
                                return value
                    return ''

                def stringify_generation_detail(value):
                    if is_empty_generation_detail_value(value):
                        return ''
                    if isinstance(value, (list, tuple)):
                        return ', '.join([
                            stringify_generation_detail(item)
                            for item in value
                            if not is_empty_generation_detail_value(item)
                        ])
                    if isinstance(value, dict):
                        return json.dumps(value, ensure_ascii=False, sort_keys=True)
                    return str(value)

                def format_generation_detail_value(value, empty='Not recorded', multiline=False):
                    text = stringify_generation_detail(value).strip()
                    if text == '':
                        return f'<span class="generation-detail-muted">{html_lib.escape(empty)}</span>'
                    escaped = html_lib.escape(text)
                    if multiline:
                        escaped = escaped.replace('\n', '<br>')
                    return escaped

                def parse_resolved_wildprompts(value):
                    if isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except Exception:
                            try:
                                value = ast.literal_eval(value)
                            except Exception:
                                value = []
                    if not isinstance(value, list):
                        return []
                    resolved = []
                    for item in value:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get('name', '') or '').strip()
                        prompt_value = str(item.get('prompt', '') or '').strip()
                        if name == '' and prompt_value == '':
                            continue
                        resolved.append({'name': name, 'prompt': prompt_value})
                    return resolved

                def resolved_wildprompt_detail_rows(config_data, summary=None):
                    value = selected_generation_config_value(
                        config_data,
                        summary or {},
                        'resolved_wildprompts',
                    )
                    rows = []
                    for item in parse_resolved_wildprompts(value):
                        rows.append(generation_detail_row(
                            f"Wildprompt · {item['name'] or 'Unknown'}",
                            item['prompt'],
                            multiline=True,
                            full=True,
                        ))
                    return rows

                def parse_generation_lora_value(value):
                    if not isinstance(value, str):
                        return None
                    parts = [part.strip() for part in value.split(' : ')]
                    if len(parts) == 3:
                        if parts[0].casefold() not in ['true', '1', 'yes', 'on']:
                            return None
                        name = parts[1]
                        weight = parts[2]
                    elif len(parts) >= 2:
                        name = parts[0]
                        weight = parts[1]
                    else:
                        return None
                    if name == '' or name == 'None':
                        return None
                    return {'name': name, 'weight': weight, 'role': 'active'}

                def get_generation_lora_details(config_data, image_path):
                    loras = []
                    if isinstance(config_data, dict):
                        for index in range(modules.config.default_max_lora_number):
                            parsed = parse_generation_lora_value(config_data.get(f'lora_combined_{index + 1}'))
                            if parsed is not None:
                                loras.append(parsed)
                        testing_lora = stringify_generation_detail(config_data.get('testing_lora')).strip()
                        if testing_lora not in ['', 'None']:
                            loras.append({'name': testing_lora, 'weight': '1.0', 'role': 'testing'})

                    if len(loras) == 0:
                        image_id = modules.history_db.get_image_id_by_path(image_path)
                        if image_id is not None:
                            loras = modules.history_db.get_image_loras(image_id)
                    return loras

                def format_generation_loras(loras):
                    if len(loras or []) == 0:
                        return '<span class="generation-detail-muted">None recorded</span>'
                    formatted = []
                    for lora in loras:
                        name = html_lib.escape(str(lora.get('name', '') or ''))
                        weight = html_lib.escape(str(lora.get('weight', '') or ''))
                        role = str(lora.get('role', 'active') or 'active')
                        if name == '':
                            continue
                        suffix = f' ({weight})' if weight != '' else ''
                        if role == 'testing':
                            formatted.append(f'Testing: {name}{suffix}')
                        else:
                            formatted.append(f'{name}{suffix}')
                    return '<br>'.join(formatted) if len(formatted) > 0 else '<span class="generation-detail-muted">None recorded</span>'

                def generation_detail_row(label, value, multiline=False, full=False):
                    classes = 'generation-detail-row'
                    if full:
                        classes += ' generation-detail-row-full'
                    if multiline:
                        classes += ' generation-detail-row-multiline'
                    return (
                        f'<div class="{classes}">'
                        f'<div class="generation-detail-label">{html_lib.escape(label)}</div>'
                        f'<div class="generation-detail-value">{format_generation_detail_value(value, multiline=multiline)}</div>'
                        f'</div>'
                    )

                def format_selected_generation_details(selected_index, session_history):
                    selected_index = normalize_generation_index(selected_index)
                    if selected_index is None:
                        return ''
                    image_path = get_selected_generation_image_path(selected_index, session_history)
                    if image_path is None:
                        return '<div class="generation-detail-panel generation-detail-empty">Selected image is no longer in session history.</div>'

                    config_data, _ = get_selected_generation_config(selected_index, session_history)
                    image_id = modules.history_db.get_image_id_by_path(image_path)
                    summary = modules.history_db.get_image_summary(image_id) if image_id is not None else {}
                    filename = summary.get('filename') or os.path.basename(image_path)
                    prompt_text = selected_generation_config_value(config_data, summary, 'prompt')
                    checkpoint = selected_generation_config_value(config_data, summary, 'base_model', 'checkpoint')
                    refiner = selected_generation_config_value(config_data, summary, 'refiner_model', 'refiner')
                    loras = get_generation_lora_details(config_data, image_path)

                    title_parts = [f'Selected image #{selected_index + 1}']
                    if image_id is not None:
                        title_parts.append(f'History #{image_id}')
                    header = html_lib.escape(' | '.join(title_parts))

                    rows = [
                        generation_detail_row('Prompt', prompt_text, multiline=True, full=True),
                    ]
                    rows.extend(resolved_wildprompt_detail_rows(config_data, summary))
                    rows.append(generation_detail_row('Checkpoint', checkpoint, full=True))
                    if stringify_generation_detail(refiner).strip() not in ['', 'None']:
                        rows.append(generation_detail_row('Refiner', refiner, full=True))
                    rows.extend([
                        (
                            '<div class="generation-detail-row generation-detail-row-full generation-detail-row-multiline">'
                            '<div class="generation-detail-label">LoRA</div>'
                            f'<div class="generation-detail-value">{format_generation_loras(loras)}</div>'
                            '</div>'
                        ),
                        generation_detail_row('Seed', selected_generation_config_value(config_data, summary, 'seed')),
                        generation_detail_row('Steps', selected_generation_config_value(config_data, summary, 'steps')),
                        generation_detail_row('Sampler', selected_generation_config_value(config_data, summary, 'sampler')),
                        generation_detail_row('Scheduler', selected_generation_config_value(config_data, summary, 'scheduler')),
                        generation_detail_row('VAE', selected_generation_config_value(config_data, summary, 'vae')),
                        generation_detail_row('Resolution', selected_generation_config_value(config_data, summary, 'resolution')),
                        generation_detail_row('Performance', selected_generation_config_value(config_data, summary, 'performance')),
                        generation_detail_row('File', filename),
                    ])

                    return (
                        '<div class="generation-detail-panel">'
                        f'<div class="generation-detail-title">{header}</div>'
                        '<div class="generation-detail-grid">'
                        + ''.join(rows) +
                        '</div>'
                        '</div>'
                    )

                def select_generation_image_by_index(selected_index, session_history):
                    selected_index = normalize_generation_index(selected_index)
                    if selected_index is None:
                        return gr.update(), ''
                    return selected_index, format_selected_generation_details(selected_index, session_history)

                def select_generation_image(session_history, evt: gr.SelectData):
                    if evt is None or not hasattr(evt, 'index') or evt.index is None:
                        return gr.update(), gr.update()
                    selected_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                    return select_generation_image_by_index(selected_index, session_history)

                def apply_selected_generation_config(selected_index, session_history, is_generating, inpaint_mode):
                    config_data, image_path = get_selected_generation_config(selected_index, session_history)
                    if len(config_data) == 0:
                        return prompt_config_to_ui_updates(config_data, is_generating, inpaint_mode, 'Select a generated image with metadata first.')

                    return prompt_config_to_ui_updates(config_data, is_generating, inpaint_mode,
                                                       f'Applied config from {os.path.basename(image_path)}.')

                def get_selected_generation_quality_config(selected_index, session_history):
                    config_data, image_path = get_selected_generation_config(selected_index, session_history)
                    if len(config_data) == 0:
                        return {}, None, 'Select a quick preview with metadata first.'

                    config_data = config_data.copy()
                    config_data['performance'] = flags.Performance.QUALITY.value
                    config_data['steps'] = 60
                    config_data['quick_preview'] = False
                    config_data['image_number'] = 1
                    config_data['wildprompts'] = '[]'
                    config_data['wildprompt_generate_all'] = False
                    config_data['wildprompt_generate_all_files'] = '[]'
                    config_data['wildprompt_line_selections'] = '{}'
                    return config_data, image_path, f'Regenerating one image from {os.path.basename(image_path)} at Quality, 60 steps.'

                def toggle_session_generation_favorite(selected_index, session_history):
                    image_path = get_selected_generation_image_path(selected_index, session_history)
                    if image_path is None:
                        return 'Select a session history image first.'
                    image_id = modules.history_db.get_image_id_by_path(image_path)
                    if image_id is None:
                        return 'This image is not in the history database yet.'
                    curation = modules.history_db.get_image_curation(image_id)
                    if len(curation) == 0:
                        return 'History image was not found.'
                    next_favorite = not bool(curation.get('favorite', False))
                    modules.history_db.update_image_curation(
                        image_id,
                        next_favorite,
                        curation.get('rating', 0),
                        curation.get('review_status', ''),
                        curation.get('tags', ''),
                        curation.get('note', '')
                    )
                    return f"{'Favorited' if next_favorite else 'Unfavorited'} {os.path.basename(image_path)}."

                def parse_literal(value, expected_type, default):
                    if isinstance(value, expected_type):
                        return value
                    if isinstance(value, str):
                        try:
                            parsed = ast.literal_eval(value)
                            if isinstance(parsed, expected_type):
                                return parsed
                        except Exception:
                            pass
                    return default

                def apply_config_to_generation_args(config_data, args):
                    args = list(args)

                    def set_if_present(index, key, cast=lambda x: x):
                        if key in config_data and config_data[key] is not None:
                            try:
                                args[index] = cast(config_data[key])
                            except Exception:
                                pass

                    def cast_bool_config(value):
                        if isinstance(value, str):
                            return value.strip().casefold() in ['true', '1', 'yes', 'on']
                        return bool(value)

                    args[2] = False
                    set_if_present(3, 'prompt', str)
                    set_if_present(4, 'negative_prompt', str)
                    if 'styles' in config_data:
                        args[5] = parse_literal(config_data.get('styles'), list, args[5])
                    if 'wildprompts' in config_data:
                        args[6] = parse_literal(config_data.get('wildprompts'), list, args[6])
                    if 'wildprompt_generate_all_files' in config_data:
                        args[7] = modules.sdxl_styles.normalize_wildprompt_generate_all_files(
                            args[6],
                            config_data.get('wildprompt_generate_all_files'),
                        )
                    elif 'wildprompt_generate_all' in config_data:
                        args[7] = modules.sdxl_styles.normalize_wildprompt_generate_all_files(
                            args[6],
                            cast_bool_config(config_data.get('wildprompt_generate_all')),
                        )
                    set_if_present(8, 'wildprompt_line_selections', str)
                    args[9] = flags.Performance.QUALITY.value
                    args[11] = 1
                    set_if_present(13, 'seed', int)
                    set_if_present(15, 'sharpness', float)
                    set_if_present(16, 'guidance_scale', float)
                    set_if_present(17, 'base_model', str)
                    args[18] = False
                    args[19] = [args[17]]
                    set_if_present(20, 'refiner_model', str)
                    set_if_present(21, 'refiner_switch', float)
                    lora_start = 22
                    after_loras = lora_start + modules.config.default_max_lora_number * 3
                    set_if_present(after_loras + 2, 'person_likeness_enabled', cast_bool_config)
                    set_if_present(after_loras + 3, 'person_likeness_class', str)
                    set_if_present(after_loras + 4, 'person_likeness_strength', float)
                    set_if_present(after_loras + 5, 'person_likeness_face_weight', float)
                    set_if_present(after_loras + 6, 'person_likeness_face_start', float)
                    set_if_present(after_loras + 7, 'person_likeness_paths', str)

                    resolution = parse_literal(config_data.get('resolution'), tuple, None)
                    if resolution is None:
                        resolution = parse_literal(config_data.get('resolution'), list, None)
                    if isinstance(resolution, (tuple, list)) and len(resolution) >= 2:
                        width, height = int(resolution[0]), int(resolution[1])
                        formatted = modules.config.add_ratio(f'{width}*{height}')
                        if formatted in modules.config.available_aspect_ratios_labels:
                            args[10] = formatted
                            args[after_loras + 28] = -1
                            args[after_loras + 29] = -1
                        else:
                            args[after_loras + 28] = width
                            args[after_loras + 29] = height

                    performance_lora = flags.Performance.QUALITY.lora_filename()
                    for index in range(modules.config.default_max_lora_number):
                        value = config_data.get(f'lora_combined_{index + 1}')
                        if not isinstance(value, str):
                            continue
                        parts = value.split(' : ')
                        try:
                            enabled, name, weight = True, parts[0], parts[1]
                            if len(parts) == 3:
                                enabled, name, weight = parts[0] == 'True', parts[1], parts[2]
                            if name == performance_lora:
                                continue
                            base = lora_start + index * 3
                            args[base] = enabled
                            args[base + 1] = name
                            args[base + 2] = float(weight)
                        except Exception:
                            pass

                    set_if_present(after_loras + 23, 'sampler', str)
                    set_if_present(after_loras + 24, 'scheduler', str)
                    set_if_present(after_loras + 25, 'vae', str)
                    args[after_loras + 26] = 60
                    set_if_present(after_loras + 27, 'overwrite_switch', float)

                    adm = parse_literal(config_data.get('adm_guidance'), tuple, None)
                    if adm is None:
                        adm = parse_literal(config_data.get('adm_guidance'), list, None)
                    if isinstance(adm, (tuple, list)) and len(adm) >= 3:
                        args[after_loras + 18] = float(adm[0])
                        args[after_loras + 19] = float(adm[1])
                        args[after_loras + 20] = float(adm[2])

                    set_if_present(after_loras + 21, 'adaptive_cfg', float)
                    set_if_present(after_loras + 22, 'clip_skip', int)
                    set_if_present(after_loras + 38, 'refiner_swap_method', str)
                    set_if_present(after_loras + 47, 'inpaint_engine_version', str)

                    freeu = parse_literal(config_data.get('freeu'), tuple, None)
                    if freeu is None:
                        freeu = parse_literal(config_data.get('freeu'), list, None)
                    if isinstance(freeu, (tuple, list)) and len(freeu) >= 4:
                        args[after_loras + 40] = True
                        args[after_loras + 41] = float(freeu[0])
                        args[after_loras + 42] = float(freeu[1])
                        args[after_loras + 43] = float(freeu[2])
                        args[after_loras + 44] = float(freeu[3])

                    return args

                def enqueue_selected_generation_quality_config(selected_index, session_history, *generation_args):
                    config_data, image_path, status = get_selected_generation_quality_config(selected_index, session_history)
                    if len(config_data) == 0:
                        return worker.AsyncTask(args=[]), False, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value=make_queue_panel_html()), False, status

                    task_args = apply_config_to_generation_args(config_data, generation_args)
                    task = worker.AsyncTask(args=task_args[1:])
                    pending_count = worker.append_async_task(task)
                    should_monitor = worker.begin_queue_monitor()
                    tracking_task = get_generation_tracking_task(task)
                    print(f'[Queue] Added quality regeneration task. Pending tasks: {pending_count}')
                    return tracking_task, should_monitor, \
                        gr.update(visible=False, interactive=False), \
                        gr.update(visible=False, interactive=False), \
                        gr.update(visible=True, interactive=True), \
                        gr.update(visible=True, interactive=True), \
                        gr.update(), \
                        gr.update(value=make_queue_panel_html()), \
                        True, \
                        status

                def get_history_image_quality_config(image_id):
                    image_id = parse_history_id(image_id)
                    if image_id is None:
                        return {}, None, 'Select a preview image first.'
                    summary = modules.history_db.get_image_summary(image_id)
                    if len(summary) == 0:
                        return {}, None, 'History image was not found.'
                    if not history_row_is_preview(summary):
                        return {}, None, 'Select a preview image to regenerate at Quality.'
                    config_data = modules.history_db.get_config_by_image_id(image_id)
                    if len(config_data) == 0:
                        return {}, summary.get('path'), 'Selected preview image has no saved config.'

                    config_data = config_data.copy()
                    config_data['performance'] = flags.Performance.QUALITY.value
                    config_data['steps'] = 60
                    config_data['quick_preview'] = False
                    config_data['image_number'] = 1
                    config_data['wildprompts'] = '[]'
                    config_data['wildprompt_generate_all'] = False
                    config_data['wildprompt_generate_all_files'] = '[]'
                    config_data['wildprompt_line_selections'] = '{}'
                    image_path = summary.get('path') or ''
                    return config_data, image_path, f'Regenerating one image from {os.path.basename(image_path)} at Quality, 60 steps.'

                def enqueue_history_image_quality_config(image_id, *generation_args):
                    config_data, image_path, status = get_history_image_quality_config(image_id)
                    if len(config_data) == 0:
                        return worker.AsyncTask(args=[]), False, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value=make_queue_panel_html()), False, status

                    task_args = apply_config_to_generation_args(config_data, generation_args)
                    task = worker.AsyncTask(args=task_args[1:])
                    pending_count = worker.append_async_task(task)
                    should_monitor = worker.begin_queue_monitor()
                    tracking_task = get_generation_tracking_task(task)
                    print(f'[Queue] Added history preview quality regeneration task. Pending tasks: {pending_count}')
                    return tracking_task, should_monitor, \
                        gr.update(visible=False, interactive=False), \
                        gr.update(visible=False, interactive=False), \
                        gr.update(visible=True, interactive=True), \
                        gr.update(visible=True, interactive=True), \
                        gr.update(), \
                        gr.update(value=make_queue_panel_html()), \
                        True, \
                        status

                def remove_generation_from_history(selected_index, session_history):
                    session_history = list(session_history or [])
                    image_path = get_selected_generation_image_path(selected_index, session_history)
                    if image_path is None:
                        return gr.update(value=session_history), session_history, gr.update(value=None), 'Select a history image to remove.'

                    try:
                        selected_index = int(selected_index)
                    except Exception:
                        return gr.update(value=session_history), session_history, gr.update(value=None), 'Select a history image to remove.'

                    session_history.pop(selected_index)
                    preview_indices = [
                        index for index, path in enumerate(session_history)
                        if isinstance(path, str) and bool(worker.get_generated_image_config(path).get('quick_preview', False))
                    ]
                    return gr.update(value=session_history), session_history, gr.update(value=None), \
                        gr.update(value=json.dumps(preview_indices)), \
                        ''

                def clear_session_history():
                    return gr.update(value=[]), [], gr.update(value=None), \
                        gr.update(value='[]'), 'Session history cleared. Generated image files were kept.'

                def delete_generation_from_history(selected_index, session_history):
                    session_history = list(session_history or [])
                    image_path = get_selected_generation_image_path(selected_index, session_history)
                    if image_path is None:
                        return gr.update(value=session_history), session_history, gr.update(value=None), 'Select a history image to delete.'

                    def delete_after_gradio_postprocess(path):
                        time.sleep(1.0)
                        if os.path.exists(path) and os.path.isfile(path):
                            try:
                                os.remove(path)
                            except Exception as e:
                                print(f'[History] Could not delete {os.path.basename(path)}: {e}')

                    status = ''
                    if os.path.exists(image_path) and os.path.isfile(image_path):
                        threading.Thread(
                            target=delete_after_gradio_postprocess,
                            args=(image_path,),
                            daemon=True
                        ).start()
                    else:
                        status = f'Image file was already missing: {os.path.basename(image_path)}'

                    session_history = [path for index, path in enumerate(session_history) if index != int(selected_index)]
                    preview_indices = [
                        index for index, path in enumerate(session_history)
                        if isinstance(path, str) and bool(worker.get_generated_image_config(path).get('quick_preview', False))
                    ]
                    return gr.update(value=session_history), session_history, gr.update(value=None), \
                        gr.update(value=json.dumps(preview_indices)), status
        
                save_prompt_config_button.click(save_current_prompt_config, inputs=[prompt_config_name] + prompt_config_inputs,
                                                outputs=[prompt_config_selection, prompt_config_status],
                                                queue=False, show_progress=False)
                delete_prompt_config_button.click(delete_prompt_config, inputs=prompt_config_selection,
                                                  outputs=[prompt_config_selection, prompt_config_status],
                                                  queue=False, show_progress=False)

                load_prompt_config_outputs = load_data_outputs + person_likeness_outputs + lora_prompt_ctrls + \
                    lora_note_buttons + lora_note_add_buttons + lora_note_editor_cols + [prompt_config_status]
                history_load_outputs = load_data_outputs + person_likeness_outputs + lora_prompt_ctrls + \
                    lora_note_buttons + lora_note_add_buttons + lora_note_editor_cols + [history_status]

                def parse_history_id(selection):
                    text = str(selection or '').strip()
                    if text == '':
                        return None
                    if text.startswith('stack:'):
                        text = text[len('stack:'):]
                    text = re.sub(r'^#', '', text)
                    match = re.match(r'(\d+)', text.strip())
                    if match is None:
                        match = re.search(r'\b(\d+)\b', text)
                    if match is None:
                        return None
                    try:
                        return int(match.group(1))
                    except Exception:
                        return None

                def parse_history_id_list(values):
                    if isinstance(values, str):
                        try:
                            values = json.loads(values)
                        except Exception:
                            values = [values]
                    ids = []
                    for value in values or []:
                        parsed = parse_history_id(value)
                        if parsed is not None:
                            ids.append(parsed)
                    return ids

                def format_history_image_details(image_id):
                    image_id = parse_history_id(image_id)
                    if image_id is None:
                        return ''
                    summary = modules.history_db.get_image_summary(image_id)
                    if len(summary) == 0:
                        return '<div class="generation-detail-panel generation-detail-empty">History image was not found.</div>'

                    config_data = modules.history_db.get_config_by_image_id(image_id)
                    image_path = summary.get('path') or ''
                    filename = summary.get('filename') or os.path.basename(image_path)
                    prompt_text = selected_generation_config_value(config_data, summary, 'prompt')
                    checkpoint = selected_generation_config_value(config_data, summary, 'base_model', 'checkpoint')
                    refiner = selected_generation_config_value(config_data, summary, 'refiner_model', 'refiner')
                    loras = get_generation_lora_details(config_data, image_path)

                    rows = [
                        generation_detail_row('Prompt', prompt_text, multiline=True, full=True),
                    ]
                    rows.extend(resolved_wildprompt_detail_rows(config_data, summary))
                    rows.append(generation_detail_row('Checkpoint', checkpoint, full=True))
                    if stringify_generation_detail(refiner).strip() not in ['', 'None']:
                        rows.append(generation_detail_row('Refiner', refiner, full=True))

                    person_likeness_name = selected_generation_config_value(
                        config_data, summary, 'person_likeness_name'
                    )
                    person_likeness_sliders = [
                        ('Identity Strength', 'person_likeness_strength'),
                        ('Face Weight', 'person_likeness_face_weight'),
                        ('Face Weight Start At', 'person_likeness_face_start'),
                    ]
                    has_person_likeness_details = any(
                        not is_empty_generation_detail_value(
                            selected_generation_config_value(config_data, summary, key)
                        )
                        for _, key in person_likeness_sliders
                    )
                    if has_person_likeness_details:
                        if not is_empty_generation_detail_value(person_likeness_name):
                            rows.append(generation_detail_row('Person Likeness Name', person_likeness_name, full=True))
                        rows.extend([
                            generation_detail_row(label, selected_generation_config_value(config_data, summary, key))
                            for label, key in person_likeness_sliders
                        ])

                    rows.extend([
                        (
                            '<div class="generation-detail-row generation-detail-row-full generation-detail-row-multiline">'
                            '<div class="generation-detail-label">LoRA</div>'
                            f'<div class="generation-detail-value">{format_generation_loras(loras)}</div>'
                            '</div>'
                        ),
                        generation_detail_row('Seed', selected_generation_config_value(config_data, summary, 'seed')),
                        generation_detail_row('Steps', selected_generation_config_value(config_data, summary, 'steps')),
                        generation_detail_row('Sampler', selected_generation_config_value(config_data, summary, 'sampler')),
                        generation_detail_row('Scheduler', selected_generation_config_value(config_data, summary, 'scheduler')),
                        generation_detail_row('VAE', selected_generation_config_value(config_data, summary, 'vae')),
                        generation_detail_row('Resolution', selected_generation_config_value(config_data, summary, 'resolution')),
                        generation_detail_row('Performance', selected_generation_config_value(config_data, summary, 'performance')),
                        generation_detail_row('File', filename),
                    ])

                    return (
                        '<div class="generation-detail-panel">'
                        f'<div class="generation-detail-title">History #{image_id}</div>'
                        '<div class="generation-detail-grid">'
                        + ''.join(rows) +
                        '</div>'
                        '</div>'
                    )

                def format_history_batch(row):
                    prompt_preview = str(row.get('prompt', '') or '').replace('\n', ' ').strip()
                    if len(prompt_preview) > 80:
                        prompt_preview = prompt_preview[:77] + '...'
                    favorite = 'fav | ' if row.get('favorite') else ''
                    tags = str(row.get('tags') or '').strip()
                    tags_text = f'{tags} | ' if tags != '' else ''
                    return f"{row['id']} | {favorite}{tags_text}{row['created_at']} | {row['status']} | {row.get('generated_images', 0)}/{row.get('total_images') or '?'} | {prompt_preview}"

                def history_batch_choices(rows):
                    return ['All Images'] + [format_history_batch(row) for row in rows]

                def format_history_seed_stack(row):
                    prompt_preview = str(row.get('prompt', '') or '').replace('\n', ' ').strip()
                    if len(prompt_preview) > 70:
                        prompt_preview = prompt_preview[:67] + '...'
                    return (
                        f"{row.get('id')} | seed {row.get('seed')} | "
                        f"{row.get('image_count', 0)} images | "
                        f"{row.get('checkpoint_count', 0)} ckpt | {row.get('lora_count', 0)} LoRA | {prompt_preview}"
                    )

                def seed_stack_prompt_by_choice(choice):
                    stack_id = parse_history_id(choice)
                    if stack_id is None:
                        return ''
                    _, prompt = modules.history_db.get_seed_stack_key(stack_id)
                    return prompt

                def normalize_thumbnail_visibility(value):
                    mode = str(value or 'Visible').strip().casefold()
                    if mode == 'hidden':
                        return 'hidden'
                    if mode == 'all':
                        return 'all'
                    return 'visible'

                def history_row_is_preview(row):
                    return bool(row.get('is_preview', False))

                def seed_stack_choices(search, favorite_only, review_status, tag, days, checkpoints, loras,
                                       show_preview_images=False, thumbnail_visibility='visible'):
                    rows = modules.history_db.list_seed_stacks(
                        search=search,
                        favorite_only=favorite_only,
                        review_status=review_status,
                        tag=tag,
                        days=days,
                        checkpoints=checkpoints,
                        loras=loras,
                        show_preview_images=show_preview_images,
                        thumbnail_visibility=normalize_thumbnail_visibility(thumbnail_visibility)
                    )
                    choices = [format_history_seed_stack(row) for row in rows]
                    prompt_by_choice = {
                        format_history_seed_stack(row): str(row.get('prompt', '') or '')
                        for row in rows
                    }
                    return choices, prompt_by_choice

                def effective_history_days(days):
                    days = [str(day) for day in (days or []) if str(day or '').strip() != '']
                    return [] if '__all__' in days else days

                def format_history_image(row):
                    missing = '' if row.get('file_exists') else 'missing | '
                    favorite = 'fav | ' if row.get('favorite') else ''
                    hidden = 'hidden | ' if row.get('thumbnail_hidden') else ''
                    preview = 'preview | ' if history_row_is_preview(row) else ''
                    tags = str(row.get('tags') or '').strip()
                    tags_text = f'{tags} | ' if tags != '' else ''
                    seed = row.get('seed')
                    seed_text = f'seed {seed}' if seed is not None else 'seed ?'
                    return f"{row['id']} | {missing}{hidden}{favorite}{preview}{tags_text}{row.get('filename', '')} | {seed_text} | {row.get('checkpoint', '')}"

                def format_history_comparison(rows):
                    table = []
                    for row in rows:
                        table.append([
                            row.get('checkpoint', '') or '',
                            str(row.get('seed')) if row.get('seed') is not None else '',
                            row.get('testing_lora', '') or '',
                            str(row.get('id', '') or ''),
                            row.get('filename', '') or '',
                            'yes' if row.get('favorite') else '',
                            row.get('review_status', '') or '',
                            row.get('tags', '') or ''
                        ])
                    return table

                def format_history_gallery_items(rows):
                    gallery_items = []
                    for row in rows:
                        if not row.get('file_exists') or not os.path.exists(row['path']):
                            continue
                        seed = row.get('seed')
                        seed_text = f"seed {seed}" if seed is not None else 'seed ?'
                        hidden_text = 'hidden | ' if row.get('thumbnail_hidden') else ''
                        favorite_text = 'fav | ' if row.get('favorite') else ''
                        preview_text = 'preview | ' if history_row_is_preview(row) else ''
                        label = f"{hidden_text}{favorite_text}{preview_text}#{row.get('id')} | {seed_text}"
                        checkpoint = str(row.get('checkpoint') or '').strip()
                        if checkpoint != '':
                            label += f" | {checkpoint}"
                        gallery_items.append((row['path'], label))
                    return gallery_items

                def history_stack_thumbnail_path(rows):
                    paths = [row['path'] for row in rows if row.get('file_exists') and os.path.exists(row.get('path', ''))]
                    if len(paths) == 0:
                        return None
                    if len(paths) == 1:
                        return paths[0]

                    signature_parts = []
                    for path in paths[:4]:
                        try:
                            signature_parts.append(f'{path}:{os.path.getmtime(path)}')
                        except Exception:
                            signature_parts.append(path)
                    cache_key = hashlib.sha1(('minimal-v2|' + '|'.join(signature_parts)).encode('utf-8')).hexdigest()
                    cache_dir = os.path.join(modules.config.path_outputs, 'history_stacks')
                    os.makedirs(cache_dir, exist_ok=True)
                    thumbnail_path = os.path.join(cache_dir, f'{cache_key}.png')
                    if os.path.exists(thumbnail_path):
                        return thumbnail_path

                    from PIL import Image, ImageDraw, ImageOps

                    canvas_size = 176
                    card_size = 138
                    border = 2
                    canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
                    offsets = [(24, 14), (18, 20), (12, 26)]

                    def make_placeholder(index):
                        colors = [
                            ((60, 72, 92), (126, 145, 166)),
                            ((73, 82, 105), (142, 128, 112)),
                        ]
                        start, end = colors[index % len(colors)]
                        image = Image.new('RGB', (card_size, card_size), start)
                        draw = ImageDraw.Draw(image)
                        for y in range(card_size):
                            ratio = y / max(1, card_size - 1)
                            color = tuple(int(start[channel] * (1 - ratio) + end[channel] * ratio) for channel in range(3))
                            draw.line([(0, y), (card_size, y)], fill=color)
                        draw.rectangle([12, card_size - 44, card_size - 16, card_size - 34], fill=(220, 226, 232))
                        draw.rectangle([18, card_size - 28, card_size - 48, card_size - 20], fill=(178, 190, 204))
                        return image

                    def make_card(thumb):
                        card = Image.new('RGBA', (card_size + border * 2, card_size + border * 2), (245, 248, 250, 255))
                        card.paste(thumb, (border, border))
                        draw = ImageDraw.Draw(card)
                        draw.rectangle([0, 0, card.width - 1, card.height - 1], outline=(218, 226, 234, 255), width=1)
                        return card

                    for depth in range(min(len(paths), 3) - 1, 0, -1):
                        card = make_card(make_placeholder(depth))
                        x, y = offsets[min(depth, len(offsets) - 1)]
                        shadow = Image.new('RGBA', card.size, (0, 0, 0, 40))
                        canvas.alpha_composite(shadow, (x + 4, y + 5))
                        canvas.alpha_composite(card, (x, y))

                    try:
                        with Image.open(paths[0]) as image:
                            thumb = ImageOps.fit(image.convert('RGB'), (card_size, card_size), method=Image.Resampling.LANCZOS)
                        card = make_card(thumb)
                        shadow = Image.new('RGBA', card.size, (0, 0, 0, 50))
                        x, y = offsets[0]
                        canvas.alpha_composite(shadow, (x + 4, y + 5))
                        canvas.alpha_composite(card, (x, y))
                    except Exception:
                        return None

                    canvas.save(thumbnail_path)
                    return thumbnail_path

                def format_grouped_history_gallery_items(rows):
                    gallery_items = []
                    visible_refs = []
                    grouped = {}
                    for row in rows:
                        seed = row.get('seed')
                        prompt_text = str(row.get('prompt') or '')
                        key = (seed, prompt_text) if seed is not None and prompt_text != '' else None
                        if key is not None:
                            grouped.setdefault(key, []).append(row)

                    emitted_groups = set()
                    shared_stack_count = 0
                    for row in rows:
                        if not row.get('file_exists') or not os.path.exists(row.get('path', '')):
                            continue
                        seed = row.get('seed')
                        prompt_text = str(row.get('prompt') or '')
                        key = (seed, prompt_text) if seed is not None and prompt_text != '' else None
                        group_rows = grouped.get(key, []) if key is not None else []
                        if len(group_rows) > 1:
                            if key in emitted_groups:
                                continue
                            emitted_groups.add(key)
                            thumbnail_path = history_stack_thumbnail_path(group_rows)
                            if thumbnail_path is None:
                                continue
                            grouped_row_ids = [
                                parsed_id
                                for parsed_id in (parse_history_id(group_row.get('id')) for group_row in group_rows)
                                if parsed_id is not None
                            ]
                            if len(grouped_row_ids) == 0:
                                continue
                            stack_id = min(grouped_row_ids)
                            label = f"stack:{stack_id} | seed {seed} | {len(group_rows)} images"
                            gallery_items.append((thumbnail_path, label))
                            visible_refs.append(f"stack:{stack_id}")
                            shared_stack_count += 1
                            continue

                        seed_text = f"seed {seed}" if seed is not None else 'seed ?'
                        hidden_text = 'hidden | ' if row.get('thumbnail_hidden') else ''
                        favorite_text = 'fav | ' if row.get('favorite') else ''
                        preview_text = 'preview | ' if history_row_is_preview(row) else ''
                        label = f"{hidden_text}{favorite_text}{preview_text}#{row.get('id')} | {seed_text}"
                        checkpoint = str(row.get('checkpoint') or '').strip()
                        if checkpoint != '':
                            label += f" | {checkpoint}"
                        gallery_items.append((row['path'], label))
                        visible_refs.append(row.get('id'))
                    return gallery_items, visible_refs, shared_stack_count

                def selected_history_gallery_items(image_ids):
                    image_ids = parse_history_id_list(image_ids)
                    history_debug('selected_history_gallery_items', 'count=', len(image_ids), 'ids=', image_ids)
                    gallery_items = []
                    for image_id in image_ids:
                        summary = modules.history_db.get_image_summary(image_id)
                        if len(summary) == 0:
                            history_debug('selected_history_gallery_items skip', 'id=', image_id, 'reason=missing_summary')
                            continue
                        image_path = summary.get('path')
                        if not isinstance(image_path, str) or image_path == '':
                            history_debug('selected_history_gallery_items skip', 'id=', image_id, 'reason=missing_path')
                            continue
                        if not os.path.exists(image_path):
                            history_debug('selected_history_gallery_items skip', 'id=', image_id, 'reason=path_missing', 'path=', image_path)
                            continue
                        if not bool(summary.get('file_exists')):
                            history_debug('selected_history_gallery_items', 'id=', image_id, 'file_exists_flag=', summary.get('file_exists'),
                                          'path=', image_path)
                        seed = summary.get('seed')
                        seed_text = f"seed {seed}" if seed is not None else 'seed ?'
                        hidden_text = 'hidden | ' if summary.get('thumbnail_hidden') else ''
                        favorite_text = 'fav | ' if summary.get('favorite') else ''
                        preview_text = 'preview | ' if history_row_is_preview(summary) else ''
                        gallery_items.append((summary['path'], f"{hidden_text}{favorite_text}{preview_text}#{summary.get('id')} | {seed_text}"))
                    return gallery_items

                def visible_history_gallery_items(image_ids):
                    rows = []
                    for image_id in parse_history_id_list(image_ids):
                        summary = modules.history_db.get_image_summary(image_id)
                        if len(summary) > 0:
                            rows.append(summary)
                    return format_history_gallery_items(rows)

                def history_image_choices_from_ids(image_ids):
                    choices = []
                    for image_id in parse_history_id_list(image_ids):
                        summary = modules.history_db.get_image_summary(image_id)
                        if len(summary) > 0:
                            choices.append(format_history_image(summary))
                    return choices

                def history_selected_ids_json(image_ids):
                    try:
                        return json.dumps(parse_history_id_list(image_ids))
                    except Exception:
                        return '[]'

                def empty_image_curation(status):
                    return gr.update(), False, 0, '', '', '', status

                def empty_history_view(status):
                    return gr.update(choices=['All Images'], value='All Images'), gr.update(choices=[], value=[]), \
                        gr.update(choices=[], value=[]), gr.update(choices=[], value=[]), \
                        gr.update(choices=[], value=None), '', [], gr.update(value=[]), [], [], '[]', gr.update(value=[]), \
                        gr.update(choices=[], value=None), gr.update(value=[]), False, 0, '', '', '', \
                        False, 0, '', '', '', '', status

                def history_image_view(selection, search, favorite_only, review_status, tag, days,
                                       checkpoints=None, loras=None, show_preview_images=False,
                                       thumbnail_visibility='visible', status_prefix=''):
                    thumbnail_visibility = normalize_thumbnail_visibility(thumbnail_visibility)
                    days = effective_history_days(days)
                    batch_id = parse_history_id(selection)
                    is_all_images = batch_id is None
                    batch_curation = {} if is_all_images else modules.history_db.get_batch_curation(batch_id)
                    if is_all_images:
                        rows = modules.history_db.list_images(
                            search=search,
                            favorite_only=favorite_only,
                            review_status=review_status,
                            tag=tag,
                            days=days,
                            checkpoints=checkpoints,
                            loras=loras,
                            show_preview_images=show_preview_images,
                            thumbnail_visibility=thumbnail_visibility
                        )
                        comparison_rows = []
                    else:
                        rows = modules.history_db.list_batch_images(
                            batch_id,
                            favorite_only=favorite_only,
                            review_status=review_status,
                            tag=tag,
                            show_preview_images=show_preview_images,
                            thumbnail_visibility=thumbnail_visibility
                        )
                        comparison_rows = modules.history_db.list_batch_comparison_rows(batch_id)
                    gallery_items = format_history_gallery_items(rows)
                    visible_image_ids = []
                    reason_counts = {'missing_id': 0, 'missing_path': 0, 'missing_on_disk': 0, 'not_marked_exists': 0}
                    for row in rows:
                        parsed_id = parse_history_id(row.get('id'))
                        if parsed_id is None:
                            reason_counts['missing_id'] += 1
                            continue
                        image_path = row.get('path', '')
                        if not image_path:
                            reason_counts['missing_path'] += 1
                            continue
                        if not bool(row.get('file_exists')):
                            reason_counts['not_marked_exists'] += 1
                            continue
                        if row.get('file_exists') and os.path.exists(row.get('path', '')):
                            visible_image_ids.append(parsed_id)
                            continue
                        reason_counts['missing_on_disk'] += 1
                    if len(visible_image_ids) == 0 and len(rows) > 0:
                        visible_image_ids = [parse_history_id(row.get('id')) for row in rows if parse_history_id(row.get('id')) is not None]
                        history_debug(
                            'history_image_view fallback_visible_ids',
                            'rows=', len(rows),
                            'reason_counts=', reason_counts,
                            'visible=', len(visible_image_ids)
                        )
                    elif len(visible_image_ids) > 0 and len(rows) > 0:
                        history_debug('history_image_view visible_ids', 'rows=', len(rows), 'reason_counts=', reason_counts)
                    image_choices = [format_history_image(row) for row in rows]
                    image_id = visible_image_ids[0] if len(visible_image_ids) > 0 else None
                    value = None
                    if image_id is not None:
                        image_summary = modules.history_db.get_image_summary(image_id)
                        if len(image_summary) > 0:
                            value = format_history_image(image_summary)
                    selected_image_ids = [image_id] if image_id is not None else []
                    history_debug('history_image_view', 'selection=', selection, 'rows=', len(rows), 'visible=', len(visible_image_ids),
                                 'preselect=', image_id)
                    curation = modules.history_db.get_image_curation(image_id) if image_id is not None else {}
                    missing_count = len([row for row in rows if not row.get('file_exists')])
                    scope = 'all output images' if is_all_images else 'batch images'
                    status = status_prefix + f'Loaded {len(rows)} {scope}.'
                    if missing_count > 0:
                        status += f' {missing_count} file(s) are missing on disk.'
                    return gr.update(value=gallery_items), visible_image_ids, selected_image_ids, \
                        history_selected_ids_json(selected_image_ids), \
                        gr.update(value=selected_history_gallery_items(selected_image_ids)), \
                        gr.update(choices=image_choices, value=value), gr.update(value=format_history_comparison(comparison_rows)), \
                        bool(batch_curation.get('favorite', False)), int(batch_curation.get('rating', 0) or 0), \
                        batch_curation.get('review_status', ''), batch_curation.get('tags', ''), \
                        batch_curation.get('note', ''), \
                        bool(curation.get('favorite', False)), int(curation.get('rating', 0) or 0), \
                        curation.get('review_status', ''), curation.get('tags', ''), curation.get('note', ''), \
                        format_history_image_details(image_id), status

                def history_seed_group_view(seed_stack_selection, seed_stack_prompt, search, favorite_only,
                                            review_status, tag, days, checkpoints, loras, show_preview_images=False,
                                            thumbnail_visibility='visible'):
                    history_debug('history_seed_group_view', 'selection=', seed_stack_selection, 'prompt=', seed_stack_prompt, 'days=', days)
                    thumbnail_visibility = normalize_thumbnail_visibility(thumbnail_visibility)
                    days = effective_history_days(days)
                    stack_id = parse_history_id(seed_stack_selection)
                    if stack_id is None:
                        return [], history_selected_ids_json([]), \
                            gr.update(value=selected_history_gallery_items([])), \
                            gr.update(choices=[], value=None), \
                            False, 0, '', '', '', \
                            format_history_image_details(None), 'Select a seed group.'
                    seed, prompt = modules.history_db.get_seed_stack_key(stack_id)
                    rows = modules.history_db.list_seed_stack_images(
                        seed,
                        prompt,
                        search=search,
                        favorite_only=favorite_only,
                        review_status=review_status,
                        tag=tag,
                        days=days,
                        checkpoints=checkpoints,
                        loras=loras,
                        show_preview_images=show_preview_images,
                        thumbnail_visibility=thumbnail_visibility
                    )
                    selected_image_ids = []
                    for row in rows:
                        parsed_id = parse_history_id(row.get('id'))
                        if parsed_id is None:
                            continue
                        if row.get('file_exists') and os.path.exists(row['path']):
                            selected_image_ids.append(parsed_id)
                    if len(selected_image_ids) == 0 and len(rows) > 0:
                        selected_image_ids = [
                            parse_history_id(row.get('id'))
                            for row in rows
                            if parse_history_id(row.get('id')) is not None
                        ]
                    image_choices = [format_history_image(row) for row in rows]
                    value = image_choices[0] if len(image_choices) > 0 else None
                    image_id = parse_history_id(value)
                    curation = modules.history_db.get_image_curation(image_id) if image_id is not None else {}
                    status = f'Loaded {len(rows)} image(s) for seed group.'
                    if stack_id is None:
                        status = 'Select a seed group.'
                    return selected_image_ids, history_selected_ids_json(selected_image_ids), \
                        gr.update(value=selected_history_gallery_items(selected_image_ids)), \
                        gr.update(choices=image_choices, value=value), \
                        bool(curation.get('favorite', False)), int(curation.get('rating', 0) or 0), \
                        curation.get('review_status', ''), curation.get('tags', ''), curation.get('note', ''), \
                        format_history_image_details(image_id), status

                def history_seed_stack_gallery_view(search, favorite_only, review_status, tag, days, checkpoints, loras,
                                                    show_preview_images=False, thumbnail_visibility='visible',
                                                    status_prefix=''):
                    history_debug('history_seed_stack_gallery_view', 'preview=', show_preview_images, 'grouping=', 'on')
                    thumbnail_visibility = normalize_thumbnail_visibility(thumbnail_visibility)
                    days = effective_history_days(days)
                    stacks = modules.history_db.list_seed_stacks(
                        search=search,
                        favorite_only=favorite_only,
                        review_status=review_status,
                        tag=tag,
                        days=days,
                        checkpoints=checkpoints,
                        loras=loras,
                        show_preview_images=show_preview_images,
                        thumbnail_visibility=thumbnail_visibility
                    )
                    rows = modules.history_db.list_images(
                        search=search,
                        favorite_only=favorite_only,
                        review_status=review_status,
                        tag=tag,
                        days=days,
                        checkpoints=checkpoints,
                        loras=loras,
                        show_preview_images=show_preview_images,
                        thumbnail_visibility=thumbnail_visibility
                    )
                    gallery_items, visible_refs, shared_stack_count = format_grouped_history_gallery_items(rows)
                    selected_image_ids = []
                    image_choices = []
                    value = None
                    curation = {}
                    first_ref = visible_refs[0] if len(visible_refs) > 0 else None
                    if isinstance(first_ref, str) and first_ref.startswith('stack:'):
                        stack_id = parse_history_id(first_ref.replace('stack:', '', 1))
                        if stack_id is None:
                            first_ref = None
                        else:
                            seed, prompt = modules.history_db.get_seed_stack_key(stack_id)
                            selected_rows = modules.history_db.list_seed_stack_images(
                                seed,
                                prompt,
                                search=search,
                                favorite_only=favorite_only,
                                review_status=review_status,
                                tag=tag,
                                days=days,
                                checkpoints=checkpoints,
                                loras=loras,
                                show_preview_images=show_preview_images,
                                thumbnail_visibility=thumbnail_visibility
                            )
                            selected_image_ids = []
                            for row in selected_rows:
                                parsed_id = parse_history_id(row.get('id'))
                                if parsed_id is None:
                                    continue
                                if row.get('file_exists') and os.path.exists(row.get('path', '')):
                                    selected_image_ids.append(parsed_id)
                            image_choices = [format_history_image(row) for row in selected_rows]
                            if len(selected_image_ids) == 0 and len(selected_rows) > 0:
                                selected_image_ids = [
                                    parse_history_id(row.get('id'))
                                    for row in selected_rows
                                    if parse_history_id(row.get('id')) is not None
                                ]
                    elif first_ref is not None:
                        image_id = parse_history_id(first_ref)
                        summary = modules.history_db.get_image_summary(image_id) if image_id is not None else {}
                        if len(summary) > 0:
                            parsed_id = parse_history_id(summary.get('id'))
                            if parsed_id is not None:
                                selected_image_ids = [parsed_id]
                            image_choices = [format_history_image(summary)]

                    if len(image_choices) > 0:
                        value = image_choices[0] if len(image_choices) > 0 else None
                        image_id = parse_history_id(value)
                        curation = modules.history_db.get_image_curation(image_id) if image_id is not None else {}

                    status = status_prefix + f'Loaded {len(gallery_items)} grouped thumbnail(s). {shared_stack_count} shared seed stack(s).'
                    return gr.update(value=gallery_items), visible_refs, selected_image_ids, \
                        history_selected_ids_json(selected_image_ids), \
                        gr.update(value=selected_history_gallery_items(selected_image_ids)), \
                        gr.update(choices=image_choices, value=value), gr.update(value=[]), \
                        False, 0, '', '', '', \
                        bool(curation.get('favorite', False)), int(curation.get('rating', 0) or 0), \
                        curation.get('review_status', ''), curation.get('tags', ''), curation.get('note', ''), \
                        format_history_image_details(parse_history_id(value)), status

                def default_history_days(days):
                    return days[:1] if len(days) > 0 else []

                def format_history_day_label(day, counts):
                    try:
                        parsed = datetime.strptime(str(day), '%Y-%m-%d').date()
                        today = date.today()
                        if parsed == today:
                            label = 'Today'
                        elif parsed == today - timedelta(days=1):
                            label = 'Yesterday'
                        elif parsed.year == today.year:
                            label = parsed.strftime('%b %d')
                        else:
                            label = parsed.strftime('%b %d, %Y')
                    except Exception:
                        label = str(day)
                    count = int(counts.get(day, 0) or 0)
                    return f'{label} ({count})'

                def history_day_choices(days):
                    counts = modules.history_db.list_output_day_counts()
                    sorted_days = sorted(days, key=history_day_sort_key, reverse=True)
                    total_count = sum(int(count or 0) for count in counts.values())
                    return [(f'All Days ({total_count})', '__all__')] + [
                        (format_history_day_label(day, counts), day) for day in sorted_days
                    ]

                def history_day_sort_key(day):
                    try:
                        return datetime.strptime(str(day), '%Y-%m-%d').date()
                    except Exception:
                        return date.min

                def normalize_history_days(selected_days, previous_days, selection_mode):
                    all_days = modules.history_db.list_output_days()
                    all_choices = ['__all__'] + all_days
                    selected_days = [str(day) for day in (selected_days or []) if str(day or '') != '']
                    previous_days = [str(day) for day in (previous_days or []) if str(day or '') != '']
                    selection_mode = str(selection_mode or 'single').strip().casefold()
                    selected_set = set(selected_days)
                    previous_set = set(previous_days)
                    added = [day for day in all_choices if day in selected_set and day not in previous_set]
                    removed = [day for day in all_choices if day in previous_set and day not in selected_set]
                    clicked = (added + removed)[0] if len(added + removed) > 0 else (selected_days[0] if len(selected_days) > 0 else None)
                    if clicked is None:
                        return []
                    if clicked == '__all__':
                        return ['__all__'] if '__all__' in selected_set else []
                    selected_days = [day for day in selected_days if day != '__all__']
                    previous_days = [day for day in previous_days if day != '__all__']
                    if selection_mode == 'ctrl':
                        normalized = selected_days
                    elif selection_mode == 'shift' and len(previous_days) > 0:
                        anchor = previous_days[-1]
                        try:
                            anchor_index = all_days.index(anchor)
                            clicked_index = all_days.index(clicked)
                            start = min(anchor_index, clicked_index)
                            end = max(anchor_index, clicked_index)
                            normalized = all_days[start:end + 1]
                        except Exception:
                            normalized = [clicked]
                    else:
                        normalized = [clicked]
                    return [day for day in all_days if day in set(normalized)]

                def refresh_history(search, favorite_only, review_status, tag, checkpoints, loras,
                                    show_preview_images=False, group_by_seed=False, thumbnail_visibility='Visible',
                                    selected_days_state=None):
                    history_debug('refresh_history', 'group_by_seed=', group_by_seed, 'preview=', show_preview_images,
                                 'visibility=', thumbnail_visibility, 'review=', review_status)
                    thumbnail_visibility = normalize_thumbnail_visibility(thumbnail_visibility)
                    days = modules.history_db.list_output_days()
                    filter_values = modules.history_db.list_filter_values()
                    batches = modules.history_db.list_batches(
                        search=search,
                        favorite_only=favorite_only,
                        review_status=review_status,
                        tag=tag
                    )
                    choices = history_batch_choices(batches)
                    if len(days) == 0 and len(choices) <= 1:
                        return empty_history_view('No history batches found.')
                    value = 'All Images'
                    selected_days = [str(day) for day in (selected_days_state or []) if str(day or '') != '']
                    valid_day_values = set(days + ['__all__'])
                    selected_days = [day for day in selected_days if day in valid_day_values]
                    if len(selected_days) == 0:
                        selected_days = default_history_days(days)
                    stack_choices, prompt_by_stack = seed_stack_choices(
                        search, favorite_only, review_status, tag, selected_days, checkpoints, loras,
                        show_preview_images=show_preview_images,
                        thumbnail_visibility=thumbnail_visibility
                    )
                    stack_value = stack_choices[0] if len(stack_choices) > 0 else None
                    stack_prompt = prompt_by_stack.get(stack_value, '') if stack_value else ''
                    if group_by_seed:
                        image_outputs = history_seed_stack_gallery_view(
                            search, favorite_only, review_status, tag, selected_days, checkpoints, loras,
                            show_preview_images=show_preview_images,
                            thumbnail_visibility=thumbnail_visibility,
                            status_prefix=f'Loaded {max(0, len(choices) - 1)} batch(es). '
                        )
                    else:
                        image_outputs = history_image_view(value, search, favorite_only, review_status, tag,
                                                           selected_days, checkpoints, loras,
                                                           show_preview_images=show_preview_images,
                                                           thumbnail_visibility=thumbnail_visibility,
                                                           status_prefix=f'Loaded {max(0, len(choices) - 1)} batch(es). ')
                    return (gr.update(choices=choices, value=value),
                            gr.update(choices=history_day_choices(days), value=selected_days),
                            gr.update(choices=filter_values['checkpoints'], value=checkpoints),
                            gr.update(choices=filter_values['loras'], value=loras),
                            gr.update(choices=stack_choices, value=stack_value), stack_prompt,
                            selected_days) + image_outputs

                def requery_history_outputs(search, favorite_only, review_status, tag, checkpoints, loras,
                                            show_preview_images=False, group_by_seed=False, thumbnail_visibility='Visible',
                                            selected_days_state=None):
                    history_debug('requery_history_outputs', 'group_by_seed=', group_by_seed, 'preview=', show_preview_images,
                                 'visibility=', thumbnail_visibility, 'review=', review_status)
                    thumbnail_visibility = normalize_thumbnail_visibility(thumbnail_visibility)
                    result = modules.history_db.reconcile_outputs_folder()
                    days = modules.history_db.list_output_days()
                    filter_values = modules.history_db.list_filter_values()
                    batches = modules.history_db.list_batches(
                        search=search,
                        favorite_only=favorite_only,
                        review_status=review_status,
                        tag=tag
                    )
                    choices = history_batch_choices(batches)
                    status = (
                        f"Re-query complete. Added {result['added']}, removed {result['removed']}, "
                        f"updated metadata {result.get('updated', 0)}, "
                        f"unchanged {result['unchanged']}, imported batches {result['imported_batches']}, "
                        f"removed batches {result['removed_batches']}, skipped {result['skipped']}, failed {result['failed']}."
                    )
                    if len(days) == 0 and len(choices) <= 1:
                        return empty_history_view(status + ' No history batches found.')
                    value = 'All Images'
                    selected_days = [str(day) for day in (selected_days_state or []) if str(day or '') != '']
                    valid_day_values = set(days + ['__all__'])
                    selected_days = [day for day in selected_days if day in valid_day_values]
                    if len(selected_days) == 0:
                        selected_days = default_history_days(days)
                    stack_choices, prompt_by_stack = seed_stack_choices(
                        search, favorite_only, review_status, tag, selected_days, checkpoints, loras,
                        show_preview_images=show_preview_images,
                        thumbnail_visibility=thumbnail_visibility
                    )
                    stack_value = stack_choices[0] if len(stack_choices) > 0 else None
                    stack_prompt = prompt_by_stack.get(stack_value, '') if stack_value else ''
                    if group_by_seed:
                        image_outputs = history_seed_stack_gallery_view(
                            search, favorite_only, review_status, tag, selected_days, checkpoints, loras,
                            show_preview_images=show_preview_images,
                            thumbnail_visibility=thumbnail_visibility,
                            status_prefix=status + ' '
                        )
                    else:
                        image_outputs = history_image_view(value, search, favorite_only, review_status, tag,
                                                           selected_days, checkpoints, loras,
                                                           show_preview_images=show_preview_images,
                                                           thumbnail_visibility=thumbnail_visibility,
                                                           status_prefix=status + ' ')
                    return (gr.update(choices=choices, value=value),
                            gr.update(choices=history_day_choices(days), value=selected_days),
                            gr.update(choices=filter_values['checkpoints'], value=checkpoints),
                            gr.update(choices=filter_values['loras'], value=loras),
                            gr.update(choices=stack_choices, value=stack_value), stack_prompt,
                            selected_days) + image_outputs

                def load_history_batch(selection, search, favorite_only, review_status, tag, days, checkpoints, loras,
                                       show_preview_images=False, thumbnail_visibility='Visible'):
                    return history_image_view(selection, search, favorite_only, review_status, tag, days,
                                              checkpoints, loras, show_preview_images=show_preview_images,
                                              thumbnail_visibility=thumbnail_visibility)

                def load_history_days(selection, previous_days, selection_mode, batch_selection, search,
                                      favorite_only, review_status, tag, checkpoints, loras,
                                      show_preview_images=False, group_by_seed=False, thumbnail_visibility='Visible'):
                    thumbnail_visibility = normalize_thumbnail_visibility(thumbnail_visibility)
                    selected_days = normalize_history_days(selection, previous_days, selection_mode)
                    stack_choices, prompt_by_stack = seed_stack_choices(
                        search, favorite_only, review_status, tag, selected_days, checkpoints, loras,
                        show_preview_images=show_preview_images,
                        thumbnail_visibility=thumbnail_visibility
                    )
                    stack_value = stack_choices[0] if len(stack_choices) > 0 else None
                    stack_prompt = prompt_by_stack.get(stack_value, '') if stack_value else ''
                    if group_by_seed:
                        image_outputs = history_seed_stack_gallery_view(
                            search, favorite_only, review_status, tag, selected_days, checkpoints, loras,
                            show_preview_images=show_preview_images,
                            thumbnail_visibility=thumbnail_visibility
                        )
                    else:
                        image_outputs = history_image_view(batch_selection, search, favorite_only, review_status, tag,
                                                           selected_days, checkpoints, loras,
                                                           show_preview_images=show_preview_images,
                                                           thumbnail_visibility=thumbnail_visibility)
                    return (gr.update(value=selected_days), selected_days,
                            gr.update(choices=stack_choices, value=stack_value), stack_prompt) + image_outputs

                def load_history_batch_curation(selection):
                    batch_id = parse_history_id(selection)
                    if batch_id is None:
                        return False, 0, '', '', '', 'Select a history batch.'
                    curation = modules.history_db.get_batch_curation(batch_id)
                    if len(curation) == 0:
                        return False, 0, '', '', '', 'History batch was not found.'
                    return bool(curation.get('favorite', False)), int(curation.get('rating', 0) or 0), \
                        curation.get('review_status', ''), curation.get('tags', ''), curation.get('note', ''), \
                        f'Loaded curation for history batch #{batch_id}.'

                def load_history_image_curation(selection):
                    history_debug('load_history_image_curation', 'selection=', selection)
                    image_id = parse_history_id(selection)
                    if image_id is None:
                        return False, 0, '', '', '', '', 'Select a history image.'
                    curation = modules.history_db.get_image_curation(image_id)
                    if len(curation) == 0:
                        return False, 0, '', '', '', '', 'History image was not found.'
                    return bool(curation.get('favorite', False)), int(curation.get('rating', 0) or 0), \
                        curation.get('review_status', ''), curation.get('tags', ''), curation.get('note', ''), \
                        format_history_image_details(image_id), f'Loaded curation for history image #{image_id}.'

                def select_history_selected_gallery_image(selected_image_ids, evt: gr.SelectData):
                    if evt is None or not hasattr(evt, 'index') or evt.index is None:
                        return gr.update(), False, 0, '', '', '', '', 'Select a large history image.'
                    try:
                        index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                        index = int(index)
                    except Exception:
                        return gr.update(), False, 0, '', '', '', '', 'Select a large history image.'

                    image_ids = parse_history_id_list(selected_image_ids)
                    if index < 0 or index >= len(image_ids):
                        return gr.update(), False, 0, '', '', '', '', 'Selected large history image was not found.'

                    image_id = image_ids[index]
                    summary = modules.history_db.get_image_summary(image_id)
                    if len(summary) == 0:
                        return gr.update(), False, 0, '', '', '', format_history_image_details(None), \
                            f'Selected large history image #{image_id} was not found.'

                    selection_value = format_history_image(summary)
                    curation = modules.history_db.get_image_curation(image_id)
                    if len(curation) == 0:
                        return selection_value, False, 0, '', '', '', format_history_image_details(image_id), \
                            'History image was not found.'

                    return selection_value, bool(curation.get('favorite', False)), int(curation.get('rating', 0) or 0), \
                        curation.get('review_status', ''), curation.get('tags', ''), curation.get('note', ''), \
                        format_history_image_details(image_id), f'Selected history image #{image_id} from large preview.'

                def select_history_thumbnail(visible_image_ids, selected_image_ids, selection_mode, search,
                                             favorite_only, review_status, tag, days, checkpoints, loras,
                                             show_preview_images=False, thumbnail_visibility='Visible',
                                             evt: gr.SelectData = None):
                    thumbnail_visibility = normalize_thumbnail_visibility(thumbnail_visibility)
                    days = effective_history_days(days)
                    try:
                        index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                        clicked_index = int(index)
                        image_ref = (visible_image_ids or [])[clicked_index]
                    except Exception:
                        return [], '[]', gr.update(value=[]), gr.update(), False, 0, '', '', '', '', 'Select a thumbnail.'

                    if isinstance(image_ref, str) and image_ref.startswith('stack:'):
                        stack_id = parse_history_id(image_ref.replace('stack:', '', 1))
                        seed, prompt = modules.history_db.get_seed_stack_key(stack_id)
                        rows = modules.history_db.list_seed_stack_images(
                            seed,
                            prompt,
                            search=search,
                            favorite_only=favorite_only,
                            review_status=review_status,
                            tag=tag,
                            days=days,
                            checkpoints=checkpoints,
                            loras=loras,
                            show_preview_images=show_preview_images,
                            thumbnail_visibility=thumbnail_visibility
                        )
                        selected = [
                            parsed_id
                            for parsed_id in (
                                parse_history_id(row.get('id')) for row in rows
                                if row.get('file_exists') and os.path.exists(row.get('path', ''))
                            )
                            if parsed_id is not None
                        ]
                        image_choices = [format_history_image(row) for row in rows]
                        value = image_choices[0] if len(image_choices) > 0 else None
                        image_id = parse_history_id(value)
                        curation = modules.history_db.get_image_curation(image_id) if image_id is not None else {}
                        return selected, history_selected_ids_json(selected), \
                            gr.update(value=selected_history_gallery_items(selected)), \
                            gr.update(choices=image_choices, value=value), \
                            bool(curation.get('favorite', False)), int(curation.get('rating', 0) or 0), \
                            curation.get('review_status', ''), curation.get('tags', ''), curation.get('note', ''), \
                            format_history_image_details(image_id), \
                            f'Loaded {len(selected)} image(s) from seed stack #{stack_id}.'

                    image_id = parse_history_id(image_ref)
                    if image_id is None:
                        return [], '[]', gr.update(value=[]), gr.update(), False, 0, '', '', '', '', 'Select a thumbnail.'

                    selected = parse_history_id_list(selected_image_ids)
                    selection_mode = str(selection_mode or 'single').strip().casefold()
                    if selection_mode == 'ctrl':
                        if image_id in selected:
                            selected = [x for x in selected if x != image_id]
                        else:
                            selected.append(image_id)
                    elif selection_mode == 'shift' and len(selected) > 0:
                        visible = []
                        for visible_ref in visible_image_ids or []:
                            parsed_visible = parse_history_id(visible_ref)
                            if parsed_visible is not None:
                                visible.append(parsed_visible)
                        anchor = selected[-1]
                        try:
                            anchor_index = visible.index(anchor)
                            clicked_visible_index = visible.index(image_id)
                            start = min(anchor_index, clicked_visible_index)
                            end = max(anchor_index, clicked_visible_index)
                            selected = visible[start:end + 1]
                        except Exception:
                            selected = [image_id]
                    else:
                        selected = [image_id]
                    history_debug(
                        'select_history_thumbnail',
                        'mode=', selection_mode,
                        'image_ref=', image_ref,
                        'before=', parse_history_id_list(selected_image_ids),
                        'after=', selected
                    )

                    summary = modules.history_db.get_image_summary(image_id)
                    curation = modules.history_db.get_image_curation(image_id)
                    if len(summary) == 0 or len(curation) == 0:
                        return selected, history_selected_ids_json(selected), \
                            gr.update(value=selected_history_gallery_items(selected)), gr.update(), \
                            False, 0, '', '', '', '', 'History image was not found.'

                    status = f'Selected {len(selected)} image(s). Use Ctrl-click to add/remove one or Shift-click to select a range.'
                    return selected, history_selected_ids_json(selected), \
                        gr.update(value=selected_history_gallery_items(selected)), \
                        gr.update(value=format_history_image(summary)), \
                        bool(curation.get('favorite', False)), int(curation.get('rating', 0) or 0), \
                        curation.get('review_status', ''), curation.get('tags', ''), curation.get('note', ''), \
                        format_history_image_details(image_id), status

                def select_history_thumbnail_by_ref(image_ref, visible_image_ids, selected_image_ids, selection_mode,
                                                    search, favorite_only, review_status, tag, days, checkpoints, loras,
                                                    show_preview_images=False, thumbnail_visibility='Visible'):
                    image_ref = str(image_ref or '').strip()
                    if image_ref == '':
                        return [], '[]', gr.update(value=[]), gr.update(), False, 0, '', '', '', '', 'Select a thumbnail.'

                    raw_image_ref = image_ref
                    if '|' in image_ref:
                        parts = image_ref.split('|', 1)
                        image_ref = parts[0].strip()
                        raw_image_ref = parts[1].strip()

                    if isinstance(visible_image_ids, str):
                        try:
                            visible_image_ids = json.loads(visible_image_ids)
                        except Exception:
                            visible_image_ids = [v for v in (visible_image_ids.split(',') if isinstance(visible_image_ids, str) else []) if v.strip() != '']

                    def matches_visible_ref(visible_ref, target_ref):
                        if visible_ref is None or target_ref is None:
                            return False
                        visible_ref = str(visible_ref).strip()
                        target_ref = str(target_ref).strip()
                        if visible_ref == target_ref:
                            return True

                        visible_is_stack = visible_ref.lower().startswith('stack:')
                        target_is_stack = target_ref.lower().startswith('stack:')
                        if visible_is_stack != target_is_stack:
                            return False

                        try:
                            return parse_history_id(visible_ref) == parse_history_id(target_ref)
                        except Exception:
                            return False

                    history_debug(
                        'select_history_thumbnail_by_ref',
                        'incoming=', image_ref,
                        'raw=', raw_image_ref,
                        'selected=', selected_image_ids,
                        'visible=', len(visible_image_ids or []),
                        'mode=', selection_mode
                    )

                    clicked_index = None
                    if image_ref.lower().startswith('index:'):
                        try:
                            requested_index = int(image_ref[len('index:'):])
                        except Exception:
                            requested_index = None
                        if requested_index is not None and 0 <= requested_index < len(visible_image_ids or []):
                            clicked_index = requested_index
                            if raw_image_ref:
                                current_ref = (visible_image_ids or [])[clicked_index]
                                if not matches_visible_ref(current_ref, raw_image_ref):
                                    clicked_index = None
                    else:
                        for index, visible_ref in enumerate(visible_image_ids or []):
                            if str(visible_ref) == image_ref:
                                clicked_index = index
                                break
                            try:
                                if parse_history_id(visible_ref) == parse_history_id(image_ref):
                                    clicked_index = index
                                    break
                            except Exception:
                                pass
                    if clicked_index is None:
                        # Fallback to direct reference match (for cases where index changed while gallery was rerendered).
                        if raw_image_ref:
                            for index, visible_ref in enumerate(visible_image_ids or []):
                                if matches_visible_ref(visible_ref, raw_image_ref):
                                    clicked_index = index
                                    break
                    if clicked_index is None:
                        history_debug('select_history_thumbnail_by_ref', 'failed_resolve=', image_ref, raw_image_ref,
                                     'visible=', visible_image_ids)
                        return [], '[]', gr.update(value=[]), gr.update(), False, 0, '', '', '', '', 'Selected thumbnail is no longer visible.'
                    history_debug('select_history_thumbnail_by_ref', 'resolved_index=', clicked_index, 'ref=', raw_image_ref)

                    class ThumbnailSelectEvent:
                        def __init__(self, index):
                            self.index = index

                    return select_history_thumbnail(
                        visible_image_ids, selected_image_ids, selection_mode, search,
                        favorite_only, review_status, tag, days, checkpoints, loras,
                        show_preview_images=show_preview_images,
                        thumbnail_visibility=thumbnail_visibility,
                        evt=ThumbnailSelectEvent(clicked_index)
                    )

                def select_history_comparison(table_rows, evt: gr.SelectData):
                    try:
                        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                        row = table_rows[int(row_index)]
                        image_id = parse_history_id(row[3])
                    except Exception:
                        image_id = None
                    if image_id is None:
                        return [], '[]', gr.update(value=[]), gr.update(), False, 0, '', '', '', '', 'Select a comparison row with an image id.'
                    summary = modules.history_db.get_image_summary(image_id)
                    curation = modules.history_db.get_image_curation(image_id)
                    if len(summary) == 0 or len(curation) == 0:
                        return [], '[]', gr.update(value=[]), gr.update(), False, 0, '', '', '', '', 'History image was not found.'
                    selected = [image_id]
                    return selected, history_selected_ids_json(selected), \
                        gr.update(value=selected_history_gallery_items(selected)), \
                        gr.update(value=format_history_image(summary)), \
                        bool(curation.get('favorite', False)), int(curation.get('rating', 0) or 0), \
                        curation.get('review_status', ''), curation.get('tags', ''), curation.get('note', ''), \
                        format_history_image_details(image_id), \
                        f'Selected history image #{image_id} from comparison.'

                def remove_history_selected_image(selected_image_ids, remove_image_id):
                    remove_image_id = parse_history_id(remove_image_id)
                    selected = parse_history_id_list(selected_image_ids)
                    if remove_image_id is not None:
                        selected = [image_id for image_id in selected if image_id != remove_image_id]
                    status = f'Selected {len(selected)} image(s).'
                    return selected, history_selected_ids_json(selected), \
                        gr.update(value=selected_history_gallery_items(selected)), \
                        format_history_image_details(selected[0] if len(selected) > 0 else None), status

                def delete_history_selected_image(selected_image_ids, visible_image_ids, delete_image_id):
                    history_debug('delete_history_selected_image',
                                 'delete=', delete_image_id,
                                 'selected=', selected_image_ids,
                                 'visible=', visible_image_ids)
                    delete_image_id = parse_history_id(delete_image_id)
                    if delete_image_id is None:
                        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), 'Select an image to delete.'
                    deleted, path = modules.history_db.delete_image(delete_image_id, delete_file=True)
                    selected = [
                        image_id for image_id in parse_history_id_list(selected_image_ids)
                        if image_id != delete_image_id
                    ]
                    visible = [
                        image_id for image_id in parse_history_id_list(visible_image_ids)
                        if image_id != delete_image_id
                    ]
                    choices = history_image_choices_from_ids(visible)
                    next_choice = choices[0] if len(choices) > 0 else None
                    status = f"Deleted {os.path.basename(path)}." if deleted and path else 'Could not delete selected image.'
                    return gr.update(value=visible_history_gallery_items(visible)), visible, selected, \
                        history_selected_ids_json(selected), gr.update(value=selected_history_gallery_items(selected)), \
                        gr.update(choices=choices, value=next_choice), format_history_image_details(next_choice), status

                def toggle_history_image_favorite(image_id):
                    history_debug('toggle_history_image_favorite', 'image=', image_id)
                    image_id = parse_history_id(image_id)
                    if image_id is None:
                        return 'Select an image to favorite.'
                    curation = modules.history_db.get_image_curation(image_id)
                    if len(curation) == 0:
                        return 'History image was not found.'
                    next_favorite = not bool(curation.get('favorite', False))
                    modules.history_db.update_image_curation(
                        image_id,
                        next_favorite,
                        curation.get('rating', 0),
                        curation.get('review_status', ''),
                        curation.get('tags', ''),
                        curation.get('note', '')
                    )
                    return f"{'Favorited' if next_favorite else 'Unfavorited'} history image #{image_id}."

                def hide_history_thumbnail(image_id, search, favorite_only, review_status, tag, checkpoints, loras,
                                           selected_days_state,
                                           show_preview_images=False, group_by_seed=False,
                                           thumbnail_visibility='Visible'):
                    image_id = parse_history_id(image_id)
                    if image_id is None:
                        return refresh_history(
                            search, favorite_only, review_status, tag, checkpoints, loras,
                            show_preview_images, group_by_seed, thumbnail_visibility, selected_days_state
                        )
                    summary = modules.history_db.get_image_summary(image_id)
                    next_hidden = not bool(summary.get('thumbnail_hidden')) if len(summary) > 0 else True
                    saved = modules.history_db.set_image_thumbnail_hidden(image_id, next_hidden)
                    status_prefix = (
                        f"{'Hidden' if next_hidden else 'Restored'} history image #{image_id} "
                        f"{'from' if next_hidden else 'to'} thumbnails. "
                        if saved else
                        'History image was not found. '
                    )
                    outputs = refresh_history(
                        search, favorite_only, review_status, tag, checkpoints, loras,
                        show_preview_images, group_by_seed, thumbnail_visibility, selected_days_state
                    )
                    outputs = list(outputs)
                    outputs[-1] = status_prefix + str(outputs[-1] or '')
                    return tuple(outputs)

                def refresh_history_after_bulk_action(status_prefix, search, favorite_only, review_status, tag,
                                                      checkpoints, loras, show_preview_images=False,
                                                      group_by_seed=False, thumbnail_visibility='Visible',
                                                      selected_days_state=None):
                    outputs = list(refresh_history(
                        search, favorite_only, review_status, tag, checkpoints, loras,
                        show_preview_images, group_by_seed, thumbnail_visibility, selected_days_state
                    ))
                    outputs[-1] = status_prefix + str(outputs[-1] or '')
                    return tuple(outputs)

                def bulk_delete_history_images(selected_image_ids, search, favorite_only, review_status, tag,
                                               checkpoints, loras, show_preview_images=False,
                                               group_by_seed=False, thumbnail_visibility='Visible',
                                               selected_days_state=None):
                    image_ids = parse_history_id_list(selected_image_ids)
                    deleted_count = 0
                    failed_count = 0
                    for image_id in image_ids:
                        deleted, _ = modules.history_db.delete_image(image_id, delete_file=True)
                        if deleted:
                            deleted_count += 1
                        else:
                            failed_count += 1
                    if len(image_ids) == 0:
                        status = 'Select one or more thumbnails to delete. '
                    else:
                        status = f'Deleted {deleted_count} selected thumbnail image file(s) and history record(s). '
                        if failed_count > 0:
                            status += f'Could not delete {failed_count} image(s). '
                    return refresh_history_after_bulk_action(
                        status, search, favorite_only, review_status, tag, checkpoints, loras,
                        show_preview_images, group_by_seed, thumbnail_visibility, selected_days_state
                    )

                def bulk_favorite_history_images(selected_image_ids, search, favorite_only, review_status, tag,
                                                 checkpoints, loras, show_preview_images=False,
                                                 group_by_seed=False, thumbnail_visibility='Visible',
                                                 selected_days_state=None):
                    image_ids = parse_history_id_list(selected_image_ids)
                    favorite_count = 0
                    failed_count = 0
                    for image_id in image_ids:
                        curation = modules.history_db.get_image_curation(image_id)
                        if len(curation) == 0 or not modules.history_db.update_image_curation(
                                image_id, True, curation.get('rating', 0), curation.get('review_status', ''),
                                curation.get('tags', ''), curation.get('note', '')):
                            failed_count += 1
                            continue
                        favorite_count += 1
                    if len(image_ids) == 0:
                        status = 'Select one or more thumbnails to favorite. '
                    else:
                        status = f'Favorited {favorite_count} selected image(s). '
                        if failed_count > 0:
                            status += f'Could not favorite {failed_count} image(s). '
                    return refresh_history_after_bulk_action(
                        status, search, favorite_only, review_status, tag, checkpoints, loras,
                        show_preview_images, group_by_seed, thumbnail_visibility, selected_days_state
                    )

                def bulk_hide_history_thumbnails(selected_image_ids, search, favorite_only, review_status, tag,
                                                 checkpoints, loras, show_preview_images=False,
                                                 group_by_seed=False, thumbnail_visibility='Visible',
                                                 selected_days_state=None):
                    image_ids = parse_history_id_list(selected_image_ids)
                    hidden_count = 0
                    failed_count = 0
                    for image_id in image_ids:
                        if modules.history_db.set_image_thumbnail_hidden(image_id, True):
                            hidden_count += 1
                        else:
                            failed_count += 1
                    if len(image_ids) == 0:
                        status = 'Select one or more thumbnails to hide. '
                    else:
                        status = f'Hidden {hidden_count} selected image(s) from the thumbnail view. '
                        if failed_count > 0:
                            status += f'Could not hide {failed_count} image(s). '
                    return refresh_history_after_bulk_action(
                        status, search, favorite_only, review_status, tag, checkpoints, loras,
                        show_preview_images, group_by_seed, thumbnail_visibility, selected_days_state
                    )

                def load_history_image_config_by_id(image_id, current_prompt, is_generating, inpaint_mode):
                    image_id = parse_history_id(image_id)
                    if image_id is None:
                        return prompt_config_to_ui_updates({}, is_generating, inpaint_mode, 'Select a history image first.')
                    return load_history_image_config(str(image_id), 'Full Config', current_prompt, is_generating, inpaint_mode)

                def apply_history_seed_group(group_by_seed, seed_stack_selection, seed_stack_prompt, search,
                                             favorite_only, review_status, tag, days, checkpoints, loras,
                                             show_preview_images=False, thumbnail_visibility='Visible'):
                    if not group_by_seed:
                        return gr.update(), gr.update(), gr.update(), gr.update(), False, 0, '', '', '', gr.update(), 'Group By Seed is off.'
                    return history_seed_group_view(
                        seed_stack_selection,
                        seed_stack_prompt,
                        search,
                        favorite_only,
                        review_status,
                        tag,
                        days,
                        checkpoints,
                        loras,
                        show_preview_images=show_preview_images,
                        thumbnail_visibility=thumbnail_visibility
                    )

                def save_history_image_curation(selection, favorite, rating, review_status, tags, note, batch_selection,
                                                filter_favorite_only, filter_review_status, filter_tag):
                    image_id = parse_history_id(selection)
                    if image_id is None:
                        return gr.update(), gr.update(), gr.update(), 'Select a history image first.'
                    if not modules.history_db.update_image_curation(image_id, favorite, rating, review_status, tags, note):
                        return gr.update(), gr.update(), gr.update(), 'History image was not found.'
                    batch_id = parse_history_id(batch_selection)
                    if batch_id is None:
                        summary = modules.history_db.get_image_summary(image_id)
                        selected_update = gr.update(value=format_history_image(summary)) if len(summary) > 0 else gr.update()
                        return selected_update, \
                            gr.update(), gr.update(), f'Saved curation for history image #{image_id}.'
                    rows = modules.history_db.list_batch_images(
                        batch_id,
                        favorite_only=filter_favorite_only,
                        review_status=filter_review_status,
                        tag=filter_tag
                    )
                    comparison_rows = modules.history_db.list_batch_comparison_rows(batch_id)
                    image_choices = [format_history_image(row) for row in rows]
                    selected_value = next((choice for choice in image_choices if parse_history_id(choice) == image_id), None)
                    return gr.update(choices=image_choices, value=selected_value), gr.update(), \
                        gr.update(value=format_history_comparison(comparison_rows)), \
                        f'Saved curation for history image #{image_id}.'

                def save_history_batch_curation(selection, favorite, rating, review_status, tags, note,
                                                search, filter_favorite_only, filter_review_status, filter_tag):
                    batch_id = parse_history_id(selection)
                    if batch_id is None:
                        return gr.update(), 'Select a history batch first.'
                    if not modules.history_db.update_batch_curation(batch_id, favorite, rating, review_status, tags, note):
                        return gr.update(), 'History batch was not found.'
                    batches = modules.history_db.list_batches(
                        search=search,
                        favorite_only=filter_favorite_only,
                        review_status=filter_review_status,
                        tag=filter_tag
                    )
                    choices = [format_history_batch(row) for row in batches]
                    selected_value = next((choice for choice in choices if parse_history_id(choice) == batch_id), None)
                    return gr.update(choices=choices, value=selected_value), f'Saved curation for history batch #{batch_id}.'

                def load_history_image_config(selection, mode, current_prompt, is_generating, inpaint_mode):
                    image_id = parse_history_id(selection)
                    if image_id is None:
                        return prompt_config_to_ui_updates({}, is_generating, inpaint_mode, 'Select a history image first.')
                    config_data = modules.history_db.get_config_by_image_id(image_id)
                    if len(config_data) == 0:
                        return prompt_config_to_ui_updates({}, is_generating, inpaint_mode, 'History image has no saved config.')
                    if mode in ['Replace Prompt', 'Append Prompt']:
                        return prompt_only_config_to_ui_updates(
                            config_data,
                            current_prompt,
                            mode,
                            f'{mode} from history image #{image_id}.'
                        )
                    return prompt_config_to_ui_updates(config_data, is_generating, inpaint_mode,
                                                       f'Loaded config from history image #{image_id}.')

                def load_full_prompt_config(name, current_prompt, is_generating, inpaint_mode):
                    return load_prompt_config(name, 'Full Config', current_prompt, is_generating, inpaint_mode)

                def replace_prompt_from_config(name, current_prompt, is_generating, inpaint_mode):
                    return load_prompt_config(name, 'Replace Prompt', current_prompt, is_generating, inpaint_mode)

                def append_prompt_from_config(name, current_prompt, is_generating, inpaint_mode):
                    return load_prompt_config(name, 'Append Prompt', current_prompt, is_generating, inpaint_mode)

                def load_full_history_config(selection, current_prompt, is_generating, inpaint_mode):
                    return load_history_image_config(selection, 'Full Config', current_prompt, is_generating, inpaint_mode)

                def replace_prompt_from_history(selection, current_prompt, is_generating, inpaint_mode):
                    return load_history_image_config(selection, 'Replace Prompt', current_prompt, is_generating, inpaint_mode)

                def append_prompt_from_history(selection, current_prompt, is_generating, inpaint_mode):
                    return load_history_image_config(selection, 'Append Prompt', current_prompt, is_generating, inpaint_mode)

                def send_history_image_to_inpaint(selection):
                    image_id = parse_history_id(selection)
                    if image_id is None:
                        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), 'Select a history image first.'
                    summary = modules.history_db.get_image_summary(image_id)
                    image_path = summary.get('path') if len(summary) > 0 else None
                    if not image_path or not os.path.exists(image_path):
                        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), 'History image file was not found.'
                    return True, gr.update(visible=True), 'inpaint', gr.update(selected='inpaint_tab'), image_path, \
                        f'Sent {os.path.basename(image_path)} to Inpaint.'

                history_filter_inputs = [
                    history_search, history_filter_favorites, history_filter_status, history_filter_tag,
                    history_filter_checkpoints, history_filter_loras, history_show_preview_images,
                    history_stack_by_seed, history_thumbnail_visibility, history_selected_days
                ]
                history_refresh_outputs = [
                    history_batch_selection, history_day_selection, history_filter_checkpoints,
                    history_filter_loras, history_seed_stack_selection, history_seed_stack_prompt,
                    history_selected_days,
                    history_gallery, history_visible_image_ids,
                    history_selected_image_ids, history_selected_image_ids_json,
                    history_selected_gallery, history_image_selection,
                    history_comparison_table, history_batch_favorite, history_batch_rating,
                    history_batch_review_status, history_batch_tags, history_batch_note,
                    history_favorite, history_rating, history_review_status, history_tags,
                    history_note, history_image_details, history_status
                ]

                shared.gradio_root.load(refresh_history, inputs=history_filter_inputs,
                                        outputs=history_refresh_outputs,
                                        queue=False, show_progress=False)
                history_refresh_button.click(refresh_history, inputs=history_filter_inputs,
                                             outputs=history_refresh_outputs,
                                             queue=False, show_progress=False)
                history_requery_button.click(requery_history_outputs, inputs=history_filter_inputs,
                                             outputs=history_refresh_outputs,
                                             queue=False, show_progress=True)
                history_thumbnail_layout.change(
                    lambda layout: gr.Gallery.update(columns=2 if layout == 'Small (2 columns)' else 1),
                    inputs=history_thumbnail_layout,
                    outputs=history_gallery,
                    queue=False,
                    show_progress=False
                )
                history_batch_selection.change(load_history_batch,
                                               inputs=[history_batch_selection, history_search,
                                                       history_filter_favorites, history_filter_status,
                                                       history_filter_tag, history_day_selection,
                                                       history_filter_checkpoints, history_filter_loras,
                                                       history_show_preview_images, history_thumbnail_visibility],
                                               outputs=[history_gallery, history_visible_image_ids,
                                                        history_selected_image_ids, history_selected_image_ids_json,
                                                        history_selected_gallery,
                                                        history_image_selection, history_comparison_table,
                                                        history_batch_favorite, history_batch_rating,
                                                        history_batch_review_status, history_batch_tags,
                                                        history_batch_note, history_favorite, history_rating,
                                                        history_review_status, history_tags, history_note,
                                                        history_image_details,
                                                       history_status],
                                               queue=False, show_progress=False)
                for history_filter in [history_search, history_filter_favorites,
                                       history_filter_checkpoints, history_filter_loras,
                                       history_show_preview_images, history_thumbnail_visibility]:
                    history_filter.change(refresh_history, inputs=history_filter_inputs,
                                          outputs=history_refresh_outputs,
                                          queue=False, show_progress=False)
                history_day_selection.change(load_history_days,
                                             inputs=[history_day_selection, history_selected_days,
                                                     history_day_selection_mode, history_batch_selection,
                                                     history_search, history_filter_favorites,
                                                     history_filter_status, history_filter_tag,
                                                     history_filter_checkpoints, history_filter_loras,
                                                     history_show_preview_images, history_stack_by_seed,
                                                     history_thumbnail_visibility],
                                             outputs=[history_day_selection, history_selected_days,
                                                      history_seed_stack_selection, history_seed_stack_prompt,
                                                      history_gallery, history_visible_image_ids,
                                                      history_selected_image_ids, history_selected_image_ids_json,
                                                      history_selected_gallery,
                                                      history_image_selection, history_comparison_table,
                                                      history_batch_favorite, history_batch_rating,
                                                      history_batch_review_status, history_batch_tags,
                                                      history_batch_note, history_favorite, history_rating,
                                                      history_review_status, history_tags, history_note,
                                                      history_image_details,
                                                      history_status],
                                             queue=False, show_progress=False)
                history_stack_by_seed.change(refresh_history,
                                             inputs=history_filter_inputs,
                                             outputs=history_refresh_outputs,
                                             queue=False, show_progress=False)
                history_seed_stack_selection.change(
                    lambda selection: seed_stack_prompt_by_choice(selection),
                    inputs=history_seed_stack_selection,
                    outputs=history_seed_stack_prompt,
                    queue=False,
                    show_progress=False
                ).then(apply_history_seed_group,
                       inputs=[history_stack_by_seed, history_seed_stack_selection,
                               history_seed_stack_prompt, history_search, history_filter_favorites,
                               history_filter_status, history_filter_tag, history_day_selection,
                               history_filter_checkpoints, history_filter_loras,
                               history_show_preview_images, history_thumbnail_visibility],
                       outputs=[history_selected_image_ids, history_selected_image_ids_json,
                                history_selected_gallery, history_image_selection,
                                history_favorite, history_rating, history_review_status,
                                history_tags, history_note, history_image_details, history_status],
                       queue=False, show_progress=False)
                history_image_selection.change(load_history_image_curation, inputs=history_image_selection,
                                               outputs=[history_favorite, history_rating, history_review_status,
                                                        history_tags, history_note, history_image_details,
                                                        history_status],
                                               queue=False, show_progress=False)
                history_selected_gallery.select(select_history_selected_gallery_image,
                                               inputs=history_selected_image_ids,
                                               outputs=[history_image_selection, history_favorite,
                                                        history_rating, history_review_status, history_tags,
                                                        history_note, history_image_details, history_status],
                                               queue=False, show_progress=False)
                history_select_thumbnail_button.click(select_history_thumbnail_by_ref,
                                                      inputs=[history_select_thumbnail_image_id,
                                                              history_visible_image_ids, history_selected_image_ids,
                                                              history_selection_mode, history_search,
                                                              history_filter_favorites, history_filter_status,
                                                              history_filter_tag, history_day_selection,
                                                              history_filter_checkpoints, history_filter_loras,
                                                              history_show_preview_images, history_thumbnail_visibility],
                                                      outputs=[history_selected_image_ids,
                                                               history_selected_image_ids_json,
                                                              history_selected_gallery,
                                                              history_image_selection, history_favorite,
                                                               history_rating, history_review_status,
                                                               history_tags, history_note, history_image_details,
                                                               history_status],
                                                      queue=False, show_progress=False)
                history_comparison_table.select(select_history_comparison, inputs=history_comparison_table,
                                                outputs=[history_selected_image_ids, history_selected_image_ids_json,
                                                         history_selected_gallery,
                                                         history_image_selection, history_favorite,
                                                         history_rating, history_review_status,
                                                         history_tags, history_note, history_image_details,
                                                         history_status],
                                                queue=False, show_progress=False)
                history_bulk_delete_button.click(bulk_delete_history_images,
                                                 inputs=[history_selected_image_ids] + history_filter_inputs,
                                                 outputs=history_refresh_outputs,
                                                 queue=False, show_progress=False)
                history_bulk_favorite_button.click(bulk_favorite_history_images,
                                                   inputs=[history_selected_image_ids] + history_filter_inputs,
                                                   outputs=history_refresh_outputs,
                                                   queue=False, show_progress=False)
                history_bulk_hide_button.click(bulk_hide_history_thumbnails,
                                               inputs=[history_selected_image_ids] + history_filter_inputs,
                                               outputs=history_refresh_outputs,
                                               queue=False, show_progress=False)
                history_remove_selected_image_button.click(remove_history_selected_image,
                                                           inputs=[history_selected_image_ids,
                                                                   history_remove_selected_image_id],
                                                           outputs=[history_selected_image_ids,
                                                                   history_selected_image_ids_json,
                                                                    history_selected_gallery, history_image_details,
                                                                    history_status],
                                                           queue=False, show_progress=False)
                history_delete_selected_image_button.click(delete_history_selected_image,
                                                           inputs=[history_selected_image_ids,
                                                                   history_visible_image_ids,
                                                                   history_delete_selected_image_id],
                                                           outputs=[history_gallery, history_visible_image_ids,
                                                                    history_selected_image_ids,
                                                                    history_selected_image_ids_json,
                                                                    history_selected_gallery,
                                                                    history_image_selection, history_image_details,
                                                                    history_status],
                                                           queue=False, show_progress=False)
                history_apply_selected_image_button.click(load_history_image_config_by_id,
                                                          inputs=[history_apply_selected_image_id, prompt,
                                                                  state_is_generating, inpaint_mode],
                                                          outputs=history_load_outputs,
                                                          queue=False, show_progress=False) \
                    .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.sort_wildprompts, inputs=wildprompt_selections, outputs=wildprompt_selections, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.update_wildprompt_line_sections, inputs=[wildprompt_selections, wildprompt_line_selection_json], outputs=wildprompt_line_section_outputs, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.build_wildprompt_combination_summary, inputs=[wildprompt_selections, wildprompt_generate_all, wildprompt_line_selection_json, wildprompt_generation_factors], outputs=wildprompt_combination_summary, queue=False, show_progress=False) \
                    .then(lambda: None, _js='()=>{refresh_style_localization();refresh_wildprompt_localization();}')
                history_toggle_favorite_button.click(toggle_history_image_favorite,
                                                     inputs=history_toggle_favorite_image_id,
                                                     outputs=history_status,
                                                     queue=False, show_progress=False)
                history_hide_thumbnail_button.click(
                    hide_history_thumbnail,
                    inputs=[history_hide_thumbnail_image_id, history_search,
                            history_filter_favorites, history_filter_status, history_filter_tag,
                            history_filter_checkpoints, history_filter_loras,
                            history_selected_days,
                            history_show_preview_images, history_stack_by_seed,
                            history_thumbnail_visibility],
                    outputs=history_refresh_outputs,
                    queue=False,
                    show_progress=False
                )
                history_save_curation_button.click(save_history_image_curation,
                                                   inputs=[history_image_selection, history_favorite, history_rating,
                                                           history_review_status, history_tags, history_note,
                                                           history_batch_selection, history_filter_favorites,
                                                           history_filter_status, history_filter_tag],
                                                   outputs=[history_image_selection, history_gallery,
                                                            history_comparison_table, history_status],
                                                   queue=False, show_progress=False)
                history_save_batch_curation_button.click(save_history_batch_curation,
                                                         inputs=[history_batch_selection, history_batch_favorite,
                                                                 history_batch_rating, history_batch_review_status,
                                                                 history_batch_tags, history_batch_note,
                                                                 history_search, history_filter_favorites,
                                                                 history_filter_status, history_filter_tag],
                                                         outputs=[history_batch_selection, history_status],
                                                         queue=False, show_progress=False)
                history_load_full_button.click(load_full_history_config,
                                               inputs=[history_config_action_image_id, prompt, state_is_generating, inpaint_mode],
                                               outputs=history_load_outputs,
                                               queue=False, show_progress=False) \
                    .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.sort_wildprompts, inputs=wildprompt_selections, outputs=wildprompt_selections, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.update_wildprompt_line_sections, inputs=[wildprompt_selections, wildprompt_line_selection_json], outputs=wildprompt_line_section_outputs, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.build_wildprompt_combination_summary, inputs=[wildprompt_selections, wildprompt_generate_all, wildprompt_line_selection_json, wildprompt_generation_factors], outputs=wildprompt_combination_summary, queue=False, show_progress=False) \
                    .then(lambda: None, _js='()=>{refresh_style_localization();refresh_wildprompt_localization();}')
                history_replace_prompt_button.click(replace_prompt_from_history,
                                                    inputs=[history_config_action_image_id, prompt, state_is_generating, inpaint_mode],
                                                    outputs=history_load_outputs,
                                                    queue=False, show_progress=False)
                history_append_prompt_button.click(append_prompt_from_history,
                                                   inputs=[history_config_action_image_id, prompt, state_is_generating, inpaint_mode],
                                                   outputs=history_load_outputs,
                                                   queue=False, show_progress=False)
                history_send_to_inpaint_button.click(
                    send_history_image_to_inpaint,
                    inputs=history_config_action_image_id,
                    outputs=[input_image_checkbox, image_input_panel, current_tab, image_input_tabs,
                             inpaint_input_image, history_status],
                    queue=False,
                    show_progress=False,
                    _js='(x)=>{viewer_to_bottom(500); return x;}'
                )

                load_full_prompt_config_button.click(load_full_prompt_config, inputs=[prompt_config_selection, prompt, state_is_generating, inpaint_mode],
                                                     outputs=load_prompt_config_outputs,
                                                     queue=False, show_progress=False) \
                    .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.sort_wildprompts, inputs=wildprompt_selections, outputs=wildprompt_selections, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.update_wildprompt_line_sections, inputs=[wildprompt_selections, wildprompt_line_selection_json], outputs=wildprompt_line_section_outputs, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.build_wildprompt_combination_summary, inputs=[wildprompt_selections, wildprompt_generate_all, wildprompt_line_selection_json, wildprompt_generation_factors], outputs=wildprompt_combination_summary, queue=False, show_progress=False) \
                    .then(lambda: None, _js='()=>{refresh_style_localization();refresh_wildprompt_localization();}')
                replace_prompt_config_button.click(replace_prompt_from_config, inputs=[prompt_config_selection, prompt, state_is_generating, inpaint_mode],
                                                   outputs=load_prompt_config_outputs,
                                                   queue=False, show_progress=False)
                append_prompt_config_button.click(append_prompt_from_config, inputs=[prompt_config_selection, prompt, state_is_generating, inpaint_mode],
                                                 outputs=load_prompt_config_outputs,
                                                 queue=False, show_progress=False)

                gallery.select(select_generation_image, inputs=state_session_gallery,
                               outputs=[state_selected_generation_index, selected_image_status],
                               queue=False, show_progress=False)
                show_selected_generation_detail_button.click(select_generation_image_by_index,
                                                             inputs=[selected_generation_detail_index,
                                                                     state_session_gallery],
                                                             outputs=[state_selected_generation_index,
                                                                      selected_image_status],
                                                             queue=False, show_progress=False)
                apply_selected_image_config_button.click(apply_selected_generation_config,
                                                         inputs=[selected_generation_apply_index, state_session_gallery,
                                                                 state_is_generating, inpaint_mode],
                                                         outputs=load_data_outputs + person_likeness_outputs +
                                                                 lora_prompt_ctrls + lora_note_buttons +
                                                                 lora_note_add_buttons + lora_note_editor_cols +
                                                                 [selected_image_status],
                                                         queue=False, show_progress=False) \
                    .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.sort_wildprompts, inputs=wildprompt_selections, outputs=wildprompt_selections, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.update_wildprompt_line_sections, inputs=[wildprompt_selections, wildprompt_line_selection_json], outputs=wildprompt_line_section_outputs, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.build_wildprompt_combination_summary, inputs=[wildprompt_selections, wildprompt_generate_all, wildprompt_line_selection_json, wildprompt_generation_factors], outputs=wildprompt_combination_summary, queue=False, show_progress=False) \
                    .then(lambda: None, _js='()=>{refresh_style_localization();refresh_wildprompt_localization();}')
                clear_session_history_button.click(clear_session_history,
                                                   outputs=[gallery, state_session_gallery, state_selected_generation_index,
                                                            quick_preview_generation_indices, selected_image_status],
                                                   queue=False, show_progress=False)
                remove_selected_image_button.click(remove_generation_from_history,
                                                   inputs=[selected_generation_remove_index, state_session_gallery],
                                                   outputs=[gallery, state_session_gallery, state_selected_generation_index,
                                                            quick_preview_generation_indices, selected_image_status],
                                                   queue=False, show_progress=False)
                delete_selected_image_button.click(delete_generation_from_history,
                                                   inputs=[selected_generation_delete_index, state_session_gallery],
                                                   outputs=[gallery, state_session_gallery, state_selected_generation_index,
                                                            quick_preview_generation_indices, selected_image_status],
                                                   queue=False, show_progress=False)
                favorite_selected_generation_button.click(toggle_session_generation_favorite,
                                                          inputs=[selected_generation_favorite_index,
                                                                  state_session_gallery],
                                                          outputs=selected_image_status,
                                                          queue=False, show_progress=False)
                remove_queued_task_button.click(remove_queued_task,
                                                inputs=selected_queue_remove_id,
                                                outputs=[queue_status_html, selected_image_status],
                                                queue=False, show_progress=False)
        
                if not args_manager.args.disable_preset_selection:
                    def preset_selection_change(preset, is_generating, inpaint_mode):
                        preset_content = modules.config.try_get_preset_content(preset) if preset != 'initial' else {}
                        preset_prepared = modules.meta_parser.parse_meta_from_preset(preset_content)
        
                        default_model = preset_prepared.get('base_model')
                        previous_default_models = preset_prepared.get('previous_default_models', [])
                        checkpoint_downloads = preset_prepared.get('checkpoint_downloads', {})
                        embeddings_downloads = preset_prepared.get('embeddings_downloads', {})
                        lora_downloads = preset_prepared.get('lora_downloads', {})
                        vae_downloads = preset_prepared.get('vae_downloads', {})
        
                        preset_prepared['base_model'], preset_prepared['checkpoint_downloads'] = launch.download_models(
                            default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads,
                            vae_downloads)
        
                        if 'prompt' in preset_prepared and preset_prepared.get('prompt') == '':
                            del preset_prepared['prompt']
        
                        return modules.meta_parser.load_parameter_button_click(json.dumps(preset_prepared), is_generating, inpaint_mode)
        
        
                    def inpaint_engine_state_change(inpaint_engine_version, *args):
                        if inpaint_engine_version == 'empty':
                            inpaint_engine_version = modules.config.default_inpaint_engine_version
        
                        result = []
                        for inpaint_mode in args:
                            if inpaint_mode != modules.flags.inpaint_option_detail:
                                result.append(gr.update(value=inpaint_engine_version))
                            else:
                                result.append(gr.update())
        
                        return result
        
                    preset_selection.change(preset_selection_change, inputs=[preset_selection, state_is_generating, inpaint_mode], outputs=load_data_outputs, queue=False, show_progress=True) \
                        .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                        .then(fn=wildprompt_sorter.sort_wildprompts, inputs=wildprompt_selections, outputs=wildprompt_selections, queue=False, show_progress=False) \
                        .then(fn=wildprompt_sorter.update_wildprompt_line_sections, inputs=[wildprompt_selections, wildprompt_line_selection_json], outputs=wildprompt_line_section_outputs, queue=False, show_progress=False) \
                        .then(fn=wildprompt_sorter.build_wildprompt_combination_summary, inputs=[wildprompt_selections, wildprompt_generate_all, wildprompt_line_selection_json, wildprompt_generation_factors], outputs=wildprompt_combination_summary, queue=False, show_progress=False) \
                        .then(lambda: None, _js='()=>{refresh_style_localization();refresh_wildprompt_localization();}') \
                        .then(inpaint_engine_state_change, inputs=[inpaint_engine_state] + enhance_inpaint_mode_ctrls, outputs=enhance_inpaint_engine_ctrls, queue=False, show_progress=False)
        
                performance_selection.change(lambda x: [gr.update(interactive=not flags.Performance.has_restricted_features(x))] * 11 +
                                                       [gr.update(visible=not flags.Performance.has_restricted_features(x))] * 1 +
                                                       [gr.update(value=flags.Performance.has_restricted_features(x))] * 1,
                                             inputs=performance_selection,
                                             outputs=[
                                                 guidance_scale, sharpness, adm_scaler_end, adm_scaler_positive,
                                                 adm_scaler_negative, refiner_switch, refiner_model, sampler_name,
                                                 scheduler_name, adaptive_cfg, refiner_swap_method, negative_prompt, disable_intermediate_results
                                             ], queue=False, show_progress=False)
        
                output_format.input(lambda x: gr.update(output_format=x), inputs=output_format)
        
                advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, advanced_column,
                                         queue=False, show_progress=False) \
                    .then(fn=lambda: None, _js='refresh_grid_delayed', queue=False, show_progress=False)
        
                inpaint_mode.change(inpaint_mode_change, inputs=[inpaint_mode, inpaint_engine_state], outputs=[
                    inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
                    inpaint_disable_initial_latent, inpaint_engine,
                    inpaint_strength, inpaint_respective_field
                ], show_progress=False, queue=False)
        
                # load configured default_inpaint_method
                default_inpaint_ctrls = [inpaint_mode, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field]
                for mode, disable_initial_latent, engine, strength, respective_field in [default_inpaint_ctrls] + enhance_inpaint_update_ctrls:
                    shared.gradio_root.load(inpaint_mode_change, inputs=[mode, inpaint_engine_state], outputs=[
                        inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts, disable_initial_latent,
                        engine, strength, respective_field
                    ], show_progress=False, queue=False)
        
                generate_mask_button.click(fn=generate_mask,
                                           inputs=[inpaint_input_image, inpaint_mask_model, inpaint_mask_cloth_category,
                                                   inpaint_mask_dino_prompt_text, inpaint_mask_sam_model,
                                                   inpaint_mask_box_threshold, inpaint_mask_text_threshold,
                                                   inpaint_mask_sam_max_detections, dino_erode_or_dilate, debugging_dino],
                                           outputs=inpaint_mask_image, show_progress=True, queue=True)
        
                ctrls = [currentTask, generate_image_grid]
                ctrls += [quick_preview_mode]
                ctrls += [
                    prompt, negative_prompt, style_selections, wildprompt_selections, wildprompt_generate_all,
                    wildprompt_line_selection_json,
                    performance_selection, aspect_ratios_selection, image_number, output_format, image_seed,
                    read_wildcards_in_order, sharpness, guidance_scale
                ]
        
                ctrls += [base_model, multi_checkpoint_enabled, multi_checkpoint_models, refiner_model, refiner_switch] + lora_ctrls
                ctrls += [input_image_checkbox, current_tab]
                ctrls += person_likeness_ctrls
                ctrls += [uov_method, uov_input_image]
                ctrls += [outpaint_selections, inpaint_input_image, inpaint_additional_prompt, inpaint_mask_image]
                ctrls += [disable_preview, disable_intermediate_results, disable_seed_increment, black_out_nsfw]
                ctrls += [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, clip_skip]
                ctrls += [sampler_name, scheduler_name, vae_name]
                ctrls += [overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength]
                ctrls += [overwrite_upscale_strength, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint]
                ctrls += [debugging_cn_preprocessor, skipping_cn_preprocessor, canny_low_threshold, canny_high_threshold]
                ctrls += [refiner_swap_method, controlnet_softness]
                ctrls += freeu_ctrls
                ctrls += inpaint_ctrls
                ctrls += [training_mode, testing_mode, testing_loras]
        
                if not args_manager.args.disable_image_log:
                    ctrls += [save_final_enhanced_image_only]
        
                if not args_manager.args.disable_metadata:
                    ctrls += [save_metadata_to_images, metadata_scheme]
        
                ctrls += ip_ctrls
                ctrls += [debugging_dino, dino_erode_or_dilate, debugging_enhance_masks_checkbox,
                          enhance_input_image, enhance_checkbox, enhance_uov_method, enhance_uov_processing_order,
                          enhance_uov_prompt_type]
                ctrls += enhance_ctrls

                wildprompt_line_selection_inputs = [wildprompt_selections] + wildprompt_line_selection_ctrls
                wildprompt_line_selection_arg_index = ctrls.index(wildprompt_line_selection_json)

                def enqueue_generate_task_with_current_wildprompt_lines(*args):
                    row_input_count = len(wildprompt_line_selection_inputs)
                    line_selection_json = wildprompt_sorter.encode_wildprompt_line_selections(*args[:row_input_count])
                    generation_args = list(args[row_input_count:])
                    generation_args[wildprompt_line_selection_arg_index] = line_selection_json
                    return enqueue_generate_task(*generation_args)

                regenerate_selected_quality_button.click(enqueue_selected_generation_quality_config,
                                                         inputs=[selected_generation_quality_index, state_session_gallery] + ctrls,
                          outputs=[currentTask, state_queue_monitor, stop_button, skip_button, generate_button,
                                   reset_button, gallery, queue_status_html, state_is_generating, selected_image_status],
                          queue=False, show_progress=False) \
                    .then(fn=poll_generate_queue, inputs=[currentTask, state_is_generating, state_session_gallery],
                          outputs=[progress_html, progress_window, progress_gallery, gallery,
                                   state_session_gallery, generate_button, stop_button, skip_button,
                                   queue_status_html, quick_preview_generation_indices, state_is_generating],
                          queue=False, show_progress=False) \
                    .then(fn=update_history_link, outputs=history_link) \
                    .then(fn=lambda x: x, inputs=state_queue_monitor, outputs=state_queue_monitor,
                          queue=False, show_progress=False,
                          _js='(x)=>{if(x){playNotification();} return x;}') \
                    .then(fn=lambda x: x, inputs=state_queue_monitor, outputs=state_queue_monitor,
                          queue=False, show_progress=False,
                          _js='(x)=>{if(x){refresh_grid_delayed();} return x;}')

                history_quality_selected_image_button.click(enqueue_history_image_quality_config,
                                                            inputs=[history_quality_selected_image_id] + ctrls,
                          outputs=[currentTask, state_queue_monitor, stop_button, skip_button, generate_button,
                                   reset_button, gallery, queue_status_html, state_is_generating, history_status],
                          queue=False, show_progress=False) \
                    .then(fn=poll_generate_queue, inputs=[currentTask, state_is_generating, state_session_gallery],
                          outputs=[progress_html, progress_window, progress_gallery, gallery,
                                   state_session_gallery, generate_button, stop_button, skip_button,
                                   queue_status_html, quick_preview_generation_indices, state_is_generating],
                          queue=False, show_progress=False) \
                    .then(fn=update_history_link, outputs=history_link) \
                    .then(fn=lambda x: x, inputs=state_queue_monitor, outputs=state_queue_monitor,
                          queue=False, show_progress=False,
                          _js='(x)=>{if(x){playNotification();} return x;}') \
                    .then(fn=lambda x: x, inputs=state_queue_monitor, outputs=state_queue_monitor,
                          queue=False, show_progress=False,
                          _js='(x)=>{if(x){refresh_grid_delayed();} return x;}')

                def parse_meta(raw_prompt_txt, is_generating):
                    loaded_json = None
                    if is_json(raw_prompt_txt):
                        loaded_json = json.loads(raw_prompt_txt)
        
                    if loaded_json is None:
                        if is_generating:
                            return gr.update(), gr.update(), gr.update()
                        else:
                            return gr.update(), gr.update(visible=True), gr.update(visible=False)
        
                    return json.dumps(loaded_json), gr.update(visible=True), gr.update(visible=True)
        
                prompt.input(parse_meta, inputs=[prompt, state_is_generating], outputs=[prompt, generate_button, load_parameter_button], queue=False, show_progress=False)
        
                load_parameter_button.click(modules.meta_parser.load_parameter_button_click, inputs=[prompt, state_is_generating, inpaint_mode], outputs=load_data_outputs, queue=False, show_progress=False) \
                    .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.sort_wildprompts, inputs=wildprompt_selections, outputs=wildprompt_selections, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.update_wildprompt_line_sections, inputs=[wildprompt_selections, wildprompt_line_selection_json], outputs=wildprompt_line_section_outputs, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.build_wildprompt_combination_summary, inputs=[wildprompt_selections, wildprompt_generate_all, wildprompt_line_selection_json, wildprompt_generation_factors], outputs=wildprompt_combination_summary, queue=False, show_progress=False) \
                    .then(lambda: None, _js='()=>{refresh_style_localization();refresh_wildprompt_localization();}')
        
                def trigger_metadata_import(file, state_is_generating):
                    parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
                    if parameters is None:
                        print('Could not find metadata in the image!')
                        parsed_parameters = {}
                    else:
                        metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                        parsed_parameters = metadata_parser.to_json(parameters)
        
                    return modules.meta_parser.load_parameter_button_click(parsed_parameters, state_is_generating, inpaint_mode)
        
                metadata_import_button.click(trigger_metadata_import, inputs=[metadata_input_image, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=True) \
                    .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.sort_wildprompts, inputs=wildprompt_selections, outputs=wildprompt_selections, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.update_wildprompt_line_sections, inputs=[wildprompt_selections, wildprompt_line_selection_json], outputs=wildprompt_line_section_outputs, queue=False, show_progress=False) \
                    .then(wildprompt_sorter.build_wildprompt_combination_summary, inputs=[wildprompt_selections, wildprompt_generate_all, wildprompt_line_selection_json, wildprompt_generation_factors], outputs=wildprompt_combination_summary, queue=False, show_progress=False) \
                    .then(lambda: None, _js='()=>{refresh_style_localization();refresh_wildprompt_localization();}')
        
                generate_button.click(fn=lambda: set_quick_preview_mode(False), outputs=quick_preview_mode,
                                      queue=False, show_progress=False) \
                    .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed,
                          queue=False, show_progress=False) \
                    .then(fn=wildprompt_sorter.encode_wildprompt_line_selections,
                          inputs=[wildprompt_selections] + wildprompt_line_selection_ctrls,
                          outputs=wildprompt_line_selection_json,
                          queue=False, show_progress=False) \
                    .then(fn=enqueue_generate_task_with_current_wildprompt_lines,
                          inputs=wildprompt_line_selection_inputs + ctrls,
                          outputs=[currentTask, state_queue_monitor, stop_button, skip_button, generate_button,
                                   reset_button, gallery, queue_status_html, state_is_generating],
                          queue=False, show_progress=False) \
                    .then(fn=poll_generate_queue, inputs=[currentTask, state_is_generating, state_session_gallery],
                          outputs=[progress_html, progress_window, progress_gallery, gallery,
                                   state_session_gallery, generate_button, stop_button, skip_button,
                                   queue_status_html, quick_preview_generation_indices, state_is_generating],
                          queue=False, show_progress=False) \
                    .then(fn=update_history_link, outputs=history_link) \
                    .then(fn=lambda x: x, inputs=state_queue_monitor, outputs=state_queue_monitor,
                          queue=False, show_progress=False,
                          _js='(x)=>{if(x){playNotification();} return x;}') \
                    .then(fn=lambda x: x, inputs=state_queue_monitor, outputs=state_queue_monitor,
                          queue=False, show_progress=False,
                          _js='(x)=>{if(x){refresh_grid_delayed();} return x;}')

                quick_preview_button.click(fn=lambda: set_quick_preview_mode(True), outputs=quick_preview_mode,
                                           queue=False, show_progress=False) \
                    .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed,
                          queue=False, show_progress=False) \
                    .then(fn=wildprompt_sorter.encode_wildprompt_line_selections,
                          inputs=[wildprompt_selections] + wildprompt_line_selection_ctrls,
                          outputs=wildprompt_line_selection_json,
                          queue=False, show_progress=False) \
                    .then(fn=enqueue_generate_task_with_current_wildprompt_lines,
                          inputs=wildprompt_line_selection_inputs + ctrls,
                          outputs=[currentTask, state_queue_monitor, stop_button, skip_button, generate_button,
                                   reset_button, gallery, queue_status_html, state_is_generating],
                          queue=False, show_progress=False) \
                    .then(fn=poll_generate_queue, inputs=[currentTask, state_is_generating, state_session_gallery],
                          outputs=[progress_html, progress_window, progress_gallery, gallery,
                                   state_session_gallery, generate_button, stop_button, skip_button,
                                   queue_status_html, quick_preview_generation_indices, state_is_generating],
                          queue=False, show_progress=False) \
                    .then(fn=update_history_link, outputs=history_link) \
                    .then(fn=lambda x: x, inputs=state_queue_monitor, outputs=state_queue_monitor,
                          queue=False, show_progress=False,
                          _js='(x)=>{if(x){playNotification();} return x;}') \
                    .then(fn=lambda x: x, inputs=state_queue_monitor, outputs=state_queue_monitor,
                          queue=False, show_progress=False,
                          _js='(x)=>{if(x){refresh_grid_delayed();} return x;}')
        
                reset_button.click(reconnect_generate_queue,
                                   inputs=state_session_gallery,
                                   outputs=[currentTask, state_queue_monitor, generate_button, stop_button,
                                            skip_button, reset_button, queue_status_html, state_is_generating],
                                   queue=False, show_progress=False) \
                    .then(fn=poll_generate_queue, inputs=[currentTask, state_is_generating, state_session_gallery],
                          outputs=[progress_html, progress_window, progress_gallery, gallery,
                                   state_session_gallery, generate_button, stop_button, skip_button,
                                   queue_status_html, quick_preview_generation_indices, state_is_generating],
                          queue=False, show_progress=False)

                poll_generate_button.click(poll_generate_queue,
                                           inputs=[currentTask, state_is_generating, state_session_gallery],
                                           outputs=[progress_html, progress_window, progress_gallery, gallery,
                                                    state_session_gallery, generate_button, stop_button, skip_button,
                                                    queue_status_html, quick_preview_generation_indices,
                                                    state_is_generating],
                                           queue=False,
                                           show_progress=False)

                shared.gradio_root.load(poll_generate_queue,
                                        inputs=[currentTask, state_is_generating, state_session_gallery],
                                        outputs=[progress_html, progress_window, progress_gallery, gallery,
                                                 state_session_gallery, generate_button, stop_button, skip_button,
                                                 queue_status_html, quick_preview_generation_indices,
                                                 state_is_generating],
                                        every=1,
                                        queue=False,
                                        show_progress=False)
        
                for notification_file in ['notification.ogg', 'notification.mp3']:
                    if os.path.exists(notification_file):
                        gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
                        break
        
                def trigger_describe(modes, img, apply_styles):
                    describe_prompts = []
                    styles = set()
        
                    if flags.describe_type_photo in modes:
                        from extras.interrogate import default_interrogator as default_interrogator_photo
                        describe_prompts.append(default_interrogator_photo(img))
                        styles.update(["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"])
        
                    if flags.describe_type_anime in modes:
                        from extras.wd14tagger import default_interrogator as default_interrogator_anime
                        describe_prompts.append(default_interrogator_anime(img))
                        styles.update(["Fooocus V2", "Fooocus Masterpiece"])
        
                    if len(styles) == 0 or not apply_styles:
                        styles = gr.update()
                    else:
                        styles = list(styles)
        
                    if len(describe_prompts) == 0:
                        describe_prompt = gr.update()
                    else:
                        describe_prompt = ', '.join(describe_prompts)
        
                    return describe_prompt, styles
        
                describe_btn.click(trigger_describe, inputs=[describe_methods, describe_input_image, describe_apply_styles],
                                   outputs=[prompt, style_selections], show_progress=True, queue=True) \
                    .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                    .then(lambda: None, _js='()=>{refresh_style_localization();}')
        
                if args_manager.args.enable_auto_describe_image:
                    def trigger_auto_describe(mode, img, prompt, apply_styles):
                        # keep prompt if not empty
                        if prompt == '':
                            return trigger_describe(mode, img, apply_styles)
                        return gr.update(), gr.update()
        
                    uov_input_image.upload(trigger_auto_describe, inputs=[describe_methods, uov_input_image, prompt, describe_apply_styles],
                                           outputs=[prompt, style_selections], show_progress=True, queue=True) \
                        .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                        .then(lambda: None, _js='()=>{refresh_style_localization();}')
        
                    enhance_input_image.upload(lambda: gr.update(value=True), outputs=enhance_checkbox, queue=False, show_progress=False) \
                        .then(trigger_auto_describe, inputs=[describe_methods, enhance_input_image, prompt, describe_apply_styles],
                              outputs=[prompt, style_selections], show_progress=True, queue=True) \
                        .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                        .then(lambda: None, _js='()=>{refresh_style_localization();}')
        
def dump_default_english_config():
    from modules.localization import dump_english_config
    dump_english_config(grh.all_components)


# dump_default_english_config()

shared.gradio_root.launch(
    inbrowser=args_manager.args.in_browser,
    server_name=args_manager.args.listen,
    server_port=args_manager.args.port,
    share=args_manager.args.share,
    auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
    allowed_paths=[modules.config.path_outputs],
    blocked_paths=[constants.AUTH_FILENAME]
)

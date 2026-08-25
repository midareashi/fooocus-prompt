import copy
import html
import json

import gradio as gr

import modules.localization as localization
import modules.sdxl_styles as sdxl_styles

all_wildprompts = []
sort_file = 'sorted_wildprompt.json'
max_wildprompt_detail_sections = 24
all_categories_label = 'All folders'
uncategorized_label = 'Uncategorized'


def _as_list(value):
    return value if isinstance(value, list) else []


def try_load_sorted_wildprompts():
    global all_wildprompts

    legal_wildprompts = sdxl_styles.get_legal_wildprompt_names()
    all_wildprompts = legal_wildprompts

    try:
        with open(sort_file, 'rt', encoding='utf-8') as fp:
            sorted_wildprompts = json.load(fp)
        sorted_wildprompts = _as_list(sorted_wildprompts)
        selected = [x for x in sorted_wildprompts if x in legal_wildprompts]
        unselected = [x for x in legal_wildprompts if x not in selected]
        all_wildprompts = selected + unselected
    except Exception:
        pass

    return copy.deepcopy(all_wildprompts)


def sort_wildprompts(selected):
    global all_wildprompts

    selected = [x for x in _as_list(selected) if x in all_wildprompts]
    unselected = [y for y in all_wildprompts if y not in selected]
    sorted_wildprompts = selected + unselected

    try:
        with open(sort_file, 'wt', encoding='utf-8') as fp:
            json.dump(sorted_wildprompts, fp, indent=4)
    except Exception as e:
        print('Write wildprompt sorting failed.')
        print(e)

    all_wildprompts = sorted_wildprompts
    return gr.update(choices=sorted_wildprompts)


def localization_key(x):
    return x + localization.current_translation.get(x, '')


def get_wildprompt_category(wildprompt_name):
    name = str(wildprompt_name or '').replace('\\', '/')
    return name.split('/', 1)[0] if '/' in name else uncategorized_label


def get_wildprompt_categories():
    categories = sorted({get_wildprompt_category(name) for name in all_wildprompts}, key=str.casefold)
    return [all_categories_label] + categories


def filter_wildprompts(selected, category=all_categories_label, query=''):
    selected = [x for x in _as_list(selected) if x in all_wildprompts]
    category = category if category in get_wildprompt_categories() else all_categories_label
    query = query if isinstance(query, str) else ''
    unselected = [y for y in all_wildprompts if y not in selected]
    if category != all_categories_label:
        unselected = [name for name in unselected if get_wildprompt_category(name) == category]
    if len(query.replace(' ', '')) > 0:
        unselected = [name for name in unselected if query.casefold() in localization_key(name).casefold()]
    return gr.update(choices=selected + unselected, value=selected)


def search_wildprompts(selected, query):
    return filter_wildprompts(selected, all_categories_label, query)


def refresh_wildprompt_browser(selected, category, query):
    selected = [x for x in _as_list(selected)]
    try_load_sorted_wildprompts()
    categories = get_wildprompt_categories()
    category = category if category in categories else all_categories_label
    browser_update = filter_wildprompts(selected, category, query)
    return gr.update(choices=categories, value=category), browser_update


def get_wildprompt_lines(wildprompt_name):
    return sdxl_styles.load_wildprompt_lines(wildprompt_name)


def encode_wildprompt_line_selections(selected_files, *selected_line_groups):
    selected_files = _as_list(selected_files)
    result = {}

    for index, wildprompt_name in enumerate(selected_files[:max_wildprompt_detail_sections]):
        if index >= len(selected_line_groups):
            break
        selected_lines = _as_list(selected_line_groups[index])
        result[wildprompt_name] = selected_lines

    return json.dumps(result, ensure_ascii=False)


def update_wildprompt_line_sections(selected_files, current_json=''):
    selected_files = _as_list(selected_files)

    try:
        current = json.loads(current_json) if isinstance(current_json, str) and current_json != '' else {}
    except Exception:
        current = {}
    current = current if isinstance(current, dict) else {}

    updates = []
    for index in range(max_wildprompt_detail_sections):
        if index < len(selected_files):
            wildprompt_name = selected_files[index]
            lines = get_wildprompt_lines(wildprompt_name)
            has_saved_selection = wildprompt_name in current
            selected_lines = current.get(wildprompt_name, lines)
            selected_lines = [x for x in _as_list(selected_lines) if x in lines]
            if len(selected_lines) == 0 and not has_saved_selection:
                selected_lines = lines
            updates.append(gr.update(label=wildprompt_name, visible=True, open=False))
            updates.append(wildprompt_name)
            updates.append(gr.update(choices=lines, value=selected_lines))
        else:
            updates.append(gr.update(label='Wildprompt Rows', visible=False, open=False))
            updates.append('')
            updates.append(gr.update(choices=[], value=[]))

    return updates


def select_all_wildprompt_lines(wildprompt_name):
    return gr.update(value=get_wildprompt_lines(wildprompt_name))


def select_no_wildprompt_lines():
    return gr.update(value=[])


def build_generation_factors(image_number=1, multi_checkpoint_enabled=False, multi_checkpoint_models=None,
                             testing_mode=False, testing_loras=None):
    try:
        image_count = max(1, int(image_number))
    except (TypeError, ValueError):
        image_count = 1

    checkpoint_count = max(1, len(_as_list(multi_checkpoint_models))) \
        if bool(multi_checkpoint_enabled) else 1
    testing_lora_count = max(1, len(_as_list(testing_loras))) if bool(testing_mode) else 1
    return {
        'image_number': image_count,
        'checkpoint_count': checkpoint_count,
        'testing_lora_count': testing_lora_count,
    }


def build_wildprompt_combination_summary(selected_files, generate_all=False, current_json='',
                                         generation_factors=None):
    selected_files = [name for name in _as_list(selected_files) if name in all_wildprompts]

    try:
        current = json.loads(current_json) if isinstance(current_json, str) and current_json != '' else {}
    except Exception:
        current = {}
    current = current if isinstance(current, dict) else {}

    groups = []
    for name in selected_files:
        all_lines = get_wildprompt_lines(name)
        selected_lines = current.get(name, all_lines)
        selected_lines = [line for line in _as_list(selected_lines) if line in all_lines]
        if len(selected_lines) > 0:
            groups.append((name, len(selected_lines)))

    combination_count = 1
    combination_factors = []
    if bool(generate_all) and len(groups) > 0:
        for _, line_count in groups:
            combination_count *= line_count
            combination_factors.append(str(line_count))

    generation_factors = generation_factors if isinstance(generation_factors, dict) else {}
    image_number = max(1, int(generation_factors.get('image_number', 1) or 1))
    checkpoint_count = max(1, int(generation_factors.get('checkpoint_count', 1) or 1))
    testing_lora_count = max(1, int(generation_factors.get('testing_lora_count', 1) or 1))
    total_images = combination_count * image_number * checkpoint_count * testing_lora_count

    formula_parts = []
    if bool(generate_all) and len(groups) > 0:
        formula_parts.append(f'{combination_count:,} combination(s)')
    formula_parts.append(f'{image_number:,} Image Number')
    if checkpoint_count > 1:
        formula_parts.append(f'{checkpoint_count:,} checkpoints')
    if testing_lora_count > 1:
        formula_parts.append(f'{testing_lora_count:,} testing LoRAs')

    if total_images >= 100:
        total_class = 'wildprompt-total-danger'
        warning = '<div class="wildprompt-total-warning">Large queue — review these multipliers before Generate.</div>'
    elif total_images >= 25:
        total_class = 'wildprompt-total-warning-level'
        warning = '<div class="wildprompt-total-caution">This will create a sizable queue.</div>'
    else:
        total_class = ''
        warning = ''

    total_label = 'image' if total_images == 1 else 'images'
    escaped_names = ', '.join(html.escape(name) for name, _ in groups)
    if len(groups) == 0:
        mode_summary = 'No active Wildprompt rows; Wildprompt adds no multiplier.'
    elif bool(generate_all):
        row_factors = ' &times; '.join(combination_factors)
        mode_summary = f'Every combination: {row_factors} = {combination_count:,}.'
    else:
        mode_summary = f'Random mix: one selected row from each of {len(groups)} file(s) per image.'

    return (
        '<div class="wildprompt-combination-summary">'
        f'<strong class="wildprompt-total {total_class}">{total_images:,} total {total_label}</strong>'
        f'<div class="wildprompt-total-formula">{" &times; ".join(formula_parts)}</div>'
        f'{warning}<div>{mode_summary}</div>'
        f'<span>{escaped_names}</span></div>'
    )

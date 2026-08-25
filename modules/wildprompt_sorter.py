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
            selected_lines = current.get(wildprompt_name, lines)
            selected_lines = [x for x in _as_list(selected_lines) if x in lines]
            if len(selected_lines) == 0:
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


def build_wildprompt_combination_summary(selected_files, generate_all=False, current_json=''):
    selected_files = [name for name in _as_list(selected_files) if name in all_wildprompts]
    if len(selected_files) == 0:
        return '<div class="wildprompt-combination-summary">Select files from one or more folders to build a prompt.</div>'

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

    if len(groups) == 0:
        return '<div class="wildprompt-combination-summary">No prompt rows are selected.</div>'

    escaped_names = ', '.join(html.escape(name) for name, _ in groups)
    if not bool(generate_all):
        return (
            '<div class="wildprompt-combination-summary"><strong>Random mix</strong> &middot; '
            f'one selected row from each of {len(groups)} file(s) per image.<br>'
            f'<span>{escaped_names}</span></div>'
        )

    combination_count = 1
    for _, line_count in groups:
        combination_count *= line_count
    factors = ' &times; '.join(str(line_count) for _, line_count in groups)
    color = '#f59e0b' if combination_count > 100 else 'var(--body-text-color)'
    return (
        f'<div class="wildprompt-combination-summary"><strong style="color:{color}">'
        f'{combination_count:,} combination(s)</strong> &middot; {factors}. '
        'Image Number repeats every combination.<br>'
        f'<span>{escaped_names}</span></div>'
    )

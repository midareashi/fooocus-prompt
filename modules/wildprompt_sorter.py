import copy
import json

import gradio as gr

import modules.localization as localization
import modules.sdxl_styles as sdxl_styles

all_wildprompts = []
sort_file = 'sorted_wildprompt.json'
max_wildprompt_detail_sections = 24


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


def search_wildprompts(selected, query):
    selected = [x for x in _as_list(selected) if x in all_wildprompts]
    query = query if isinstance(query, str) else ''
    unselected = [y for y in all_wildprompts if y not in selected]
    matched = [y for y in unselected if query.lower() in localization_key(y).lower()] if len(query.replace(' ', '')) > 0 else []
    unmatched = [y for y in unselected if y not in matched]
    sorted_wildprompts = matched + selected + unmatched
    return gr.update(choices=sorted_wildprompts)


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

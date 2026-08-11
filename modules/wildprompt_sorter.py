import copy
import json

import gradio as gr

import modules.localization as localization
import modules.sdxl_styles as sdxl_styles

all_wildprompts = []
sort_file = 'sorted_wildprompt.json'


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

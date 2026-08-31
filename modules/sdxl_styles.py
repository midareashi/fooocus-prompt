import os
import re
import json
import math
import itertools
import ast

from modules.extra_utils import get_files_from_folder
from random import Random

# cannot use modules.config - validators causing circular imports
styles_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sdxl_styles/'))
style_layers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../styles/'))
wildprompts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../wildprompts/'))


def normalize_key(k):
    k = k.replace('-', ' ')
    words = k.split(' ')
    words = [w[:1].upper() + w[1:].lower() for w in words]
    k = ' '.join(words)
    k = k.replace('3d', '3D')
    k = k.replace('Sai', 'SAI')
    k = k.replace('Mre', 'MRE')
    k = k.replace('(s', '(S')
    return k


styles = {}
styles_files = get_files_from_folder(styles_path, ['.json'])

for x in ['sdxl_styles_fooocus.json',
          'sdxl_styles_sai.json',
          'sdxl_styles_mre.json',
          'sdxl_styles_twri.json',
          'sdxl_styles_diva.json',
          'sdxl_styles_marc_k3nt3l.json']:
    if x in styles_files:
        styles_files.remove(x)
        styles_files.append(x)

for styles_file in styles_files:
    try:
        with open(os.path.join(styles_path, styles_file), encoding='utf-8') as f:
            for entry in json.load(f):
                name = normalize_key(entry['name'])
                prompt = entry['prompt'] if 'prompt' in entry else ''
                negative_prompt = entry['negative_prompt'] if 'negative_prompt' in entry else ''
                styles[name] = (prompt, negative_prompt)
    except Exception as e:
        print(str(e))
        print(f'Failed to load style file {styles_file}')

legacy_style_keys = list(styles.keys())


def load_style_layers(folder_path=None):
    folder_path = os.path.abspath(folder_path or style_layers_path)
    loaded_styles = {}
    categories = {}
    selection_modes = {}

    if not os.path.isdir(folder_path):
        return loaded_styles, categories, selection_modes

    for relative_path in get_files_from_folder(folder_path, ['.json']):
        normalized_path = relative_path.replace('\\', '/')
        parts = normalized_path.split('/')
        category = parts[0] if len(parts) > 1 else 'Uncategorized'
        try:
            with open(os.path.join(folder_path, relative_path), encoding='utf-8') as f:
                data = json.load(f)
            entries = data.get('styles', []) if isinstance(data, dict) else []
            selection_mode = data.get('selection_mode', 'additive') if isinstance(data, dict) else 'additive'
            selection_mode = selection_mode if selection_mode in ['single', 'additive'] else 'additive'
            selection_modes[category] = selection_mode

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get('name', '') or '').strip()
                prompt = str(entry.get('prompt', '') or '').strip()
                negative_prompt = str(entry.get('negative_prompt', '') or '').strip()
                if name == '' or prompt == '':
                    continue
                style_name = f'{category}/{name}'
                loaded_styles[style_name] = (prompt, negative_prompt)
                categories[style_name] = category
        except Exception as e:
            print(str(e))
            print(f'Failed to load style layer file {relative_path}')

    return loaded_styles, categories, selection_modes


style_layers, style_layer_categories, style_layer_selection_modes = load_style_layers()
styles.update(style_layers)
style_keys = list(styles.keys())
fooocus_expansion = 'Fooocus V2'
random_style_name = 'Random Style'
legal_style_names = [fooocus_expansion, random_style_name] + style_keys

foundation_style_names = [
    fooocus_expansion,
    'Fooocus Negative',
    'Fooocus Photograph',
]
style_layer_categories[fooocus_expansion] = 'Foundation'
style_layer_categories['Fooocus Negative'] = 'Foundation'
style_layer_categories['Fooocus Photograph'] = 'Aesthetic'
style_layer_selection_modes['Foundation'] = 'additive'


def get_legal_style_layer_names():
    names = [name for name in foundation_style_names if name in legal_style_names]
    names += list(style_layers.keys())
    return sorted(names, key=str.casefold)


def get_style_layer_category(style_name):
    return style_layer_categories.get(style_name, 'Legacy')


def get_style_layer_selection_mode(style_name):
    category = get_style_layer_category(style_name)
    return style_layer_selection_modes.get(category, 'additive')


def normalize_style_layer_selections(selected):
    selected = selected if isinstance(selected, list) else []
    selected = [name for name in selected if name in legal_style_names]
    selected = list(dict.fromkeys(selected))
    last_single_index = {}

    for index, name in enumerate(selected):
        if get_style_layer_selection_mode(name) == 'single':
            last_single_index[get_style_layer_category(name)] = index

    return [
        name for index, name in enumerate(selected)
        if get_style_layer_selection_mode(name) != 'single'
        or last_single_index.get(get_style_layer_category(name)) == index
    ]


def get_legal_wildprompt_names():
    wildprompt_files = get_files_from_folder(wildprompts_path, ['.txt'])
    names = [os.path.splitext(file)[0].replace('\\', '/') for file in wildprompt_files]
    return sorted(names, key=str.casefold)


legal_wildprompt_names = get_legal_wildprompt_names()


def get_random_style(rng: Random) -> str:
    return rng.choice(legacy_style_keys)


def apply_style(style, positive):
    p, n = styles[style]
    return p.replace('{prompt}', positive).splitlines(), n.splitlines(), '{prompt}' in p


def _wildprompt_path(wildprompt_selection):
    selection = str(wildprompt_selection or '').replace('\\', '/').strip('/')
    parts = [part for part in selection.split('/') if part not in ['', '.']]
    if len(parts) == 0 or any(part == '..' for part in parts):
        raise ValueError('Invalid wildprompt name.')

    root = os.path.abspath(wildprompts_path)
    path = os.path.abspath(os.path.join(root, *parts)) + '.txt'
    if os.path.commonpath([root, path]) != root:
        raise ValueError('Wildprompt path escaped the wildprompts folder.')
    return path


def _load_wildprompt_lines(wildprompt_selection):
    with open(_wildprompt_path(wildprompt_selection), encoding='utf-8') as f:
        return [x for x in f.read().splitlines() if x.strip() != '']


def load_wildprompt_lines(wildprompt_selection):
    try:
        return _load_wildprompt_lines(wildprompt_selection)
    except Exception:
        return []


def normalize_wildprompt_generate_all_files(wildprompt_selections, generate_all_files=None):
    wildprompt_selections = wildprompt_selections if isinstance(wildprompt_selections, list) else []
    parsed = generate_all_files

    if isinstance(parsed, str):
        value = parsed.strip()
        if value.casefold() in ['true', '1', 'yes', 'on']:
            parsed = wildprompt_selections
        elif value.casefold() in ['', 'false', '0', 'no', 'off']:
            parsed = []
        else:
            try:
                parsed = json.loads(value)
            except Exception:
                try:
                    parsed = ast.literal_eval(value)
                except Exception:
                    parsed = []
    elif isinstance(parsed, bool):
        parsed = wildprompt_selections if parsed else []

    parsed = parsed if isinstance(parsed, (list, tuple, set)) else []
    requested = {name for name in parsed if isinstance(name, str)}
    return [name for name in wildprompt_selections if name in requested]


def resolve_wildprompts(wildprompt_selections, rng, wildprompt_line_selections=None, fixed_lines=None):
    resolved = []
    wildprompt_line_selections = wildprompt_line_selections if isinstance(wildprompt_line_selections, dict) else {}
    fixed_lines = fixed_lines if isinstance(fixed_lines, dict) else {}

    for wildprompt_selection in wildprompt_selections:
        try:
            selected_lines = wildprompt_line_selections.get(wildprompt_selection, None)
            if isinstance(selected_lines, list) and len(selected_lines) == 0:
                continue
            wildprompt_lines = selected_lines if isinstance(selected_lines, list) else _load_wildprompt_lines(wildprompt_selection)
            wildprompt_lines = [x for x in wildprompt_lines if isinstance(x, str) and x.strip() != '']
            assert len(wildprompt_lines) > 0
            fixed_line = fixed_lines.get(wildprompt_selection)
            resolved.append({
                'name': wildprompt_selection,
                'prompt': fixed_line if fixed_line in wildprompt_lines else rng.choice(wildprompt_lines),
            })
        except Exception:
            print(f'[Wildprompts] Warning: {wildprompt_selection}.txt missing or empty.')

    return resolved


def apply_wildprompts(wildprompt_selections, rng, wildprompt_line_selections=None, fixed_lines=None):
    resolved = resolve_wildprompts(
        wildprompt_selections,
        rng,
        wildprompt_line_selections,
        fixed_lines,
    )
    return ', '.join(item['prompt'] for item in resolved)


def get_wildprompt_fixed_combinations(wildprompt_selections, generate_all_files,
                                      wildprompt_line_selections=None):
    wildprompt_selections = wildprompt_selections if isinstance(wildprompt_selections, list) else []
    generate_all_files = normalize_wildprompt_generate_all_files(wildprompt_selections, generate_all_files)
    wildprompt_line_selections = wildprompt_line_selections if isinstance(wildprompt_line_selections, dict) else {}
    names = []
    prompt_groups = []

    for wildprompt_selection in generate_all_files:
        try:
            selected_lines = wildprompt_line_selections.get(wildprompt_selection, None)
            if isinstance(selected_lines, list) and len(selected_lines) == 0:
                continue
            lines = selected_lines if isinstance(selected_lines, list) else \
                _load_wildprompt_lines(wildprompt_selection)
            lines = [x.strip() for x in lines if isinstance(x, str) and x.strip() != '']
            if len(lines) > 0:
                names.append(wildprompt_selection)
                prompt_groups.append(lines)
        except Exception:
            print(f'[Wildprompts] Warning: {wildprompt_selection}.txt missing or empty.')

    if len(prompt_groups) == 0:
        return []

    return [dict(zip(names, parts)) for parts in itertools.product(*prompt_groups)]


def get_wildprompt_separate_entries(wildprompt_selections, wildprompt_line_selections=None):
    wildprompt_selections = wildprompt_selections if isinstance(wildprompt_selections, list) else []
    wildprompt_line_selections = wildprompt_line_selections if isinstance(wildprompt_line_selections, dict) else {}
    entries = []

    for wildprompt_selection in wildprompt_selections:
        try:
            selected_lines = wildprompt_line_selections.get(wildprompt_selection, None)
            if isinstance(selected_lines, list) and len(selected_lines) == 0:
                continue
            lines = selected_lines if isinstance(selected_lines, list) else \
                _load_wildprompt_lines(wildprompt_selection)
            lines = [x.strip() for x in lines if isinstance(x, str) and x.strip() != '']
            entries.extend([
                {'name': wildprompt_selection, 'prompt': line}
                for line in lines
            ])
        except Exception:
            print(f'[Wildprompts] Warning: {wildprompt_selection}.txt missing or empty.')

    return entries


def get_all_wildprompts(wildprompt_selections, wildprompt_line_selections=None, use_line_selections=True):
    wildprompt_line_selections = wildprompt_line_selections if isinstance(wildprompt_line_selections, dict) else {}
    prompt_groups = []

    for wildprompt_selection in wildprompt_selections:
        try:
            selected_lines = wildprompt_line_selections.get(wildprompt_selection, None) \
                if use_line_selections else None
            if isinstance(selected_lines, list) and len(selected_lines) == 0:
                continue
            lines = selected_lines if isinstance(selected_lines, list) else \
                _load_wildprompt_lines(wildprompt_selection)
            lines = [x.strip() for x in lines if isinstance(x, str) and x.strip() != '']
            if len(lines) > 0:
                prompt_groups.append(lines)
        except Exception:
            print(f'[Wildprompts] Warning: {wildprompt_selection}.txt missing or empty.')

    if len(prompt_groups) == 0:
        return []

    return [', '.join(parts) for parts in itertools.product(*prompt_groups)]


def get_words(arrays, total_mult, index):
    if len(arrays) == 1:
        return [arrays[0].split(',')[index]]
    else:
        words = arrays[0].split(',')
        word = words[index % len(words)]
        index -= index % len(words)
        index /= len(words)
        index = math.floor(index)
        return [word] + get_words(arrays[1:], math.floor(total_mult / len(words)), index)


def apply_arrays(text, index):
    arrays = re.findall(r'\[\[(.*?)\]\]', text)
    if len(arrays) == 0:
        return text

    print(f'[Arrays] processing: {text}')
    mult = 1
    for arr in arrays:
        words = arr.split(',')
        mult *= len(words)
    
    index %= mult
    chosen_words = get_words(arrays, mult, index)
    
    i = 0
    for arr in arrays:
        text = text.replace(f'[[{arr}]]', chosen_words[i], 1)   
        i = i+1
    
    return text


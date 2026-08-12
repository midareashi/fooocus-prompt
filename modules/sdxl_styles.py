import os
import re
import json
import math

from modules.extra_utils import get_files_from_folder
from random import Random

# cannot use modules.config - validators causing circular imports
styles_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sdxl_styles/'))
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

style_keys = list(styles.keys())
fooocus_expansion = 'Fooocus V2'
random_style_name = 'Random Style'
legal_style_names = [fooocus_expansion, random_style_name] + style_keys


def get_legal_wildprompt_names():
    wildprompt_files = get_files_from_folder(wildprompts_path, ['.txt'])
    return [os.path.splitext(file)[0] for file in wildprompt_files]


legal_wildprompt_names = get_legal_wildprompt_names()


def get_random_style(rng: Random) -> str:
    return rng.choice(list(styles.items()))[0]


def apply_style(style, positive):
    p, n = styles[style]
    return p.replace('{prompt}', positive).splitlines(), n.splitlines(), '{prompt}' in p


def _load_wildprompt_lines(wildprompt_selection):
    with open(os.path.join(wildprompts_path, f'{wildprompt_selection}.txt'), encoding='utf-8') as f:
        return [x for x in f.read().splitlines() if x.strip() != '']


def load_wildprompt_lines(wildprompt_selection):
    try:
        return _load_wildprompt_lines(wildprompt_selection)
    except Exception:
        return []


def apply_wildprompts(wildprompt_selections, rng, wildprompt_line_selections=None):
    prompts = []
    wildprompt_line_selections = wildprompt_line_selections if isinstance(wildprompt_line_selections, dict) else {}

    for wildprompt_selection in wildprompt_selections:
        try:
            selected_lines = wildprompt_line_selections.get(wildprompt_selection, None)
            if isinstance(selected_lines, list) and len(selected_lines) == 0:
                continue
            wildprompt_lines = selected_lines if isinstance(selected_lines, list) else _load_wildprompt_lines(wildprompt_selection)
            wildprompt_lines = [x for x in wildprompt_lines if isinstance(x, str) and x.strip() != '']
            assert len(wildprompt_lines) > 0
            prompts.append(rng.choice(wildprompt_lines))
        except Exception:
            print(f'[Wildprompts] Warning: {wildprompt_selection}.txt missing or empty.')

    return ', '.join(prompts)


def get_all_wildprompts(wildprompt_selections, wildprompt_line_selections=None, use_line_selections=True):
    prompts = []
    wildprompt_line_selections = wildprompt_line_selections if isinstance(wildprompt_line_selections, dict) else {}

    if len(wildprompt_selections) != 1:
        return prompts

    try:
        selected_lines = wildprompt_line_selections.get(wildprompt_selections[0], None) if use_line_selections else None
        if isinstance(selected_lines, list) and len(selected_lines) == 0:
            return prompts
        prompts.extend(selected_lines if isinstance(selected_lines, list) else _load_wildprompt_lines(wildprompt_selections[0]))
        prompts = [x for x in prompts if isinstance(x, str) and x.strip() != '']
    except Exception:
        print(f'[Wildprompts] Warning: {wildprompt_selections[0]}.txt missing or empty.')

    return prompts


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


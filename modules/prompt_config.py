import json
import os
import re

import modules.config


PROMPT_CONFIG_DIR = os.path.abspath(os.path.join('input', 'prompt_configs'))


def _ensure_dir():
    os.makedirs(PROMPT_CONFIG_DIR, exist_ok=True)


def _safe_name(name: str | None, fallback: str = 'prompt-config') -> str:
    name = (name or '').strip()
    if name == '':
        name = fallback
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip(' .')
    return name[:80] or fallback


def list_prompt_configs() -> list[str]:
    _ensure_dir()
    names = []
    for filename in os.listdir(PROMPT_CONFIG_DIR):
        if filename.lower().endswith('.json'):
            names.append(os.path.splitext(filename)[0])
    return sorted(names, key=str.casefold)


def save_prompt_config(name: str | None, config_data: dict) -> str:
    base_name = _safe_name(name, _safe_name(config_data.get('prompt'), 'prompt-config'))
    _ensure_dir()
    path = os.path.join(PROMPT_CONFIG_DIR, f'{base_name}.json')

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    return base_name


def load_prompt_config(name: str | None) -> dict:
    if name is None or name == '':
        return {}

    path = os.path.join(PROMPT_CONFIG_DIR, f'{_safe_name(name)}.json')
    if not os.path.exists(path):
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data if isinstance(data, dict) else {}


def delete_prompt_config(name: str | None) -> bool:
    if name is None or name == '':
        return False

    path = os.path.join(PROMPT_CONFIG_DIR, f'{_safe_name(name)}.json')
    if not os.path.exists(path):
        return False

    os.remove(path)
    return True

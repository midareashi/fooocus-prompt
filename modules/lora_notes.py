import os
import re


LORA_NOTES_DIR = os.path.abspath(os.path.join('input', 'lora'))


def _ensure_dir():
    os.makedirs(LORA_NOTES_DIR, exist_ok=True)


def _safe_name(model_name: str | None) -> str:
    name = str(model_name or '').strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip(' .')
    return name[:160]


def _note_path(model_name: str | None) -> str | None:
    name = _safe_name(model_name)
    if name == '' or name == 'None':
        return None
    return os.path.join(LORA_NOTES_DIR, f'{name}.txt')


def load_lora_note(model_name: str | None) -> str:
    path = _note_path(model_name)
    if path is None or not os.path.exists(path):
        return ''

    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def save_lora_note(model_name: str | None, note: str | None) -> str:
    path = _note_path(model_name)
    if path is None:
        return ''

    note = str(note or '').strip()
    if note == '':
        if os.path.exists(path):
            os.remove(path)
        return ''

    _ensure_dir()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(note)
    return note

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


COMFY_ROOT = Path(os.getenv('FOOOCUS_COMFY_ROOT', r'C:\Comfy\ComfyUI'))
COMFY_SERVER = os.getenv('FOOOCUS_COMFY_SERVER', 'http://127.0.0.1:8188').rstrip('/')
WAN_WORKFLOW_PATH = Path(os.getenv(
    'FOOOCUS_WAN_WORKFLOW',
    str(COMFY_ROOT / 'user' / 'default' / 'workflows' / 'DasiwaWan.json')
))
WAN_API_WORKFLOW_PATH = Path(os.getenv(
    'FOOOCUS_WAN_API_WORKFLOW',
    str(COMFY_ROOT / 'user' / 'default' / 'workflows' / 'DasiwaWan.api.json')
))
COMFY_UNET_DIR = COMFY_ROOT / 'models' / 'unet'
COMFY_LORA_DIR = COMFY_ROOT / 'models' / 'loras'
DEFAULT_WAN_HIGH_MODEL = 'DasiwaWAN22I2V14BLightspeed_snatchkissHighV11.safetensors'
DEFAULT_WAN_LOW_MODEL = 'DasiwaWAN22I2V14BLightspeed_snatchkissLowV11.safetensors'


def _save_numpy_image(image, prefix):
    if image is None:
        return None

    input_dir = COMFY_ROOT / 'input'
    input_dir.mkdir(parents=True, exist_ok=True)

    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError('Video input image must be an RGB image.')
    if array.shape[2] > 3:
        array = array[:, :, :3]

    filename = f'fooocus_{prefix}_{time.strftime("%Y%m%d_%H%M%S")}.png'
    Image.fromarray(array.astype('uint8')).save(input_dir / filename)
    return filename


def _set_node_widgets(workflow, node_id, updates):
    for node in workflow.get('nodes', []):
        if node.get('id') == node_id:
            widgets = node.setdefault('widgets_values', [])
            for index, value in updates.items():
                while len(widgets) <= index:
                    widgets.append(None)
                widgets[index] = value
            return


def _set_lora_enabled(workflow, node_id, enabled):
    for node in workflow.get('nodes', []):
        if node.get('id') == node_id:
            for widget in node.get('widgets_values', []):
                if isinstance(widget, dict) and 'on' in widget:
                    widget['on'] = bool(enabled)


def _set_lora_selections(workflow, node_id, selected_loras):
    selected_loras = set(selected_loras or [])
    for node in workflow.get('nodes', []):
        if node.get('id') == node_id:
            for widget in node.get('widgets_values', []):
                if isinstance(widget, dict) and 'lora' in widget:
                    widget['on'] = widget.get('lora') in selected_loras


def _list_relative_model_files(root):
    if not root.exists():
        return []
    extensions = {'.safetensors', '.gguf', '.ckpt', '.pt'}
    files = []
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(str(path.relative_to(root)).replace('/', '\\'))
    return sorted(files, key=str.lower)


def list_wan_high_low_models():
    files = _list_relative_model_files(COMFY_UNET_DIR)
    wan_files = [x for x in files if 'wan' in x.lower()]
    return wan_files or files


def list_wan_high_loras():
    files = _list_relative_model_files(COMFY_LORA_DIR)
    return [x for x in files if x.lower().startswith('img2vid\\high\\')]


def list_wan_low_loras():
    files = _list_relative_model_files(COMFY_LORA_DIR)
    return [x for x in files if x.lower().startswith('img2vid\\low\\')]


def _sync_subgraph_defaults(workflow, seconds, fps, steps, headroom, resolution):
    for subgraph in workflow.get('definitions', {}).get('subgraphs', []):
        if subgraph.get('id') != 'e0940f57-80b2-480e-8a38-36968dc1763d':
            continue
        nodes = subgraph.get('nodes', [])
        for node in nodes:
            node_id = node.get('id')
            widgets = node.setdefault('widgets_values', [])
            if node_id == 1668 and widgets:
                widgets[0] = int(seconds)
            elif node_id == 1669 and widgets:
                widgets[0] = float(fps)
            elif node_id == 1671 and len(widgets) >= 5:
                widgets[0] = int(steps)
                widgets[1] = max(1, int(steps) // 2)
                widgets[2] = 1
                widgets[3] = 'euler'
                widgets[4] = 'linear_quadratic'
            elif node_id == 2264 and widgets:
                widgets[0] = int(headroom)
            elif node_id == 2531 and widgets:
                widgets[0] = resolution


def _comfy_server_available():
    try:
        with urllib.request.urlopen(f'{COMFY_SERVER}/system_stats', timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def prepare_wan_img2vid(first_frame, last_frame, prompt, negative_prompt, seconds, fps,
                        resolution, headroom, high_model, low_model, high_loras, low_loras):
    if first_frame is None:
        return None, '<div class="error">Add a first frame image before generating video.</div>'
    if not WAN_WORKFLOW_PATH.exists():
        return None, f'<div class="error">Wan workflow not found: {WAN_WORKFLOW_PATH}</div>'

    first_filename = _save_numpy_image(first_frame, 'first')
    last_filename = _save_numpy_image(last_frame, 'last') if last_frame is not None else first_filename

    with open(WAN_WORKFLOW_PATH, 'r', encoding='utf-8-sig') as f:
        workflow = json.load(f)

    _set_node_widgets(workflow, 23, {0: first_filename})
    _set_node_widgets(workflow, 24, {0: last_filename})
    _set_node_widgets(workflow, 2368, {0: str(prompt or '').strip()})
    _set_node_widgets(workflow, 2371, {0: str(negative_prompt or '').strip()})

    steps = 4
    _set_node_widgets(workflow, 1512, {
        2: high_model or DEFAULT_WAN_HIGH_MODEL,
        3: low_model or DEFAULT_WAN_LOW_MODEL,
        15: int(headroom),
        28: int(seconds),
        29: float(fps),
        30: steps,
        31: max(1, steps // 2),
        32: 1,
        33: 'euler',
        34: 'linear_quadratic',
        35: resolution,
        36: False,
    })
    _set_lora_selections(workflow, 26, high_loras)
    _set_lora_selections(workflow, 18, low_loras)
    _sync_subgraph_defaults(workflow, seconds, fps, steps, headroom, resolution)

    with open(WAN_WORKFLOW_PATH, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2)

    server_note = 'ComfyUI is reachable.' if _comfy_server_available() else 'ComfyUI is not reachable at the configured URL.'
    api_note = (
        f'Automatic queueing is ready once an API-format workflow exists at {WAN_API_WORKFLOW_PATH}.'
        if not WAN_API_WORKFLOW_PATH.exists()
        else f'API workflow found at {WAN_API_WORKFLOW_PATH}, but queue patching is not wired yet.'
    )

    status = (
        '<div>'
        '<b>Wan img2vid workflow prepared.</b><br>'
        f'First frame: {first_filename}<br>'
        f'Last frame: {last_filename}<br>'
        f'Settings: {seconds}s, {fps} FPS, {resolution}, headroom {headroom}, '
        f'high model {high_model or DEFAULT_WAN_HIGH_MODEL}, low model {low_model or DEFAULT_WAN_LOW_MODEL}.<br>'
        f'High LoRAs: {", ".join(high_loras or []) or "none"}<br>'
        f'Low LoRAs: {", ".join(low_loras or []) or "none"}<br>'
        f'{server_note}<br>'
        f'{api_note}<br>'
        'Open ComfyUI, reload DasiwaWan.json, and queue the workflow.'
        '</div>'
    )
    return None, status

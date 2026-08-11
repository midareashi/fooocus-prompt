import os
import args_manager
import modules.config
import json
import urllib.parse
import ast
import html

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from modules.flags import OutputFormat
from modules.meta_parser import MetadataParser, get_exif
from modules.util import generate_temp_filename

log_cache = {}


def _metadata_value(metadata, key, default=''):
    for _, metadata_key, value in metadata:
        if metadata_key == key:
            return value
    return default


def _metadata_list_value(metadata, key):
    value = _metadata_value(metadata, key, [])
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def get_current_html_path(output_format=None):
    output_format = output_format if output_format else modules.config.default_output_format
    date_string, local_temp_filename, only_name = generate_temp_filename(folder=modules.config.path_outputs,
                                                                         extension=output_format)
    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')
    return html_name


def log(img, metadata, metadata_parser: MetadataParser | None = None, output_format=None, task=None, persist_image=True) -> str:
    path_outputs = modules.config.temp_path if args_manager.args.disable_image_log or not persist_image else modules.config.path_outputs
    output_format = output_format if output_format else modules.config.default_output_format
    date_string, local_temp_filename, only_name = generate_temp_filename(folder=path_outputs, extension=output_format)
    os.makedirs(os.path.dirname(local_temp_filename), exist_ok=True)

    parsed_parameters = metadata_parser.to_string(metadata.copy()) if metadata_parser is not None else ''
    image = Image.fromarray(img)

    if output_format == OutputFormat.PNG.value:
        if parsed_parameters != '':
            pnginfo = PngInfo()
            pnginfo.add_text('parameters', parsed_parameters)
            pnginfo.add_text('fooocus_scheme', metadata_parser.get_scheme().value)
        else:
            pnginfo = None
        image.save(local_temp_filename, pnginfo=pnginfo)
    elif output_format == OutputFormat.JPEG.value:
        image.save(local_temp_filename, quality=95, optimize=True, progressive=True, exif=get_exif(parsed_parameters, metadata_parser.get_scheme().value) if metadata_parser else Image.Exif())
    elif output_format == OutputFormat.WEBP.value:
        image.save(local_temp_filename, quality=95, lossless=False, exif=get_exif(parsed_parameters, metadata_parser.get_scheme().value) if metadata_parser else Image.Exif())
    else:
        image.save(local_temp_filename)

    if args_manager.args.disable_image_log:
        return local_temp_filename

    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')

    css_styles = (
        "<style>"
        "body { background-color: #121212; color: #E0E0E0; } "
        "a { color: #BB86FC; } "
        ".metadata { border-collapse: collapse; width: 100%; } "
        ".metadata .label { width: 15%; } "
        ".metadata .value { width: 85%; font-weight: bold; } "
        ".metadata th, .metadata td { border: 1px solid #4d4d4d; padding: 4px; } "
        ".image-container img { height: auto; max-width: 512px; display: block; padding-right:10px; } "
        ".image-container div { text-align: center; padding: 4px; } "
        "hr { border-color: gray; } "
        "button, .button-link { background-color: black; color: white; border: 1px solid grey; border-radius: 5px; padding: 5px 10px; text-align: center; display: inline-block; font-size: 16px; cursor: pointer; text-decoration: none; margin-right: 6px; }"
        "button:hover, .button-link:hover {background-color: grey; color: black;}"
        "#filters { display: flex; gap: 24px; flex-wrap: wrap; margin: 12px 0; } "
        "#filters label { display: block; margin: 3px 0; } "
        ".filter-heading { font-weight: bold; margin-bottom: 4px; } "
        "</style>"
    )

    js = (
        """<script>
        function to_clipboard(txt) { 
        txt = decodeURIComponent(txt);
        if (navigator.clipboard && navigator.permissions) {
            navigator.clipboard.writeText(txt)
        } else {
            const textArea = document.createElement('textArea')
            textArea.value = txt
            textArea.style.width = 0
            textArea.style.position = 'fixed'
            textArea.style.left = '-999px'
            textArea.style.top = '10px'
            textArea.setAttribute('readonly', 'readonly')
            document.body.appendChild(textArea)

            textArea.select()
            document.execCommand('copy')
            document.body.removeChild(textArea)
        }
        alert('Copied to Clipboard!\\nPaste to prompt area to load parameters.\\nCurrent clipboard content is:\\n\\n' + txt);
        }
        function updateFilters() {
            const modelFilters = Array.from(document.querySelectorAll('input[name="baseModelFilter"]:checked')).map((x) => x.value);
            const wildpromptFilters = Array.from(document.querySelectorAll('input[name="wildpromptFilter"]:checked')).map((x) => x.value);
            document.querySelectorAll('.image-container').forEach(function(item) {
                const model = item.getAttribute('data-model') || '';
                let wildprompts = [];
                try {
                    wildprompts = JSON.parse(item.getAttribute('data-wildprompts') || '[]');
                } catch (e) {}
                const modelMatch = modelFilters.length === 0 || modelFilters.includes(model);
                const wildpromptMatch = wildpromptFilters.length === 0 || wildpromptFilters.some((x) => wildprompts.includes(x));
                item.style.display = modelMatch && wildpromptMatch ? 'block' : 'none';
            });
        }
        function initFilters() {
            const modelCounts = {};
            const wildpromptCounts = {};
            document.querySelectorAll('.image-container').forEach(function(item) {
                const model = item.getAttribute('data-model') || '';
                if (model) modelCounts[model] = (modelCounts[model] || 0) + 1;
                let wildprompts = [];
                try {
                    wildprompts = JSON.parse(item.getAttribute('data-wildprompts') || '[]');
                } catch (e) {}
                wildprompts.forEach((prompt) => wildpromptCounts[prompt] = (wildpromptCounts[prompt] || 0) + 1);
            });
            function addFilters(containerId, name, counts) {
                const container = document.getElementById(containerId);
                if (!container) return;
                Object.keys(counts).sort().forEach(function(value) {
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.name = name;
                    checkbox.value = value;
                    checkbox.addEventListener('change', updateFilters);
                    const label = document.createElement('label');
                    label.appendChild(checkbox);
                    label.appendChild(document.createTextNode(value + ' (' + counts[value] + ')'));
                    container.appendChild(label);
                });
            }
            addFilters('baseModelFilters', 'baseModelFilter', modelCounts);
            addFilters('wildpromptFilters', 'wildpromptFilter', wildpromptCounts);
        }
        window.addEventListener('load', initFilters);
        </script>"""
    )

    begin_part = f"<!DOCTYPE html><html><head><title>Fooocus Log {date_string}</title>{css_styles}</head><body>{js}<p>Fooocus Log {date_string} (private)</p>\n<p>Metadata is embedded if enabled in the config or developer debug mode. You can find the information for each image in line Metadata Scheme.</p><div id=\"filters\"><div id=\"baseModelFilters\"><div class=\"filter-heading\">Base Model</div></div><div id=\"wildpromptFilters\"><div class=\"filter-heading\">Wildprompts</div></div></div><!--fooocus-log-split-->\n\n"
    end_part = f'\n<!--fooocus-log-split--></body></html>'

    middle_part = log_cache.get(html_name, "")

    if middle_part == "":
        if os.path.exists(html_name):
            existing_split = open(html_name, 'r', encoding='utf-8').read().split('<!--fooocus-log-split-->')
            if len(existing_split) == 3:
                middle_part = existing_split[1]
            else:
                middle_part = existing_split[0]

    div_name = only_name.replace('.', '_')
    base_model = html.escape(str(_metadata_value(metadata, 'base_model')), quote=True)
    wildprompts = html.escape(json.dumps(_metadata_list_value(metadata, 'wildprompts')), quote=True)
    item = f"<div id=\"{div_name}\" class=\"image-container\" data-model=\"{base_model}\" data-wildprompts=\"{wildprompts}\"><hr><table><tr>\n"
    item += f"<td><a href=\"{only_name}\" target=\"_blank\"><img src='{only_name}' onerror=\"this.closest('.image-container').style.display='none';\" loading='lazy'/></a><div>{only_name}</div></td>"
    item += "<td><table class='metadata'>"
    for label, key, value in metadata:
        value_txt = str(value).replace('\n', ' </br> ')
        item += f"<tr><td class='label'>{label}</td><td class='value'>{value_txt}</td></tr>\n"

    if task is not None and 'positive' in task and 'negative' in task:
        full_prompt_details = f"""<details><summary>Positive</summary>{', '.join(task['positive'])}</details>
        <details><summary>Negative</summary>{', '.join(task['negative'])}</details>"""
        item += f"<tr><td class='label'>Full raw prompt</td><td class='value'>{full_prompt_details}</td></tr>\n"

    item += "</table>"

    prompt_config_json = json.dumps({k: v for _, k, v, in metadata}, indent=2)
    js_txt = urllib.parse.quote(prompt_config_json, safe='')
    item += f"</br><button onclick=\"to_clipboard('{js_txt}')\">Copy to Clipboard</button>"
    item += f"<a class=\"button-link\" download=\"{div_name}_prompt_config.json\" href=\"data:application/json;charset=utf-8,{js_txt}\">Save Prompt Config</a>"

    item += "</td>"
    item += "</tr></table></div>\n\n"

    middle_part = item + middle_part

    with open(html_name, 'w', encoding='utf-8') as f:
        f.write(begin_part + middle_part + end_part)

    print(f'Image generated with private log at: {html_name}')

    log_cache[html_name] = middle_part

    return local_temp_filename

import gradio as gr
import modules.localization as localization
import modules.sdxl_styles as sdxl_styles


all_styles = []
all_categories_label = 'All folders'


def _as_list(value):
    return value if isinstance(value, list) else []


def try_load_sorted_styles(style_names=None, default_selected=None):
    global all_styles

    all_styles = sdxl_styles.get_legal_style_layer_names()
    return sdxl_styles.normalize_style_layer_selections(_as_list(default_selected))


def get_style_category(style_name):
    return sdxl_styles.get_style_layer_category(style_name)


def get_style_categories():
    return sorted({get_style_category(name) for name in all_styles}, key=str.casefold)


def filter_styles_by_folders(selected, folders=None, query=''):
    selected = sdxl_styles.normalize_style_layer_selections(_as_list(selected))
    valid_folders = set(get_style_categories())
    folders = [name for name in _as_list(folders) if name in valid_folders]
    query = query.strip().casefold() if isinstance(query, str) else ''
    visible = [
        name for name in all_styles
        if (len(folders) == 0 or get_style_category(name) in folders)
        and (query == '' or query in localization_key(name).casefold())
    ]
    return gr.update(choices=visible, value=selected)


def refresh_style_browser(selected, folders=None, query=''):
    try_load_sorted_styles()
    folder_names = get_style_categories()
    folders = [name for name in _as_list(folders) if name in folder_names]
    return (
        gr.update(choices=folder_names, value=folders),
        filter_styles_by_folders(selected, folders, query),
    )


def reset_style_browser():
    return (
        gr.update(choices=get_style_categories(), value=[]),
        '',
        gr.update(choices=all_styles, value=[]),
    )


def sort_styles(selected):
    return gr.update(value=sdxl_styles.normalize_style_layer_selections(_as_list(selected)))


def localization_key(x):
    return x + localization.current_translation.get(x, '')


def enforce_style_selection_rules(selected, folders=None, query=''):
    return filter_styles_by_folders(selected, folders, query)

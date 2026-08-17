import threading
import time

from extras.inpaint_mask import generate_mask_from_image, SAMOptions
from modules.patch import PatchSettings, patch_settings, patch_all
import modules.config
import modules.history_db
import modules.lora_notes

patch_all()


class AsyncTask:
    def __init__(self, args):
        from modules.flags import Performance, MetadataScheme, ip_list, disabled
        from modules.util import get_enabled_loras
        from modules.config import default_max_lora_number
        from PIL import Image
        import numpy as np
        import json
        import args_manager

        self.args = args.copy()
        self.yields = []
        self.results = []
        self.last_stop = False
        self.processing = False
        self.completed = False

        self.performance_loras = []

        if len(args) == 0:
            return

        def get_uploaded_file_path(file):
            if file is None:
                return None
            if isinstance(file, str):
                return file
            if isinstance(file, dict):
                return file.get('name') or file.get('path')
            return getattr(file, 'name', None)

        def load_uploaded_image(file):
            if isinstance(file, np.ndarray):
                return file
            image_path = get_uploaded_file_path(file)
            if image_path is None:
                return None
            try:
                return np.asarray(Image.open(image_path).convert('RGB'))
            except Exception as e:
                print(f'[Person Likeness] Failed to load image {image_path}: {e}')
                return None

        def flatten_uploaded_files(files):
            if files is None:
                return []
            if isinstance(files, str) and files.strip().startswith('['):
                try:
                    files = json.loads(files)
                except Exception:
                    files = [files]
            if isinstance(files, (str, dict, np.ndarray)) or hasattr(files, 'name'):
                files = [files]
            return [file for file in files if file is not None]

        args.reverse()
        self.generate_image_grid = args.pop()
        self.quick_preview = bool(args.pop())
        self.prompt = args.pop()
        self.negative_prompt = args.pop()
        self.style_selections = args.pop()
        self.wildprompt_selections = args.pop()
        if not isinstance(self.wildprompt_selections, list):
            self.wildprompt_selections = []
        self.wildprompt_generate_all = args.pop()
        try:
            self.wildprompt_line_selections = json.loads(args.pop())
        except Exception:
            self.wildprompt_line_selections = {}
        if not isinstance(self.wildprompt_line_selections, dict):
            self.wildprompt_line_selections = {}

        self.performance_selection = Performance(args.pop())
        self.steps = self.performance_selection.steps()
        self.original_steps = self.steps

        self.aspect_ratios_selection = args.pop()
        self.image_number = args.pop()
        self.output_format = args.pop()
        self.seed = int(args.pop())
        self.read_wildcards_in_order = args.pop()
        self.sharpness = args.pop()
        self.cfg_scale = args.pop()
        self.base_model_name = args.pop()
        self.multi_checkpoint_enabled = bool(args.pop())
        self.multi_checkpoint_model_names = args.pop()
        if not isinstance(self.multi_checkpoint_model_names, list):
            self.multi_checkpoint_model_names = []
        self.multi_checkpoint_model_names = [
            str(x) for x in self.multi_checkpoint_model_names
            if isinstance(x, str) and x != 'None'
        ]
        if not self.multi_checkpoint_enabled or len(self.multi_checkpoint_model_names) == 0:
            self.multi_checkpoint_model_names = [self.base_model_name]
        self.refiner_model_name = args.pop()
        self.refiner_switch = args.pop()
        self.loras = get_enabled_loras([(bool(args.pop()), str(args.pop()), float(args.pop())) for _ in
                                        range(default_max_lora_number)])
        self.input_image_checkbox = args.pop()
        self.current_tab = args.pop()
        self.person_likeness_enabled = args.pop()
        self.person_likeness_class = args.pop()
        self.person_likeness_strength = args.pop()
        self.person_likeness_face_weight = args.pop()
        self.person_likeness_face_start = args.pop()
        self.person_likeness_images = []
        self.person_likeness_prepared_images = None
        self.person_likeness_cache_key = None
        person_likeness_files = flatten_uploaded_files(args.pop())
        for person_file in person_likeness_files:
            person_image = load_uploaded_image(person_file)
            if person_image is not None:
                self.person_likeness_images.append(person_image)
        self.uov_method = args.pop()
        self.uov_input_image = args.pop()
        self.outpaint_selections = args.pop()
        self.inpaint_input_image = args.pop()
        self.inpaint_additional_prompt = args.pop()
        self.inpaint_mask_image_upload = args.pop()

        self.disable_preview = args.pop()
        self.disable_intermediate_results = args.pop()
        self.disable_seed_increment = args.pop()
        self.black_out_nsfw = args.pop()
        self.adm_scaler_positive = args.pop()
        self.adm_scaler_negative = args.pop()
        self.adm_scaler_end = args.pop()
        self.adaptive_cfg = args.pop()
        self.clip_skip = args.pop()
        self.sampler_name = args.pop()
        self.scheduler_name = args.pop()
        self.vae_name = args.pop()
        self.overwrite_step = args.pop()
        if self.quick_preview:
            self.overwrite_step = 10
            self.steps = 10
            self.original_steps = 10
        self.overwrite_switch = args.pop()
        self.overwrite_width = args.pop()
        self.overwrite_height = args.pop()
        self.overwrite_vary_strength = args.pop()
        self.overwrite_upscale_strength = args.pop()
        self.mixing_image_prompt_and_vary_upscale = args.pop()
        self.mixing_image_prompt_and_inpaint = args.pop()
        self.debugging_cn_preprocessor = args.pop()
        self.skipping_cn_preprocessor = args.pop()
        self.canny_low_threshold = args.pop()
        self.canny_high_threshold = args.pop()
        self.refiner_swap_method = args.pop()
        self.controlnet_softness = args.pop()
        self.freeu_enabled = args.pop()
        self.freeu_b1 = args.pop()
        self.freeu_b2 = args.pop()
        self.freeu_s1 = args.pop()
        self.freeu_s2 = args.pop()
        self.debugging_inpaint_preprocessor = args.pop()
        self.inpaint_disable_initial_latent = args.pop()
        self.inpaint_engine = args.pop()
        self.inpaint_strength = args.pop()
        self.inpaint_respective_field = args.pop()
        self.inpaint_advanced_masking_checkbox = args.pop()
        self.invert_mask_checkbox = args.pop()
        self.inpaint_erode_or_dilate = args.pop()
        self.training_mode = args.pop()
        self.testing_mode = bool(args.pop())
        self.testing_loras = args.pop()
        if not isinstance(self.testing_loras, list):
            self.testing_loras = []
        self.testing_loras = [
            str(x) for x in self.testing_loras
            if isinstance(x, str) and x != 'None'
        ]
        if not self.testing_mode or len(self.testing_loras) == 0:
            self.testing_loras = []
        self.testing_lora_name = None
        self.save_final_enhanced_image_only = args.pop() if not args_manager.args.disable_image_log else False
        self.save_metadata_to_images = args.pop() if not args_manager.args.disable_metadata else False
        self.metadata_scheme = MetadataScheme(
            args.pop()) if not args_manager.args.disable_metadata else MetadataScheme.FOOOCUS

        self.cn_tasks = {x: [] for x in ip_list}
        for _ in range(modules.config.default_controlnet_image_count):
            cn_img = args.pop()
            cn_stop = args.pop()
            cn_weight = args.pop()
            cn_type = args.pop()
            if cn_img is not None:
                self.cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])

        self.debugging_dino = args.pop()
        self.dino_erode_or_dilate = args.pop()
        self.debugging_enhance_masks_checkbox = args.pop()

        self.enhance_input_image = args.pop()
        self.enhance_checkbox = args.pop()
        self.enhance_uov_method = args.pop()
        self.enhance_uov_processing_order = args.pop()
        self.enhance_uov_prompt_type = args.pop()
        self.enhance_ctrls = []
        for _ in range(modules.config.default_enhance_tabs):
            enhance_enabled = args.pop()
            enhance_mask_dino_prompt_text = args.pop()
            enhance_prompt = args.pop()
            enhance_negative_prompt = args.pop()
            enhance_mask_model = args.pop()
            enhance_mask_cloth_category = args.pop()
            enhance_mask_sam_model = args.pop()
            enhance_mask_text_threshold = args.pop()
            enhance_mask_box_threshold = args.pop()
            enhance_mask_sam_max_detections = args.pop()
            enhance_inpaint_disable_initial_latent = args.pop()
            enhance_inpaint_engine = args.pop()
            enhance_inpaint_strength = args.pop()
            enhance_inpaint_respective_field = args.pop()
            enhance_inpaint_erode_or_dilate = args.pop()
            enhance_mask_invert = args.pop()
            if enhance_enabled:
                self.enhance_ctrls.append([
                    enhance_mask_dino_prompt_text,
                    enhance_prompt,
                    enhance_negative_prompt,
                    enhance_mask_model,
                    enhance_mask_cloth_category,
                    enhance_mask_sam_model,
                    enhance_mask_text_threshold,
                    enhance_mask_box_threshold,
                    enhance_mask_sam_max_detections,
                    enhance_inpaint_disable_initial_latent,
                    enhance_inpaint_engine,
                    enhance_inpaint_strength,
                    enhance_inpaint_respective_field,
                    enhance_inpaint_erode_or_dilate,
                    enhance_mask_invert
                ])
        self.should_enhance = self.enhance_checkbox and (self.enhance_uov_method != disabled.casefold() or len(self.enhance_ctrls) > 0)
        self.images_to_enhance_count = 0
        self.enhance_stats = {}

async_tasks = []
async_tasks_lock = threading.Lock()
current_task = None
display_tasks = []
display_tasks_lock = threading.Lock()
queue_monitor_lock = threading.Lock()
queue_monitor_running = False
queue_monitor_last_heartbeat = 0.0
QUEUE_MONITOR_STALE_SECONDS = 3.0
next_async_task_id = 1
generated_image_configs = {}
generated_image_configs_lock = threading.Lock()
person_likeness_preprocess_cache = {}
person_likeness_ip_adapter_cache = {}


def append_async_task(task):
    global next_async_task_id
    if getattr(task, 'args', None):
        try:
            task.history_batch_id = modules.history_db.create_batch_from_task(task)
        except Exception as e:
            task.history_batch_id = None
            print(f'[History] Failed to create batch record: {e}')
    with async_tasks_lock:
        task.queue_id = next_async_task_id
        next_async_task_id += 1
        async_tasks.append(task)
        return len(async_tasks)


def pop_async_task():
    with async_tasks_lock:
        if len(async_tasks) == 0:
            return None
        return async_tasks.pop(0)


def get_pending_task_count():
    with async_tasks_lock:
        return len(async_tasks)


def get_current_task():
    return current_task


def _prune_display_tasks_locked():
    active_task = current_task
    display_tasks[:] = [
        task for task in display_tasks
        if task is active_task or len(getattr(task, 'yields', []) or []) > 0
        or not getattr(task, 'completed', False)
    ]
    if len(display_tasks) > 8:
        del display_tasks[:-8]


def register_display_task(task):
    if task is None or not hasattr(task, 'yields'):
        return
    with display_tasks_lock:
        if task not in display_tasks:
            display_tasks.append(task)
        _prune_display_tasks_locked()


def append_task_yield(task, flag, product):
    if task is None or not hasattr(task, 'yields'):
        return

    with display_tasks_lock:
        if task not in display_tasks:
            display_tasks.append(task)
        if flag == 'preview':
            task.yields[:] = [item for item in task.yields if item[0] != 'preview']
        elif flag == 'results':
            task.yields[:] = [item for item in task.yields if item[0] != 'results']
        elif flag == 'finish':
            task.yields[:] = [item for item in task.yields if item[0] != 'finish']
        task.yields.append([flag, product])
        _prune_display_tasks_locked()


def get_task_yield_count(task):
    if task is None or not hasattr(task, 'yields'):
        return 0

    with display_tasks_lock:
        return len(task.yields)


def has_display_yields():
    with display_tasks_lock:
        _prune_display_tasks_locked()
        return any(len(getattr(task, 'yields', []) or []) > 0 for task in display_tasks)


def get_latest_display_yield(preferred_task=None, required_flag=None, same_task_only=False):
    with display_tasks_lock:
        candidates = []
        if preferred_task is not None and hasattr(preferred_task, 'yields'):
            candidates.append(preferred_task)

        if not same_task_only:
            for task in display_tasks:
                if task not in candidates:
                    candidates.append(task)
            if current_task is not None and current_task not in candidates:
                candidates.append(current_task)

        for task in candidates:
            if task is None or not hasattr(task, 'yields') or len(task.yields) == 0:
                continue
            for flag, product in reversed(task.yields):
                if required_flag is not None and flag != required_flag:
                    continue
                _prune_display_tasks_locked()
                return task, flag, product

        _prune_display_tasks_locked()
        return None


def get_task_summary(task):
    if task is None or not hasattr(task, 'prompt'):
        return {}

    total_images = int(getattr(task, 'image_number', 1) or 1)
    checkpoint_count = len(getattr(task, 'multi_checkpoint_model_names', []) or [])
    if checkpoint_count > 1:
        total_images *= checkpoint_count
    testing_lora_count = len(getattr(task, 'testing_loras', []) or [])
    if testing_lora_count > 1:
        total_images *= testing_lora_count

    wildprompt_selections = getattr(task, 'wildprompt_selections', []) or []
    if getattr(task, 'wildprompt_generate_all', False) and len(wildprompt_selections) == 1:
        try:
            from modules.sdxl_styles import get_all_wildprompts
            total_images *= max(1, len(get_all_wildprompts(
                wildprompt_selections,
                getattr(task, 'wildprompt_line_selections', {})
            )))
        except Exception:
            pass

    prompt = str(getattr(task, 'prompt', '') or '').splitlines()[0].strip()
    if len(prompt) > 96:
        prompt = prompt[:93] + '...'

    steps = int(getattr(task, 'steps', 0) or 0)

    return {
        'id': getattr(task, 'queue_id', 0),
        'prompt': prompt or '(empty prompt)',
        'images': total_images,
        'steps': steps,
        'total_steps': total_images * steps,
        'performance': getattr(getattr(task, 'performance_selection', None), 'value', ''),
        'quick_preview': bool(getattr(task, 'quick_preview', False)),
        'processing': bool(getattr(task, 'processing', False)),
    }


def get_queue_snapshot():
    with async_tasks_lock:
        pending = [get_task_summary(task) for task in async_tasks]
    return {
        'active': get_task_summary(current_task)
        if current_task is not None and not getattr(current_task, 'completed', False)
        else None,
        'pending': pending
    }


def remove_pending_task(queue_id):
    try:
        queue_id = int(queue_id)
    except Exception:
        return False

    with async_tasks_lock:
        for index, task in enumerate(async_tasks):
            if getattr(task, 'queue_id', None) == queue_id:
                del async_tasks[index]
                return True
    return False


def clear_pending_tasks():
    with async_tasks_lock:
        removed = len(async_tasks)
        async_tasks.clear()
        return removed


def begin_queue_monitor():
    global queue_monitor_running, queue_monitor_last_heartbeat

    with queue_monitor_lock:
        now = time.time()
        if queue_monitor_running and now - queue_monitor_last_heartbeat < QUEUE_MONITOR_STALE_SECONDS:
            return False
        queue_monitor_running = True
        queue_monitor_last_heartbeat = now
        return True


def heartbeat_queue_monitor():
    global queue_monitor_last_heartbeat

    with queue_monitor_lock:
        if queue_monitor_running:
            queue_monitor_last_heartbeat = time.time()


def end_queue_monitor():
    global queue_monitor_running, queue_monitor_last_heartbeat

    with queue_monitor_lock:
        queue_monitor_running = False
        queue_monitor_last_heartbeat = 0.0


def register_generated_image_config(path, config_data):
    with generated_image_configs_lock:
        generated_image_configs[path] = config_data
        if len(generated_image_configs) > 1000:
            oldest_key = next(iter(generated_image_configs))
            del generated_image_configs[oldest_key]


def get_generated_image_config(path):
    with generated_image_configs_lock:
        return generated_image_configs.get(path, {}).copy()


def is_person_likeness_active(async_task):
    return bool(
        getattr(async_task, 'person_likeness_enabled', False) and
        len(getattr(async_task, 'person_likeness_images', []) or []) > 0
    )


class EarlyReturnException(BaseException):
    pass


def worker():
    global current_task

    import os
    import traceback
    import math
    import numpy as np
    import torch
    import time
    import shared
    import random
    import copy
    import json
    import cv2
    import modules.default_pipeline as pipeline
    import modules.core as core
    import modules.flags as flags
    import modules.patch
    import ldm_patched.modules.model_management
    import extras.preprocessors as preprocessors
    import modules.inpaint_worker as inpaint_worker
    import modules.constants as constants
    import extras.ip_adapter as ip_adapter
    import extras.face_crop
    import fooocus_version
    import hashlib
    import args_manager

    from extras.censor import default_censor
    from modules.sdxl_styles import (apply_style, get_random_style, fooocus_expansion, apply_arrays,
                                     random_style_name, apply_wildprompts, get_all_wildprompts)
    from modules.private_logger import log, write_training_caption
    from extras.expansion import safe_str
    from modules.util import (remove_empty_str, HWC3, resize_image, get_image_shape_ceil, set_image_shape_ceil,
                               get_shape_ceil, resample_image, erode_or_dilate, parse_lora_references_from_prompt,
                               apply_wildcards, join_prompts)
    from modules.upscaler import perform_upscale
    from modules.flags import Performance
    from modules.meta_parser import get_metadata_parser

    pid = os.getpid()
    print(f'Started worker with PID {pid}')

    try:
        async_gradio_app = shared.gradio_root
        flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''
        if async_gradio_app.share:
            flag += f''' or {async_gradio_app.share_url}'''
        print(flag)
    except Exception as e:
        print(e)

    def progressbar(async_task, number, text):
        print(f'[Fooocus] {text}')
        append_task_yield(async_task, 'preview', (number, text, None))

    def yield_result(async_task, imgs, progressbar_index, black_out_nsfw, censor=True, do_not_show_finished_images=False):
        if not isinstance(imgs, list):
            imgs = [imgs]

        if censor and (modules.config.default_black_out_nsfw or black_out_nsfw):
            progressbar(async_task, progressbar_index, 'Checking for NSFW content ...')
            imgs = default_censor(imgs)

        async_task.results = async_task.results + imgs

        if do_not_show_finished_images:
            return

        append_task_yield(async_task, 'results', async_task.results)
        return

    def build_image_wall(async_task):
        results = []

        if len(async_task.results) < 2:
            return

        for img in async_task.results:
            if isinstance(img, str) and os.path.exists(img):
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not isinstance(img, np.ndarray):
                return
            if img.ndim != 3:
                return
            results.append(img)

        H, W, C = results[0].shape

        for img in results:
            Hn, Wn, Cn = img.shape
            if H != Hn:
                return
            if W != Wn:
                return
            if C != Cn:
                return

        cols = float(len(results)) ** 0.5
        cols = int(math.ceil(cols))
        rows = float(len(results)) / float(cols)
        rows = int(math.ceil(rows))

        wall = np.zeros(shape=(H * rows, W * cols, C), dtype=np.uint8)

        for y in range(rows):
            for x in range(cols):
                if y * cols + x < len(results):
                    img = results[y * cols + x]
                    wall[y * H:y * H + H, x * W:x * W + W, :] = img

        # must use deep copy otherwise gradio is super laggy. Do not use list.append() .
        async_task.results = async_task.results + [wall]
        return

    def process_task(all_steps, async_task, callback, controlnet_canny_path, controlnet_cpds_path, current_task_id,
                     denoising_strength, final_scheduler_name, goals, initial_latent, steps, switch, positive_cond,
                     negative_cond, task, loras, tiled, use_expansion, width, height, base_progress, preparation_steps,
                     total_count, show_intermediate_results, persist_image=True):
        if async_task.last_stop is not False:
            ldm_patched.modules.model_management.interrupt_current_processing()
        if 'cn' in goals:
            for cn_flag, cn_path in [
                (flags.cn_canny, controlnet_canny_path),
                (flags.cn_cpds, controlnet_cpds_path)
            ]:
                for cn_img, cn_stop, cn_weight in async_task.cn_tasks[cn_flag]:
                    positive_cond, negative_cond = core.apply_controlnet(
                        positive_cond, negative_cond,
                        pipeline.loaded_ControlNets[cn_path], cn_img, cn_weight, 0, cn_stop)
        imgs = pipeline.process_diffusion(
            positive_cond=positive_cond,
            negative_cond=negative_cond,
            steps=steps,
            switch=switch,
            width=width,
            height=height,
            image_seed=task['task_seed'],
            callback=callback,
            sampler_name=async_task.sampler_name,
            scheduler_name=final_scheduler_name,
            latent=initial_latent,
            denoise=denoising_strength,
            tiled=tiled,
            cfg_scale=async_task.cfg_scale,
            refiner_swap_method=async_task.refiner_swap_method,
            disable_preview=async_task.disable_preview
        )
        del positive_cond, negative_cond  # Save memory
        if inpaint_worker.current_task is not None:
            imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]
        current_progress = int(base_progress + (100 - preparation_steps) / float(all_steps) * steps)
        if modules.config.default_black_out_nsfw or async_task.black_out_nsfw:
            progressbar(async_task, current_progress, 'Checking for NSFW content ...')
            imgs = default_censor(imgs)
        progressbar(async_task, current_progress, f'Saving image {current_task_id + 1}/{total_count} to system ...')
        img_paths = save_and_log(async_task, height, imgs, task, use_expansion, width, loras, persist_image)
        yield_result(async_task, img_paths, current_progress, async_task.black_out_nsfw, False,
                     do_not_show_finished_images=not show_intermediate_results or async_task.disable_intermediate_results)

        return imgs, img_paths, current_progress

    def apply_patch_settings(async_task):
        patch_settings[pid] = PatchSettings(
            async_task.sharpness,
            async_task.adm_scaler_end,
            async_task.adm_scaler_positive,
            async_task.adm_scaler_negative,
            async_task.controlnet_softness,
            async_task.adaptive_cfg
        )

    def save_and_log(async_task, height, imgs, task, use_expansion, width, loras, persist_image=True) -> list:
        img_paths = []
        for x in imgs:
            d = [('Prompt', 'prompt', task['log_positive_prompt']),
                 ('Negative Prompt', 'negative_prompt', task['log_negative_prompt']),
                 ('Fooocus V2 Expansion', 'prompt_expansion', task['expansion']),
                 ('Styles', 'styles',
                 str(task['styles'] if not use_expansion else [fooocus_expansion] + task['styles'])),
                 ('Wildprompts', 'wildprompts', str(task['wildprompts'])),
                 ('Generate All Wildprompts', 'wildprompt_generate_all', async_task.wildprompt_generate_all),
                 ('Wildprompt Line Selections', 'wildprompt_line_selections',
                  json.dumps(async_task.wildprompt_line_selections, ensure_ascii=False)),
                 ('Quick Preview', 'quick_preview', async_task.quick_preview),
                 ('Performance', 'performance', async_task.performance_selection.value),
                 ('Steps', 'steps', async_task.steps),
                 ('Image Number', 'image_number', async_task.image_number),
                 ('Resolution', 'resolution', str((width, height))),
                 ('Guidance Scale', 'guidance_scale', async_task.cfg_scale),
                 ('Sharpness', 'sharpness', async_task.sharpness),
                 ('ADM Guidance', 'adm_guidance', str((
                     modules.patch.patch_settings[pid].positive_adm_scale,
                     modules.patch.patch_settings[pid].negative_adm_scale,
                     modules.patch.patch_settings[pid].adm_scaler_end))),
                 ('Base Model', 'base_model', async_task.base_model_name),
                 ('Refiner Model', 'refiner_model', async_task.refiner_model_name),
                 ('Refiner Switch', 'refiner_switch', async_task.refiner_switch)]

            if async_task.refiner_model_name != 'None':
                if async_task.overwrite_switch > 0:
                    d.append(('Overwrite Switch', 'overwrite_switch', async_task.overwrite_switch))
                if async_task.refiner_swap_method != flags.refiner_swap_method:
                    d.append(('Refiner Swap Method', 'refiner_swap_method', async_task.refiner_swap_method))
            if modules.patch.patch_settings[pid].adaptive_cfg != modules.config.default_cfg_tsnr:
                d.append(
                    ('CFG Mimicking from TSNR', 'adaptive_cfg', modules.patch.patch_settings[pid].adaptive_cfg))

            if async_task.clip_skip > 1:
                d.append(('CLIP Skip', 'clip_skip', async_task.clip_skip))
            d.append(('Sampler', 'sampler', async_task.sampler_name))
            d.append(('Scheduler', 'scheduler', async_task.scheduler_name))
            d.append(('VAE', 'vae', async_task.vae_name))
            d.append(('Seed', 'seed', str(task['task_seed'])))

            if async_task.freeu_enabled:
                d.append(('FreeU', 'freeu',
                          str((async_task.freeu_b1, async_task.freeu_b2, async_task.freeu_s1, async_task.freeu_s2))))

            if is_person_likeness_active(async_task):
                d.append(('Person Likeness', 'person_likeness_enabled', async_task.person_likeness_enabled))
                d.append(('Person Likeness Subject', 'person_likeness_class', async_task.person_likeness_class))
                d.append(('Person Likeness Identity Strength', 'person_likeness_strength',
                          async_task.person_likeness_strength))
                d.append(('Person Likeness Face Weight', 'person_likeness_face_weight',
                          async_task.person_likeness_face_weight))
                d.append(('Person Likeness Face Weight Start At', 'person_likeness_face_start',
                          async_task.person_likeness_face_start))

            for li, (n, w) in enumerate(loras):
                if n != 'None':
                    d.append((f'LoRA {li + 1}', f'lora_combined_{li + 1}', f'{n} : {w}'))
            if async_task.testing_lora_name is not None:
                d.append(('Testing LoRA', 'testing_lora', async_task.testing_lora_name))

            metadata_parser = None
            if async_task.save_metadata_to_images:
                metadata_parser = modules.meta_parser.get_metadata_parser(async_task.metadata_scheme)
                metadata_parser.set_data(task['log_positive_prompt'], task['positive'],
                                         task['log_negative_prompt'], task['negative'],
                                         async_task.steps, async_task.base_model_name, async_task.refiner_model_name,
                                         loras, async_task.vae_name)
            d.append(('Metadata Scheme', 'metadata_scheme',
                      async_task.metadata_scheme.value if async_task.save_metadata_to_images else async_task.save_metadata_to_images))
            d.append(('Training Mode', 'training_mode', async_task.training_mode))
            d.append(('Version', 'version', 'Fooocus v' + fooocus_version.version))
            image_path = log(x, d, metadata_parser, async_task.output_format, task, persist_image)
            if async_task.training_mode:
                write_training_caption(image_path, d)
            register_generated_image_config(image_path, {key: value for _, key, value in d})
            try:
                modules.history_db.record_image(
                    getattr(async_task, 'history_batch_id', None),
                    image_path,
                    d,
                    task=task,
                    loras=loras,
                    width=width,
                    height=height,
                    image_index=task.get('history_image_index')
                )
            except Exception as e:
                print(f'[History] Failed to record image {image_path}: {e}')
            img_paths.append(image_path)

        return img_paths

    def apply_control_nets(async_task, height, ip_adapter_face_path, ip_adapter_path, width, current_progress):
        for task in async_task.cn_tasks[flags.cn_canny]:
            cn_img, cn_stop, cn_weight = task
            if not isinstance(cn_img, np.ndarray):
                continue
            cn_img = resize_image(HWC3(cn_img), width=width, height=height)

            if not async_task.skipping_cn_preprocessor:
                cn_img = preprocessors.canny_pyramid(cn_img, async_task.canny_low_threshold,
                                                     async_task.canny_high_threshold)

            cn_img = HWC3(cn_img)
            task[0] = core.numpy_to_pytorch(cn_img)
            if async_task.debugging_cn_preprocessor:
                yield_result(async_task, cn_img, current_progress, async_task.black_out_nsfw, do_not_show_finished_images=True)
        for task in async_task.cn_tasks[flags.cn_cpds]:
            cn_img, cn_stop, cn_weight = task
            if not isinstance(cn_img, np.ndarray):
                continue
            cn_img = resize_image(HWC3(cn_img), width=width, height=height)

            if not async_task.skipping_cn_preprocessor:
                cn_img = preprocessors.cpds(cn_img)

            cn_img = HWC3(cn_img)
            task[0] = core.numpy_to_pytorch(cn_img)
            if async_task.debugging_cn_preprocessor:
                yield_result(async_task, cn_img, current_progress, async_task.black_out_nsfw, do_not_show_finished_images=True)
        for task in async_task.cn_tasks[flags.cn_ip]:
            cn_img, cn_stop, cn_weight = task
            if isinstance(cn_img, tuple):
                continue
            cn_img = HWC3(cn_img)

            # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
            cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

            task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_path)
            if async_task.debugging_cn_preprocessor:
                yield_result(async_task, cn_img, current_progress, async_task.black_out_nsfw, do_not_show_finished_images=True)
        for task in async_task.cn_tasks[flags.cn_ip_face]:
            cn_img, cn_stop, cn_weight = task
            if isinstance(cn_img, tuple):
                continue
            cn_img = HWC3(cn_img)

            if not async_task.skipping_cn_preprocessor:
                cn_img = extras.face_crop.crop_image(cn_img)

            # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
            cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

            task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_face_path)
            if async_task.debugging_cn_preprocessor:
                yield_result(async_task, cn_img, current_progress, async_task.black_out_nsfw, do_not_show_finished_images=True)
        person_face_tasks = []
        if is_person_likeness_active(async_task) and isinstance(ip_adapter_face_path, str):
            face_weight = max(
                0.0,
                min(
                    modules.config.default_person_likeness_face_weight_max,
                    float(async_task.person_likeness_face_weight)
                )
            )
            if face_weight > 0:
                face_start = max(0.0, min(0.95, float(async_task.person_likeness_face_start)))
                prepared_images = get_prepared_person_likeness_images(async_task, current_progress)
                ip_cache_key = (async_task.person_likeness_cache_key, ip_adapter_face_path)
                if ip_cache_key in person_likeness_ip_adapter_cache:
                    print(f'[Person Likeness] Using cached FaceSwap embeddings for {len(prepared_images)} image(s).')
                    cached_ip_tasks = person_likeness_ip_adapter_cache[ip_cache_key]
                else:
                    cached_ip_tasks = [
                        ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_face_path)
                        for cn_img in prepared_images
                    ]
                    person_likeness_ip_adapter_cache[ip_cache_key] = cached_ip_tasks
                    if len(person_likeness_ip_adapter_cache) > 12:
                        oldest_key = next(iter(person_likeness_ip_adapter_cache))
                        del person_likeness_ip_adapter_cache[oldest_key]
                    print(f'[Person Likeness] Cached FaceSwap embeddings for {len(prepared_images)} image(s).')

                for cached_ip_task in cached_ip_tasks:
                    person_face_tasks.append([cached_ip_task, 0.95, face_weight, face_start, True])

        all_ip_tasks = async_task.cn_tasks[flags.cn_ip] + async_task.cn_tasks[flags.cn_ip_face] + person_face_tasks
        if len(all_ip_tasks) > 0:
            pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)

    def apply_vary(async_task, uov_method, denoising_strength, uov_input_image, switch, current_progress, advance_progress=False):
        if 'subtle' in uov_method:
            denoising_strength = 0.5
        if 'strong' in uov_method:
            denoising_strength = 0.85
        if async_task.overwrite_vary_strength > 0:
            denoising_strength = async_task.overwrite_vary_strength
        shape_ceil = get_image_shape_ceil(uov_input_image)
        if shape_ceil < 1024:
            print(f'[Vary] Image is resized because it is too small.')
            shape_ceil = 1024
        elif shape_ceil > 2048:
            print(f'[Vary] Image is resized because it is too big.')
            shape_ceil = 2048
        uov_input_image = set_image_shape_ceil(uov_input_image, shape_ceil)
        initial_pixels = core.numpy_to_pytorch(uov_input_image)
        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, 'VAE encoding ...')
        candidate_vae, _ = pipeline.get_candidate_vae(
            steps=async_task.steps,
            switch=switch,
            denoise=denoising_strength,
            refiner_swap_method=async_task.refiner_swap_method
        )
        initial_latent = core.encode_vae(vae=candidate_vae, pixels=initial_pixels)
        B, C, H, W = initial_latent['samples'].shape
        width = W * 8
        height = H * 8
        print(f'Final resolution is {str((width, height))}.')
        return uov_input_image, denoising_strength, initial_latent, width, height, current_progress

    def apply_inpaint(async_task, initial_latent, inpaint_head_model_path, inpaint_image,
                      inpaint_mask, inpaint_parameterized, denoising_strength, inpaint_respective_field, switch,
                      inpaint_disable_initial_latent, current_progress, skip_apply_outpaint=False,
                      advance_progress=False):
        if not skip_apply_outpaint:
            inpaint_image, inpaint_mask = apply_outpaint(async_task, inpaint_image, inpaint_mask)

        inpaint_worker.current_task = inpaint_worker.InpaintWorker(
            image=inpaint_image,
            mask=inpaint_mask,
            use_fill=denoising_strength > 0.99,
            k=inpaint_respective_field
        )
        if async_task.debugging_inpaint_preprocessor:
            yield_result(async_task, inpaint_worker.current_task.visualize_mask_processing(), 100,
                         async_task.black_out_nsfw, do_not_show_finished_images=True)
            raise EarlyReturnException

        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, 'VAE Inpaint encoding ...')
        inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
        inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
        inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)
        candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
            steps=async_task.steps,
            switch=switch,
            denoise=denoising_strength,
            refiner_swap_method=async_task.refiner_swap_method
        )
        latent_inpaint, latent_mask = core.encode_vae_inpaint(
            mask=inpaint_pixel_mask,
            vae=candidate_vae,
            pixels=inpaint_pixel_image)
        latent_swap = None
        if candidate_vae_swap is not None:
            if advance_progress:
                current_progress += 1
            progressbar(async_task, current_progress, 'VAE SD15 encoding ...')
            latent_swap = core.encode_vae(
                vae=candidate_vae_swap,
                pixels=inpaint_pixel_fill)['samples']
        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, 'VAE encoding ...')
        latent_fill = core.encode_vae(
            vae=candidate_vae,
            pixels=inpaint_pixel_fill)['samples']
        inpaint_worker.current_task.load_latent(
            latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)
        if inpaint_parameterized:
            pipeline.final_unet = inpaint_worker.current_task.patch(
                inpaint_head_model_path=inpaint_head_model_path,
                inpaint_latent=latent_inpaint,
                inpaint_latent_mask=latent_mask,
                model=pipeline.final_unet
            )
        if not inpaint_disable_initial_latent:
            initial_latent = {'samples': latent_fill}
        B, C, H, W = latent_fill.shape
        height, width = H * 8, W * 8
        final_height, final_width = inpaint_worker.current_task.image.shape[:2]
        print(f'Final resolution is {str((final_width, final_height))}, latent is {str((width, height))}.')

        return denoising_strength, initial_latent, width, height, current_progress

    def apply_outpaint(async_task, inpaint_image, inpaint_mask):
        if len(async_task.outpaint_selections) > 0:
            H, W, C = inpaint_image.shape
            if 'top' in async_task.outpaint_selections:
                inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant',
                                      constant_values=255)
            if 'bottom' in async_task.outpaint_selections:
                inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant',
                                      constant_values=255)

            H, W, C = inpaint_image.shape
            if 'left' in async_task.outpaint_selections:
                inpaint_image = np.pad(inpaint_image, [[0, 0], [int(W * 0.3), 0], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(W * 0.3), 0]], mode='constant',
                                      constant_values=255)
            if 'right' in async_task.outpaint_selections:
                inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(W * 0.3)], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(W * 0.3)]], mode='constant',
                                      constant_values=255)

            inpaint_image = np.ascontiguousarray(inpaint_image.copy())
            inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())
            async_task.inpaint_strength = 1.0
            async_task.inpaint_respective_field = 1.0
        return inpaint_image, inpaint_mask

    def apply_upscale(async_task, uov_input_image, uov_method, switch, current_progress, advance_progress=False):
        H, W, C = uov_input_image.shape
        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, f'Upscaling image from {str((W, H))} ...')
        uov_input_image = perform_upscale(uov_input_image)
        print(f'Image upscaled.')
        if '1.5x' in uov_method:
            f = 1.5
        elif '2x' in uov_method:
            f = 2.0
        else:
            f = 1.0
        shape_ceil = get_shape_ceil(H * f, W * f)
        if shape_ceil < 1024:
            print(f'[Upscale] Image is resized because it is too small.')
            uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
            shape_ceil = 1024
        else:
            uov_input_image = resample_image(uov_input_image, width=W * f, height=H * f)
        image_is_super_large = shape_ceil > 2800
        if 'fast' in uov_method:
            direct_return = True
        elif image_is_super_large:
            print('Image is too large. Directly returned the SR image. '
                  'Usually directly return SR image at 4K resolution '
                  'yields better results than SDXL diffusion.')
            direct_return = True
        else:
            direct_return = False
        if direct_return:
            return direct_return, uov_input_image, None, None, None, None, None, current_progress

        tiled = True
        denoising_strength = 0.382
        if async_task.overwrite_upscale_strength > 0:
            denoising_strength = async_task.overwrite_upscale_strength
        initial_pixels = core.numpy_to_pytorch(uov_input_image)
        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, 'VAE encoding ...')
        candidate_vae, _ = pipeline.get_candidate_vae(
            steps=async_task.steps,
            switch=switch,
            denoise=denoising_strength,
            refiner_swap_method=async_task.refiner_swap_method
        )
        initial_latent = core.encode_vae(
            vae=candidate_vae,
            pixels=initial_pixels, tiled=True)
        B, C, H, W = initial_latent['samples'].shape
        width = W * 8
        height = H * 8
        print(f'Final resolution is {str((width, height))}.')
        return direct_return, uov_input_image, denoising_strength, initial_latent, tiled, width, height, current_progress

    def apply_overrides(async_task, steps, height, width):
        if async_task.overwrite_step > 0:
            steps = async_task.overwrite_step
        switch = int(round(async_task.steps * async_task.refiner_switch))
        if async_task.overwrite_switch > 0:
            switch = async_task.overwrite_switch
        if async_task.overwrite_width > 0:
            width = async_task.overwrite_width
        if async_task.overwrite_height > 0:
            height = async_task.overwrite_height
        return steps, switch, width, height

    def get_image_cache_key(image):
        image = np.ascontiguousarray(HWC3(image))
        digest = hashlib.sha256()
        digest.update(str(image.shape).encode('utf-8'))
        digest.update(str(image.dtype).encode('utf-8'))
        digest.update(image.tobytes())
        return digest.hexdigest()

    def get_person_likeness_cache_key(async_task):
        return (
            bool(async_task.skipping_cn_preprocessor),
            tuple(get_image_cache_key(image) for image in async_task.person_likeness_images)
        )

    def get_prepared_person_likeness_images(async_task, current_progress):
        global person_likeness_preprocess_cache

        if async_task.person_likeness_prepared_images is not None:
            return async_task.person_likeness_prepared_images

        cache_key = get_person_likeness_cache_key(async_task)
        async_task.person_likeness_cache_key = cache_key
        if cache_key in person_likeness_preprocess_cache:
            print(f'[Person Likeness] Using cached face crops for {len(async_task.person_likeness_images)} image(s).')
            prepared = person_likeness_preprocess_cache[cache_key]
            async_task.person_likeness_prepared_images = prepared
            return prepared

        images = []
        for image in async_task.person_likeness_images:
            image = HWC3(image)
            if not async_task.skipping_cn_preprocessor:
                image = extras.face_crop.crop_image(image)
            image = resize_image(image, width=224, height=224, resize_mode=0)
            images.append(np.ascontiguousarray(image))
            if async_task.debugging_cn_preprocessor:
                yield_result(async_task, image, current_progress, async_task.black_out_nsfw,
                             do_not_show_finished_images=True)

        person_likeness_preprocess_cache[cache_key] = images
        if len(person_likeness_preprocess_cache) > 12:
            oldest_key = next(iter(person_likeness_preprocess_cache))
            del person_likeness_preprocess_cache[oldest_key]

        async_task.person_likeness_prepared_images = images
        print(f'[Person Likeness] Cached face crops for {len(images)} image(s).')
        return images

    def prepare_person_likeness_images(async_task, current_progress):
        images = get_prepared_person_likeness_images(async_task, current_progress)
        if len(images) == 0:
            return None

        normalized = [image.astype(np.float32) / 255.0 for image in images]
        return torch.from_numpy(np.ascontiguousarray(np.stack(normalized, axis=0))).float()

    def process_prompt(async_task, prompt, negative_prompt, base_model_additional_loras, image_number, disable_seed_increment, use_expansion, use_style,
                       use_synthetic_refiner, current_progress, advance_progress=False, lora_override=None):
        prompts = remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')
        negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.splitlines()], default='')
        prompt = prompts[0]
        negative_prompt = negative_prompts[0]
        if prompt == '':
            # disable expansion when empty since it is not meaningful and influences image prompt
            use_expansion = False
        extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
        extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []
        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, 'Loading models ...')
        lora_filenames = modules.util.remove_performance_lora(modules.config.lora_filenames,
                                                              async_task.performance_selection)
        prompt_loras = async_task.loras if lora_override is None else lora_override
        loras, prompt = parse_lora_references_from_prompt(prompt, prompt_loras,
                                                          modules.config.default_max_lora_number,
                                                          lora_filenames=lora_filenames)
        loras += async_task.performance_loras
        pipeline.refresh_everything(refiner_model_name=async_task.refiner_model_name,
                                    base_model_name=async_task.base_model_name,
                                    loras=loras, base_model_additional_loras=base_model_additional_loras,
                                    use_synthetic_refiner=use_synthetic_refiner, vae_name=async_task.vae_name)
        pipeline.set_clip_skip(async_task.clip_skip)
        person_likeness_pixels = None
        if is_person_likeness_active(async_task):
            progressbar(async_task, current_progress, 'Downloading PhotoMaker model ...')
            photomaker_path = modules.config.downloading_photomaker()
            progressbar(async_task, current_progress, 'Loading PhotoMaker model ...')
            pipeline.refresh_photomaker_model(photomaker_path)
            person_likeness_pixels = prepare_person_likeness_images(async_task, current_progress)
        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, 'Processing prompts ...')
        tasks = []
        raw_wildprompt_selections = copy.deepcopy(async_task.wildprompt_selections)
        use_wildprompt = len(async_task.wildprompt_selections) > 0
        all_wildprompts = []
        if async_task.wildprompt_generate_all and len(async_task.wildprompt_selections) == 1:
            all_wildprompts = get_all_wildprompts(async_task.wildprompt_selections,
                                                  async_task.wildprompt_line_selections)
            print(f'[Wildprompts] Generate All expanded {async_task.wildprompt_selections[0]} to {len(all_wildprompts)} prompt(s).')

        total_prompts = len(all_wildprompts) * image_number if len(all_wildprompts) > 0 else image_number
        for i in range(total_prompts):
            if disable_seed_increment:
                task_seed = async_task.seed % (constants.MAX_SEED + 1)
            else:
                task_seed = (async_task.seed + i) % (constants.MAX_SEED + 1)  # randint is inclusive, % is not

            task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future
            wildprompt_index = i // image_number
            if len(all_wildprompts) > 0:
                wildprompt_prompt = all_wildprompts[wildprompt_index]
            else:
                wildprompt_prompt = apply_wildprompts(async_task.wildprompt_selections, task_rng,
                                                      async_task.wildprompt_line_selections) if use_wildprompt else ''
            wildprompt_prompt = apply_wildcards(wildprompt_prompt, task_rng, wildprompt_index,
                                                async_task.read_wildcards_in_order)

            task_prompt = apply_wildcards(prompt, task_rng, i, async_task.read_wildcards_in_order)
            task_prompt = apply_arrays(task_prompt, i)
            task_prompt = join_prompts(task_prompt, wildprompt_prompt)
            task_negative_prompt = apply_wildcards(negative_prompt, task_rng, i, async_task.read_wildcards_in_order)
            task_extra_positive_prompts = [apply_wildcards(pmt, task_rng, i, async_task.read_wildcards_in_order) for pmt
                                           in
                                           extra_positive_prompts]
            task_extra_negative_prompts = [apply_wildcards(pmt, task_rng, i, async_task.read_wildcards_in_order) for pmt
                                           in
                                           extra_negative_prompts]

            positive_basic_workloads = []
            negative_basic_workloads = []

            task_styles = async_task.style_selections.copy()
            if use_style:
                placeholder_replaced = False

                for j, s in enumerate(task_styles):
                    if s == random_style_name:
                        s = get_random_style(task_rng)
                        task_styles[j] = s
                    p, n, style_has_placeholder = apply_style(s, positive=task_prompt)
                    if style_has_placeholder:
                        placeholder_replaced = True
                    positive_basic_workloads = positive_basic_workloads + p
                    negative_basic_workloads = negative_basic_workloads + n

                if not placeholder_replaced:
                    positive_basic_workloads = [task_prompt] + positive_basic_workloads
            else:
                positive_basic_workloads.append(task_prompt)

            negative_basic_workloads.append(task_negative_prompt)  # Always use independent workload for negative.

            positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
            negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

            positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
            negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

            tasks.append(dict(
                task_seed=task_seed,
                task_prompt=task_prompt,
                task_negative_prompt=task_negative_prompt,
                positive=positive_basic_workloads,
                negative=negative_basic_workloads,
                expansion='',
                c=None,
                uc=None,
                positive_top_k=len(positive_basic_workloads),
                negative_top_k=len(negative_basic_workloads),
                log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
                log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
                styles=task_styles,
                wildprompts=raw_wildprompt_selections
            ))
        if use_expansion:
            if advance_progress:
                current_progress += 1
            for i, t in enumerate(tasks):

                progressbar(async_task, current_progress, f'Preparing Fooocus text #{i + 1} ...')
                expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
                print(f'[Prompt Expansion] {expansion}')
                t['expansion'] = expansion
                t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.
        if advance_progress:
            current_progress += 1
        for i, t in enumerate(tasks):
            progressbar(async_task, current_progress, f'Encoding positive #{i + 1} ...')
            if person_likeness_pixels is not None:
                t['c'] = pipeline.clip_encode_photomaker(
                    texts=t['positive'],
                    id_pixel_values=person_likeness_pixels,
                    class_name=async_task.person_likeness_class,
                    strength=async_task.person_likeness_strength,
                    pool_top_k=t['positive_top_k']
                )
            else:
                t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])
        if advance_progress:
            current_progress += 1
        for i, t in enumerate(tasks):
            if abs(float(async_task.cfg_scale) - 1.0) < 1e-4:
                t['uc'] = pipeline.clone_cond(t['c'])
            else:
                progressbar(async_task, current_progress, f'Encoding negative #{i + 1} ...')
                t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])
        return tasks, use_expansion, loras, current_progress

    def apply_freeu(async_task):
        print(f'FreeU is enabled!')
        pipeline.final_unet = core.apply_freeu(
            pipeline.final_unet,
            async_task.freeu_b1,
            async_task.freeu_b2,
            async_task.freeu_s1,
            async_task.freeu_s2
        )

    def patch_discrete(unet, scheduler_name):
        return core.opModelSamplingDiscrete.patch(unet, scheduler_name, False)[0]

    def patch_edm(unet, scheduler_name):
        return core.opModelSamplingContinuousEDM.patch(unet, scheduler_name, 120.0, 0.002)[0]

    def patch_samplers(async_task):
        final_scheduler_name = async_task.scheduler_name

        if async_task.scheduler_name in ['lcm', 'tcd']:
            final_scheduler_name = 'sgm_uniform'
            if pipeline.final_unet is not None:
                pipeline.final_unet = patch_discrete(pipeline.final_unet, async_task.scheduler_name)
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = patch_discrete(pipeline.final_refiner_unet, async_task.scheduler_name)

        elif async_task.scheduler_name == 'edm_playground_v2.5':
            final_scheduler_name = 'karras'
            if pipeline.final_unet is not None:
                pipeline.final_unet = patch_edm(pipeline.final_unet, async_task.scheduler_name)
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = patch_edm(pipeline.final_refiner_unet, async_task.scheduler_name)

        return final_scheduler_name

    def set_hyper_sd_defaults(async_task, current_progress, advance_progress=False):
        print('Enter Hyper-SD mode.')
        if advance_progress:
            current_progress += 1
        progressbar(async_task, current_progress, 'Downloading Hyper-SD components ...')
        async_task.performance_loras += [(modules.config.downloading_sdxl_hyper_sd_lora(), 0.8)]
        if async_task.refiner_model_name != 'None':
            print(f'Refiner disabled in Hyper-SD mode.')
        async_task.refiner_model_name = 'None'
        async_task.sampler_name = 'dpmpp_sde_gpu'
        async_task.scheduler_name = 'karras'
        async_task.sharpness = 0.0
        async_task.cfg_scale = 1.0
        async_task.adaptive_cfg = 1.0
        async_task.refiner_switch = 1.0
        async_task.adm_scaler_positive = 1.0
        async_task.adm_scaler_negative = 1.0
        async_task.adm_scaler_end = 0.0
        return current_progress

    def set_lightning_defaults(async_task, current_progress, advance_progress=False):
        print('Enter Lightning mode.')
        if advance_progress:
            current_progress += 1
        progressbar(async_task, 1, 'Downloading Lightning components ...')
        async_task.performance_loras += [(modules.config.downloading_sdxl_lightning_lora(), 1.0)]
        if async_task.refiner_model_name != 'None':
            print(f'Refiner disabled in Lightning mode.')
        async_task.refiner_model_name = 'None'
        async_task.sampler_name = 'euler'
        async_task.scheduler_name = 'sgm_uniform'
        async_task.sharpness = 0.0
        async_task.cfg_scale = 1.0
        async_task.adaptive_cfg = 1.0
        async_task.refiner_switch = 1.0
        async_task.adm_scaler_positive = 1.0
        async_task.adm_scaler_negative = 1.0
        async_task.adm_scaler_end = 0.0
        return current_progress

    def set_lcm_defaults(async_task, current_progress, advance_progress=False):
        print('Enter LCM mode.')
        if advance_progress:
            current_progress += 1
        progressbar(async_task, 1, 'Downloading LCM components ...')
        async_task.performance_loras += [(modules.config.downloading_sdxl_lcm_lora(), 1.0)]
        if async_task.refiner_model_name != 'None':
            print(f'Refiner disabled in LCM mode.')
        async_task.refiner_model_name = 'None'
        async_task.sampler_name = 'lcm'
        async_task.scheduler_name = 'lcm'
        async_task.sharpness = 0.0
        async_task.cfg_scale = 1.0
        async_task.adaptive_cfg = 1.0
        async_task.refiner_switch = 1.0
        async_task.adm_scaler_positive = 1.0
        async_task.adm_scaler_negative = 1.0
        async_task.adm_scaler_end = 0.0
        return current_progress

    def apply_image_input(async_task, base_model_additional_loras, clip_vision_path, controlnet_canny_path,
                          controlnet_cpds_path, goals, inpaint_head_model_path, inpaint_image, inpaint_mask,
                          inpaint_parameterized,  ip_adapter_face_path, ip_adapter_path, ip_negative_path,
                          skip_prompt_processing, use_synthetic_refiner):
        if (async_task.current_tab == 'uov' or (
                async_task.current_tab == 'ip' and async_task.mixing_image_prompt_and_vary_upscale)) \
                and async_task.uov_method != flags.disabled.casefold() and async_task.uov_input_image is not None:
            async_task.uov_input_image, skip_prompt_processing, async_task.steps = prepare_upscale(
                async_task, goals, async_task.uov_input_image, async_task.uov_method, async_task.performance_selection,
                async_task.steps, 1, skip_prompt_processing=skip_prompt_processing)
        if (async_task.current_tab == 'inpaint' or (
                async_task.current_tab == 'ip' and async_task.mixing_image_prompt_and_inpaint)) \
                and isinstance(async_task.inpaint_input_image, dict):
            inpaint_image = async_task.inpaint_input_image['image']
            inpaint_mask = async_task.inpaint_input_image['mask'][:, :, 0]

            if async_task.inpaint_advanced_masking_checkbox:
                if isinstance(async_task.inpaint_mask_image_upload, dict):
                    if (isinstance(async_task.inpaint_mask_image_upload['image'], np.ndarray)
                            and isinstance(async_task.inpaint_mask_image_upload['mask'], np.ndarray)
                            and async_task.inpaint_mask_image_upload['image'].ndim == 3):
                        async_task.inpaint_mask_image_upload = np.maximum(
                            async_task.inpaint_mask_image_upload['image'],
                            async_task.inpaint_mask_image_upload['mask'])
                if isinstance(async_task.inpaint_mask_image_upload,
                              np.ndarray) and async_task.inpaint_mask_image_upload.ndim == 3:
                    H, W, C = inpaint_image.shape
                    async_task.inpaint_mask_image_upload = resample_image(async_task.inpaint_mask_image_upload,
                                                                          width=W, height=H)
                    async_task.inpaint_mask_image_upload = np.mean(async_task.inpaint_mask_image_upload, axis=2)
                    async_task.inpaint_mask_image_upload = (async_task.inpaint_mask_image_upload > 127).astype(
                        np.uint8) * 255
                    inpaint_mask = np.maximum(inpaint_mask, async_task.inpaint_mask_image_upload)

            if int(async_task.inpaint_erode_or_dilate) != 0:
                inpaint_mask = erode_or_dilate(inpaint_mask, async_task.inpaint_erode_or_dilate)

            if async_task.invert_mask_checkbox:
                inpaint_mask = 255 - inpaint_mask

            inpaint_image = HWC3(inpaint_image)
            if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                    and (np.any(inpaint_mask > 127) or len(async_task.outpaint_selections) > 0):
                progressbar(async_task, 1, 'Downloading upscale models ...')
                modules.config.downloading_upscale_model()
                if inpaint_parameterized:
                    progressbar(async_task, 1, 'Downloading inpainter ...')
                    inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                        async_task.inpaint_engine)
                    base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                    print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
                    if async_task.refiner_model_name == 'None':
                        if args_manager.args.fast_inpaint:
                            print('[Inpaint] Fast inpaint enabled; skipping synthetic refiner finishing.')
                            async_task.refiner_switch = 1.0
                        else:
                            use_synthetic_refiner = True
                            async_task.refiner_switch = 0.8
                else:
                    inpaint_head_model_path, inpaint_patch_model_path = None, None
                    print(f'[Inpaint] Parameterized inpaint is disabled.')
                if async_task.inpaint_additional_prompt != '':
                    if async_task.prompt == '':
                        async_task.prompt = async_task.inpaint_additional_prompt
                    else:
                        async_task.prompt = async_task.inpaint_additional_prompt + '\n' + async_task.prompt
                goals.append('inpaint')
        if async_task.current_tab == 'ip' or \
                (is_person_likeness_active(async_task) and float(async_task.person_likeness_face_weight) > 0) or \
                async_task.mixing_image_prompt_and_vary_upscale or \
                async_task.mixing_image_prompt_and_inpaint:
            goals.append('cn')
            progressbar(async_task, 1, 'Downloading control models ...')
            if len(async_task.cn_tasks[flags.cn_canny]) > 0:
                controlnet_canny_path = modules.config.downloading_controlnet_canny()
            if len(async_task.cn_tasks[flags.cn_cpds]) > 0:
                controlnet_cpds_path = modules.config.downloading_controlnet_cpds()
            if len(async_task.cn_tasks[flags.cn_ip]) > 0:
                clip_vision_path, ip_negative_path, ip_adapter_path = modules.config.downloading_ip_adapters('ip')
            if len(async_task.cn_tasks[flags.cn_ip_face]) > 0:
                clip_vision_path, ip_negative_path, ip_adapter_face_path = modules.config.downloading_ip_adapters(
                    'face')
            if is_person_likeness_active(async_task) and float(async_task.person_likeness_face_weight) > 0:
                clip_vision_path, ip_negative_path, ip_adapter_face_path = modules.config.downloading_ip_adapters(
                    'face')
        if async_task.current_tab == 'enhance' and async_task.enhance_input_image is not None:
            goals.append('enhance')
            skip_prompt_processing = True
            async_task.enhance_input_image = HWC3(async_task.enhance_input_image)
        return base_model_additional_loras, clip_vision_path, controlnet_canny_path, controlnet_cpds_path, inpaint_head_model_path, inpaint_image, inpaint_mask, ip_adapter_face_path, ip_adapter_path, ip_negative_path, skip_prompt_processing, use_synthetic_refiner

    def prepare_upscale(async_task, goals, uov_input_image, uov_method, performance, steps, current_progress,
                        advance_progress=False, skip_prompt_processing=False):
        uov_input_image = HWC3(uov_input_image)
        if 'vary' in uov_method:
            goals.append('vary')
        elif 'upscale' in uov_method:
            goals.append('upscale')
            if 'fast' in uov_method:
                skip_prompt_processing = True
                steps = 0
            else:
                steps = performance.steps_uov()

            if advance_progress:
                current_progress += 1
            progressbar(async_task, current_progress, 'Downloading upscale models ...')
            modules.config.downloading_upscale_model()
        return uov_input_image, skip_prompt_processing, steps

    def prepare_enhance_prompt(prompt: str, fallback_prompt: str):
        if safe_str(prompt) == '' or len(remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')) == 0:
            prompt = fallback_prompt

        return prompt

    def stop_processing(async_task, processing_start_time):
        async_task.processing = False
        processing_time = time.perf_counter() - processing_start_time
        print(f'Processing time (total): {processing_time:.2f} seconds')

    def process_enhance(all_steps, async_task, callback, controlnet_canny_path, controlnet_cpds_path,
                        current_progress, current_task_id, denoising_strength, inpaint_disable_initial_latent,
                        inpaint_engine, inpaint_respective_field, inpaint_strength,
                        prompt, negative_prompt, final_scheduler_name, goals, height, img, mask,
                        preparation_steps, steps, switch, tiled, total_count, use_expansion, use_style,
                        use_synthetic_refiner, width, show_intermediate_results=True, persist_image=True):
        base_model_additional_loras = []
        inpaint_head_model_path = None
        inpaint_parameterized = inpaint_engine != 'None'  # inpaint_engine = None, improve detail
        initial_latent = None

        prompt = prepare_enhance_prompt(prompt, async_task.prompt)
        negative_prompt = prepare_enhance_prompt(negative_prompt, async_task.negative_prompt)

        if 'vary' in goals:
            img, denoising_strength, initial_latent, width, height, current_progress = apply_vary(
                async_task, async_task.enhance_uov_method, denoising_strength, img, switch, current_progress)
        if 'upscale' in goals:
            direct_return, img, denoising_strength, initial_latent, tiled, width, height, current_progress = apply_upscale(
                async_task, img, async_task.enhance_uov_method, switch, current_progress)
            if direct_return:
                d = [('Upscale (Fast)', 'upscale_fast', '2x')]
                if modules.config.default_black_out_nsfw or async_task.black_out_nsfw:
                    progressbar(async_task, current_progress, 'Checking for NSFW content ...')
                    img = default_censor(img)
                progressbar(async_task, current_progress, f'Saving image {current_task_id + 1}/{total_count} to system ...')
                uov_image_path = log(img, d, output_format=async_task.output_format, persist_image=persist_image)
                yield_result(async_task, uov_image_path, current_progress, async_task.black_out_nsfw, False,
                             do_not_show_finished_images=not show_intermediate_results or async_task.disable_intermediate_results)
                return current_progress, img, prompt, negative_prompt

        if 'inpaint' in goals and inpaint_parameterized:
            progressbar(async_task, current_progress, 'Downloading inpainter ...')
            inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                inpaint_engine)
            if inpaint_patch_model_path not in base_model_additional_loras:
                base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
        progressbar(async_task, current_progress, 'Preparing enhance prompts ...')
        # positive and negative conditioning aren't available here anymore, process prompt again
        tasks_enhance, use_expansion, loras, current_progress = process_prompt(
            async_task, prompt, negative_prompt, base_model_additional_loras, 1, True,
            use_expansion, use_style, use_synthetic_refiner, current_progress)
        task_enhance = tasks_enhance[0]
        # TODO could support vary, upscale and CN in the future
        # if 'cn' in goals:
        #     apply_control_nets(async_task, height, ip_adapter_face_path, ip_adapter_path, width)
        if async_task.freeu_enabled:
            apply_freeu(async_task)
        patch_samplers(async_task)
        if 'inpaint' in goals:
            denoising_strength, initial_latent, width, height, current_progress = apply_inpaint(
                async_task, None, inpaint_head_model_path, img, mask,
                inpaint_parameterized, inpaint_strength,
                inpaint_respective_field, switch, inpaint_disable_initial_latent,
                current_progress, True)
        imgs, img_paths, current_progress = process_task(all_steps, async_task, callback, controlnet_canny_path,
                                                         controlnet_cpds_path, current_task_id, denoising_strength,
                                                         final_scheduler_name, goals, initial_latent, steps, switch,
                                                         task_enhance['c'], task_enhance['uc'], task_enhance, loras,
                                                         tiled, use_expansion, width, height, current_progress,
                                                         preparation_steps, total_count, show_intermediate_results,
                                                         persist_image)

        del task_enhance['c'], task_enhance['uc']  # Save memory
        return current_progress, imgs[0], prompt, negative_prompt

    def enhance_upscale(all_steps, async_task, base_progress, callback, controlnet_canny_path, controlnet_cpds_path,
                        current_task_id, denoising_strength, done_steps_inpainting, done_steps_upscaling, enhance_steps,
                        prompt, negative_prompt, final_scheduler_name, height, img, preparation_steps, switch, tiled,
                        total_count, use_expansion, use_style, use_synthetic_refiner, width, persist_image=True):
        # reset inpaint worker to prevent tensor size issues and not mix upscale and inpainting
        inpaint_worker.current_task = None

        current_progress = int(base_progress + (100 - preparation_steps) / float(all_steps) * (done_steps_upscaling + done_steps_inpainting))
        goals_enhance = []
        img, skip_prompt_processing, steps = prepare_upscale(
            async_task, goals_enhance, img, async_task.enhance_uov_method, async_task.performance_selection,
            enhance_steps, current_progress)
        steps, _, _, _ = apply_overrides(async_task, steps, height, width)
        exception_result = ''
        if len(goals_enhance) > 0:
            try:
                current_progress, img, prompt, negative_prompt = process_enhance(
                    all_steps, async_task, callback, controlnet_canny_path,
                    controlnet_cpds_path, current_progress, current_task_id, denoising_strength, False,
                    'None', 0.0, 0.0, prompt, negative_prompt, final_scheduler_name,
                    goals_enhance, height, img, None, preparation_steps, steps, switch, tiled, total_count,
                    use_expansion, use_style, use_synthetic_refiner, width, persist_image=persist_image)

            except ldm_patched.modules.model_management.InterruptProcessingException:
                if async_task.last_stop == 'skip':
                    print('User skipped')
                    async_task.last_stop = False
                    # also skip all enhance steps for this image, but add the steps to the progress bar
                    if async_task.enhance_uov_processing_order == flags.enhancement_uov_before:
                        done_steps_inpainting += len(async_task.enhance_ctrls) * enhance_steps
                    exception_result = 'continue'
                else:
                    print('User stopped')
                    exception_result = 'break'
            finally:
                done_steps_upscaling += steps
        return current_task_id, done_steps_inpainting, done_steps_upscaling, img, exception_result

    @torch.no_grad()
    @torch.inference_mode()
    def handler(async_task: AsyncTask):
        preparation_start_time = time.perf_counter()
        async_task.processing = True

        async_task.outpaint_selections = [o.lower() for o in async_task.outpaint_selections]
        base_model_additional_loras = []
        async_task.uov_method = async_task.uov_method.casefold()
        async_task.enhance_uov_method = async_task.enhance_uov_method.casefold()

        if fooocus_expansion in async_task.style_selections:
            use_expansion = True
            async_task.style_selections.remove(fooocus_expansion)
        else:
            use_expansion = False

        use_style = len(async_task.style_selections) > 0

        original_refiner_model_name = async_task.refiner_model_name
        selected_base_model_names = async_task.multi_checkpoint_model_names
        if async_task.multi_checkpoint_enabled:
            print(f'[Parameters] Multi Checkpoint Models = {selected_base_model_names}')
        async_task.base_model_name = selected_base_model_names[0]

        if async_task.base_model_name == async_task.refiner_model_name:
            print(f'Refiner disabled because base model and refiner are same.')
            async_task.refiner_model_name = 'None'

        current_progress = 0
        if async_task.performance_selection == Performance.EXTREME_SPEED:
            set_lcm_defaults(async_task, current_progress, advance_progress=True)
        elif async_task.performance_selection == Performance.LIGHTNING:
            set_lightning_defaults(async_task, current_progress, advance_progress=True)
        elif async_task.performance_selection == Performance.HYPER_SD:
            set_hyper_sd_defaults(async_task, current_progress, advance_progress=True)

        print(f'[Parameters] Adaptive CFG = {async_task.adaptive_cfg}')
        print(f'[Parameters] CLIP Skip = {async_task.clip_skip}')
        print(f'[Parameters] Sharpness = {async_task.sharpness}')
        print(f'[Parameters] ControlNet Softness = {async_task.controlnet_softness}')
        print(f'[Parameters] ADM Scale = '
              f'{async_task.adm_scaler_positive} : '
              f'{async_task.adm_scaler_negative} : '
              f'{async_task.adm_scaler_end}')
        print(f'[Parameters] Seed = {async_task.seed}')

        apply_patch_settings(async_task)

        print(f'[Parameters] CFG = {async_task.cfg_scale}')

        initial_latent = None
        denoising_strength = 1.0
        tiled = False

        width, height = async_task.aspect_ratios_selection.replace('×', ' ').split(' ')[:2]
        width, height = int(width), int(height)

        skip_prompt_processing = False

        inpaint_worker.current_task = None
        inpaint_parameterized = async_task.inpaint_engine != 'None'
        inpaint_image = None
        inpaint_mask = None
        inpaint_head_model_path = None

        use_synthetic_refiner = False

        controlnet_canny_path = None
        controlnet_cpds_path = None
        clip_vision_path, ip_negative_path, ip_adapter_path, ip_adapter_face_path = None, None, None, None

        goals = []
        tasks = []
        current_progress = 1

        if async_task.input_image_checkbox:
            base_model_additional_loras, clip_vision_path, controlnet_canny_path, controlnet_cpds_path, inpaint_head_model_path, inpaint_image, inpaint_mask, ip_adapter_face_path, ip_adapter_path, ip_negative_path, skip_prompt_processing, use_synthetic_refiner = apply_image_input(
                async_task, base_model_additional_loras, clip_vision_path, controlnet_canny_path, controlnet_cpds_path,
                goals, inpaint_head_model_path, inpaint_image, inpaint_mask, inpaint_parameterized, ip_adapter_face_path,
                ip_adapter_path, ip_negative_path, skip_prompt_processing, use_synthetic_refiner)

        # Load or unload CNs
        progressbar(async_task, current_progress, 'Loading control models ...')
        pipeline.refresh_controlnets([controlnet_canny_path, controlnet_cpds_path])
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_face_path)

        async_task.steps, switch, width, height = apply_overrides(async_task, async_task.steps, height, width)

        print(f'[Parameters] Sampler = {async_task.sampler_name} - {async_task.scheduler_name}')
        print(f'[Parameters] Steps = {async_task.steps} - {switch}')

        progressbar(async_task, current_progress, 'Initializing ...')

        loras = async_task.loras
        selected_testing_loras = async_task.testing_loras if async_task.testing_mode else []
        def with_testing_lora(base_loras, testing_lora_name):
            testing_lora_override = base_loras.copy()
            if testing_lora_name is not None and testing_lora_name not in [name for name, _ in testing_lora_override]:
                testing_lora_override.append((testing_lora_name, 1.0))
            return testing_lora_override

        def prompt_with_testing_lora_note(prompt, testing_lora_name):
            prompt = str(prompt or '').strip()
            if testing_lora_name is None:
                return prompt

            lora_note = modules.lora_notes.load_lora_note(testing_lora_name)
            if lora_note == '' or lora_note in prompt:
                return prompt
            return join_prompts(prompt, lora_note)

        initial_lora_override = async_task.loras.copy()
        initial_prompt = async_task.prompt
        if len(selected_testing_loras) > 0:
            async_task.testing_lora_name = selected_testing_loras[0]
            initial_lora_override = with_testing_lora(async_task.loras, selected_testing_loras[0])
            initial_prompt = prompt_with_testing_lora_note(async_task.prompt, selected_testing_loras[0])
        if not skip_prompt_processing:
            tasks, use_expansion, loras, current_progress = process_prompt(async_task, initial_prompt, async_task.negative_prompt,
                                                         base_model_additional_loras, async_task.image_number,
                                                         async_task.disable_seed_increment, use_expansion, use_style, use_synthetic_refiner, current_progress,
                                                         advance_progress=True, lora_override=initial_lora_override)

        if len(goals) > 0:
            current_progress += 1
            progressbar(async_task, current_progress, 'Image processing ...')

        should_enhance = async_task.enhance_checkbox and (async_task.enhance_uov_method != flags.disabled.casefold() or len(async_task.enhance_ctrls) > 0)

        if 'vary' in goals:
            async_task.uov_input_image, denoising_strength, initial_latent, width, height, current_progress = apply_vary(
                async_task, async_task.uov_method, denoising_strength, async_task.uov_input_image, switch,
                current_progress)

        if 'upscale' in goals:
            direct_return, async_task.uov_input_image, denoising_strength, initial_latent, tiled, width, height, current_progress = apply_upscale(
                async_task, async_task.uov_input_image, async_task.uov_method, switch, current_progress,
                advance_progress=True)
            if direct_return:
                d = [('Upscale (Fast)', 'upscale_fast', '2x')]
                if modules.config.default_black_out_nsfw or async_task.black_out_nsfw:
                    progressbar(async_task, 100, 'Checking for NSFW content ...')
                    async_task.uov_input_image = default_censor(async_task.uov_input_image)
                progressbar(async_task, 100, 'Saving image to system ...')
                uov_input_image_path = log(async_task.uov_input_image, d, output_format=async_task.output_format)
                yield_result(async_task, uov_input_image_path, 100, async_task.black_out_nsfw, False,
                             do_not_show_finished_images=True)
                return

        if 'inpaint' in goals:
            try:
                denoising_strength, initial_latent, width, height, current_progress = apply_inpaint(async_task,
                                                                                                    initial_latent,
                                                                                                    inpaint_head_model_path,
                                                                                                    inpaint_image,
                                                                                                    inpaint_mask,
                                                                                                    inpaint_parameterized,
                                                                                                    async_task.inpaint_strength,
                                                                                                    async_task.inpaint_respective_field,
                                                                                                    switch,
                                                                                                    async_task.inpaint_disable_initial_latent,
                                                                                                    current_progress,
                                                                                                    advance_progress=True)
            except EarlyReturnException:
                return

        if 'cn' in goals:
            apply_control_nets(async_task, height, ip_adapter_face_path, ip_adapter_path, width, current_progress)
            if async_task.debugging_cn_preprocessor:
                return

        if async_task.freeu_enabled:
            apply_freeu(async_task)

        # async_task.steps can have value of uov steps here when upscale has been applied
        steps, _, _, _ = apply_overrides(async_task, async_task.steps, height, width)

        images_to_enhance = []
        if 'enhance' in goals:
            async_task.image_number = 1
            images_to_enhance += [async_task.enhance_input_image]
            height, width, _ = async_task.enhance_input_image.shape
            # input image already provided, processing is skipped
            steps = 0
            yield_result(async_task, async_task.enhance_input_image, current_progress, async_task.black_out_nsfw, False,
                         async_task.disable_intermediate_results)

        checkpoint_count = max(1, len(selected_base_model_names))
        testing_lora_count = max(1, len(selected_testing_loras))
        tasks_per_checkpoint = max(1, len(tasks) if not skip_prompt_processing else async_task.image_number)
        all_steps = steps * tasks_per_checkpoint * checkpoint_count * testing_lora_count

        if async_task.enhance_checkbox and async_task.enhance_uov_method != flags.disabled.casefold():
            enhance_upscale_steps = async_task.performance_selection.steps()
            if 'upscale' in async_task.enhance_uov_method:
                if 'fast' in async_task.enhance_uov_method:
                    enhance_upscale_steps = 0
                else:
                    enhance_upscale_steps = async_task.performance_selection.steps_uov()
            enhance_upscale_steps, _, _, _ = apply_overrides(async_task, enhance_upscale_steps, height, width)
            enhance_upscale_steps_total = tasks_per_checkpoint * checkpoint_count * testing_lora_count * enhance_upscale_steps
            all_steps += enhance_upscale_steps_total

        if async_task.enhance_checkbox and len(async_task.enhance_ctrls) != 0:
            enhance_steps, _, _, _ = apply_overrides(async_task, async_task.original_steps, height, width)
            all_steps += tasks_per_checkpoint * checkpoint_count * testing_lora_count * len(async_task.enhance_ctrls) * enhance_steps

        all_steps = max(all_steps, 1)

        print(f'[Parameters] Denoising Strength = {denoising_strength}')

        if isinstance(initial_latent, dict) and 'samples' in initial_latent:
            log_shape = initial_latent['samples'].shape
        else:
            log_shape = f'Image Space {(height, width)}'

        print(f'[Parameters] Initial Latent shape: {log_shape}')

        preparation_time = time.perf_counter() - preparation_start_time
        print(f'Preparation time: {preparation_time:.2f} seconds')

        final_scheduler_name = patch_samplers(async_task)
        print(f'Using {final_scheduler_name} scheduler.')

        append_task_yield(async_task, 'preview', (current_progress, 'Moving model to GPU ...', None))

        processing_start_time = time.perf_counter()

        preparation_steps = current_progress
        total_count = tasks_per_checkpoint * checkpoint_count * testing_lora_count

        def callback(step, x0, x, total_steps, y):
            if step == 0:
                async_task.callback_steps = 0
            async_task.callback_steps += (100 - preparation_steps) / float(all_steps)
            completed_batch_steps = current_task_id * total_steps + step + 1
            total_batch_steps = total_count * total_steps
            append_task_yield(async_task, 'preview', (
                int(current_progress + async_task.callback_steps),
                f'Image {current_task_id + 1}/{total_count} | Step {step + 1}/{total_steps} | Batch step {completed_batch_steps}/{total_batch_steps}',
                y))

        show_intermediate_results = total_count > 1 or async_task.should_enhance
        persist_image = not async_task.should_enhance or not async_task.save_final_enhanced_image_only

        should_stop_generation = False
        for checkpoint_index, checkpoint_model_name in enumerate(selected_base_model_names):
            async_task.base_model_name = checkpoint_model_name
            async_task.refiner_model_name = original_refiner_model_name
            if async_task.base_model_name == async_task.refiner_model_name:
                print(f'Refiner disabled because base model and refiner are same.')
                async_task.refiner_model_name = 'None'

            testing_lora_names = selected_testing_loras if len(selected_testing_loras) > 0 else [None]
            for testing_lora_index, testing_lora_name in enumerate(testing_lora_names):
                async_task.testing_lora_name = testing_lora_name
                testing_lora_override = with_testing_lora(async_task.loras, testing_lora_name)

                if checkpoint_index == 0 and testing_lora_index == 0:
                    checkpoint_tasks = tasks
                    checkpoint_loras = loras
                    checkpoint_scheduler_name = final_scheduler_name
                else:
                    load_label = f'Loading checkpoint {checkpoint_index + 1}/{checkpoint_count}: {checkpoint_model_name}'
                    if testing_lora_name is not None:
                        load_label += f' | Testing LoRA {testing_lora_index + 1}/{testing_lora_count}: {testing_lora_name}'
                    progressbar(async_task, current_progress, load_label + ' ...')
                    checkpoint_prompt = prompt_with_testing_lora_note(async_task.prompt, testing_lora_name)
                    checkpoint_tasks, _, checkpoint_loras, current_progress = process_prompt(
                        async_task, checkpoint_prompt, async_task.negative_prompt,
                        base_model_additional_loras, async_task.image_number,
                        async_task.disable_seed_increment, use_expansion, use_style,
                        use_synthetic_refiner, current_progress, advance_progress=False,
                        lora_override=testing_lora_override)
                    if 'cn' in goals:
                        apply_control_nets(async_task, height, ip_adapter_face_path, ip_adapter_path, width, current_progress)
                    if async_task.freeu_enabled:
                        apply_freeu(async_task)
                    checkpoint_scheduler_name = patch_samplers(async_task)
                    print(f'Using {checkpoint_scheduler_name} scheduler.')

                for task_index, task in enumerate(checkpoint_tasks):
                    current_task_id = checkpoint_index * testing_lora_count * tasks_per_checkpoint + testing_lora_index * tasks_per_checkpoint + task_index
                    task['history_image_index'] = current_task_id
                    progressbar(async_task, current_progress,
                                f'Preparing task {current_task_id + 1}/{total_count} with {checkpoint_model_name} ...')
                    execution_start_time = time.perf_counter()

                    try:
                        imgs, img_paths, current_progress = process_task(
                            all_steps, async_task, callback, controlnet_canny_path,
                            controlnet_cpds_path, current_task_id,
                            denoising_strength, checkpoint_scheduler_name, goals,
                            initial_latent, async_task.steps, switch, task['c'],
                            task['uc'], task, checkpoint_loras, tiled, use_expansion, width,
                            height, current_progress, preparation_steps,
                            total_count, show_intermediate_results,
                            persist_image)

                        completed_tasks = current_task_id + 1
                        current_progress = int(preparation_steps + (100 - preparation_steps) / float(all_steps) * async_task.steps * completed_tasks)
                        images_to_enhance += imgs

                    except ldm_patched.modules.model_management.InterruptProcessingException:
                        if async_task.last_stop == 'skip':
                            print('User skipped')
                            async_task.last_stop = False
                            continue
                        else:
                            print('User stopped')
                            should_stop_generation = True
                            break

                    del task['c'], task['uc']  # Save memory
                    execution_time = time.perf_counter() - execution_start_time
                    print(f'Generating and saving time: {execution_time:.2f} seconds')

                if should_stop_generation:
                    break

            if should_stop_generation:
                break

        if not async_task.should_enhance:
            print(f'[Enhance] Skipping, preconditions aren\'t met')
            stop_processing(async_task, processing_start_time)
            return

        progressbar(async_task, current_progress, 'Processing enhance ...')

        active_enhance_tabs = len(async_task.enhance_ctrls)
        should_process_enhance_uov = async_task.enhance_uov_method != flags.disabled.casefold()
        enhance_uov_before = False
        enhance_uov_after = False
        if should_process_enhance_uov:
            active_enhance_tabs += 1
            enhance_uov_before = async_task.enhance_uov_processing_order == flags.enhancement_uov_before
            enhance_uov_after = async_task.enhance_uov_processing_order == flags.enhancement_uov_after
        total_count = len(images_to_enhance) * active_enhance_tabs
        async_task.images_to_enhance_count = len(images_to_enhance)

        base_progress = current_progress
        current_task_id = -1
        done_steps_upscaling = 0
        done_steps_inpainting = 0
        enhance_steps, _, _, _ = apply_overrides(async_task, async_task.original_steps, height, width)
        exception_result = None
        for index, img in enumerate(images_to_enhance):
            async_task.enhance_stats[index] = 0
            enhancement_image_start_time = time.perf_counter()

            last_enhance_prompt = async_task.prompt
            last_enhance_negative_prompt = async_task.negative_prompt

            if enhance_uov_before:
                current_task_id += 1
                persist_image = not async_task.save_final_enhanced_image_only or active_enhance_tabs == 0
                current_task_id, done_steps_inpainting, done_steps_upscaling, img, exception_result = enhance_upscale(
                    all_steps, async_task, base_progress, callback, controlnet_canny_path, controlnet_cpds_path,
                    current_task_id, denoising_strength, done_steps_inpainting, done_steps_upscaling, enhance_steps,
                    async_task.prompt, async_task.negative_prompt, final_scheduler_name, height, img, preparation_steps,
                    switch, tiled, total_count, use_expansion, use_style, use_synthetic_refiner, width, persist_image)
                async_task.enhance_stats[index] += 1

                if exception_result == 'continue':
                    continue
                elif exception_result == 'break':
                    break

            # inpaint for all other tabs
            for enhance_mask_dino_prompt_text, enhance_prompt, enhance_negative_prompt, enhance_mask_model, enhance_mask_cloth_category, enhance_mask_sam_model, enhance_mask_text_threshold, enhance_mask_box_threshold, enhance_mask_sam_max_detections, enhance_inpaint_disable_initial_latent, enhance_inpaint_engine, enhance_inpaint_strength, enhance_inpaint_respective_field, enhance_inpaint_erode_or_dilate, enhance_mask_invert in async_task.enhance_ctrls:
                current_task_id += 1
                current_progress = int(base_progress + (100 - preparation_steps) / float(all_steps) * (done_steps_upscaling + done_steps_inpainting))
                progressbar(async_task, current_progress, f'Preparing enhancement {current_task_id + 1}/{total_count} ...')
                enhancement_task_start_time = time.perf_counter()
                is_last_enhance_for_image = (current_task_id + 1) % active_enhance_tabs == 0 and not enhance_uov_after
                persist_image = not async_task.save_final_enhanced_image_only or is_last_enhance_for_image

                extras = {}
                if enhance_mask_model == 'sam':
                    print(f'[Enhance] Searching for "{enhance_mask_dino_prompt_text}"')
                elif enhance_mask_model == 'u2net_cloth_seg':
                    extras['cloth_category'] = enhance_mask_cloth_category

                mask, dino_detection_count, sam_detection_count, sam_detection_on_mask_count = generate_mask_from_image(
                    img, mask_model=enhance_mask_model, extras=extras, sam_options=SAMOptions(
                        dino_prompt=enhance_mask_dino_prompt_text,
                        dino_box_threshold=enhance_mask_box_threshold,
                        dino_text_threshold=enhance_mask_text_threshold,
                        dino_erode_or_dilate=async_task.dino_erode_or_dilate,
                        dino_debug=async_task.debugging_dino,
                        max_detections=enhance_mask_sam_max_detections,
                        model_type=enhance_mask_sam_model,
                    ))
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]

                if int(enhance_inpaint_erode_or_dilate) != 0:
                    mask = erode_or_dilate(mask, enhance_inpaint_erode_or_dilate)

                if enhance_mask_invert:
                    mask = 255 - mask

                if async_task.debugging_enhance_masks_checkbox:
                    append_task_yield(async_task, 'preview', (current_progress, 'Loading ...', mask))
                    yield_result(async_task, mask, current_progress, async_task.black_out_nsfw, False,
                                 async_task.disable_intermediate_results)
                    async_task.enhance_stats[index] += 1

                print(f'[Enhance] {dino_detection_count} boxes detected')
                print(f'[Enhance] {sam_detection_count} segments detected in boxes')
                print(f'[Enhance] {sam_detection_on_mask_count} segments applied to mask')

                if enhance_mask_model == 'sam' and (dino_detection_count == 0 or not async_task.debugging_dino and sam_detection_on_mask_count == 0):
                    print(f'[Enhance] No "{enhance_mask_dino_prompt_text}" detected, skipping')
                    continue

                goals_enhance = ['inpaint']

                try:
                    current_progress, img, enhance_prompt_processed, enhance_negative_prompt_processed = process_enhance(
                        all_steps, async_task, callback, controlnet_canny_path, controlnet_cpds_path,
                        current_progress, current_task_id, denoising_strength, enhance_inpaint_disable_initial_latent,
                        enhance_inpaint_engine, enhance_inpaint_respective_field, enhance_inpaint_strength,
                        enhance_prompt, enhance_negative_prompt, final_scheduler_name, goals_enhance, height, img, mask,
                        preparation_steps, enhance_steps, switch, tiled, total_count, use_expansion, use_style,
                        use_synthetic_refiner, width, persist_image=persist_image)
                    async_task.enhance_stats[index] += 1

                    if (should_process_enhance_uov and async_task.enhance_uov_processing_order == flags.enhancement_uov_after
                            and async_task.enhance_uov_prompt_type == flags.enhancement_uov_prompt_type_last_filled):
                        if enhance_prompt_processed != '':
                            last_enhance_prompt = enhance_prompt_processed
                        if enhance_negative_prompt_processed != '':
                            last_enhance_negative_prompt = enhance_negative_prompt_processed

                except ldm_patched.modules.model_management.InterruptProcessingException:
                    if async_task.last_stop == 'skip':
                        print('User skipped')
                        async_task.last_stop = False
                        continue
                    else:
                        print('User stopped')
                        exception_result = 'break'
                        break
                finally:
                    done_steps_inpainting += enhance_steps

                enhancement_task_time = time.perf_counter() - enhancement_task_start_time
                print(f'Enhancement time: {enhancement_task_time:.2f} seconds')

            if exception_result == 'break':
                break

            if enhance_uov_after:
                current_task_id += 1
                # last step in enhance, always save
                persist_image = True
                current_task_id, done_steps_inpainting, done_steps_upscaling, img, exception_result = enhance_upscale(
                    all_steps, async_task, base_progress, callback, controlnet_canny_path, controlnet_cpds_path,
                    current_task_id, denoising_strength, done_steps_inpainting, done_steps_upscaling, enhance_steps,
                    last_enhance_prompt, last_enhance_negative_prompt, final_scheduler_name, height, img,
                    preparation_steps, switch, tiled, total_count, use_expansion, use_style, use_synthetic_refiner,
                    width, persist_image)
                async_task.enhance_stats[index] += 1
                
                if exception_result == 'continue':
                    continue
                elif exception_result == 'break':
                    break

            enhancement_image_time = time.perf_counter() - enhancement_image_start_time
            print(f'Enhancement image time: {enhancement_image_time:.2f} seconds')

        stop_processing(async_task, processing_start_time)
        return

    while True:
        time.sleep(0.01)
        task = pop_async_task()
        if task is not None:
            current_task = task
            register_display_task(task)

            try:
                modules.history_db.update_batch_status(getattr(task, 'history_batch_id', None), 'running')
                with ldm_patched.modules.model_management.interrupt_processing_mutex:
                    ldm_patched.modules.model_management.interrupt_processing = False
                handler(task)
                if task.generate_image_grid:
                    build_image_wall(task)
                task.completed = True
                modules.history_db.update_batch_status(getattr(task, 'history_batch_id', None), 'completed')
                append_task_yield(task, 'finish', task.results)
                pipeline.prepare_text_encoder(async_call=True)
            except:
                traceback.print_exc()
                task.completed = True
                modules.history_db.update_batch_status(getattr(task, 'history_batch_id', None), 'failed')
                append_task_yield(task, 'finish', task.results)
            finally:
                if pid in modules.patch.patch_settings:
                    del modules.patch.patch_settings[pid]
                current_task = None
    pass


threading.Thread(target=worker, daemon=True).start()

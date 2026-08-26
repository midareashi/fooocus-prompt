import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINER_ROOT = Path(os.getenv('FOOOCUS_LORA_TRAINER_ROOT', r'D:\training\tools\sdxl-lora'))
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
TARGET_TRAINING_STEPS = 2000
DEFAULT_EPOCHS = 8
DEFAULT_RANK = 16


@dataclass
class PreparedTrainingRun:
    run_id: str
    source_dir: Path
    trainer_root: Path
    run_dir: Path
    dataset_dir: Path
    output_dir: Path
    log_dir: Path
    preview_dir: Path
    dataset_config: Path
    sample_prompts: Path
    log_file: Path
    lora_name: str
    output_slug: str
    trigger: str
    image_count: int
    repeats: int
    epochs: int
    rank: int
    estimated_steps: int
    skipped_images: list = field(default_factory=list)
    generated_captions: int = 0

    @property
    def expected_model(self):
        return self.output_dir / f'{self.output_slug}.safetensors'


def slugify(value):
    value = str(value or '').strip().lower()
    value = re.sub(r'[^a-z0-9]+', '_', value).strip('_')
    return value[:64]


def normalize_trigger(value, lora_name=''):
    trigger = slugify(value)
    if trigger:
        return trigger
    name_slug = slugify(lora_name)
    return f'person_{name_slug}' if name_slug else ''


def calculate_repeats(image_count, epochs=DEFAULT_EPOCHS, target_steps=TARGET_TRAINING_STEPS):
    image_count = int(image_count)
    epochs = max(1, int(epochs))
    if image_count < 1:
        raise ValueError('The dataset must contain at least one usable image.')
    return max(1, min(30, math.ceil(int(target_steps) / (image_count * epochs))))


def _toml_path(path):
    return str(Path(path).resolve()).replace('\\', '/').replace('"', '\\"')


def _caption_with_trigger(caption, trigger):
    caption = re.sub(r'\s+', ' ', str(caption or '').strip())
    if not caption:
        return f'{trigger}, person, realistic photo'
    first_tag = caption.split(',', 1)[0].strip()
    if first_tag.casefold() == trigger.casefold():
        remainder = caption.split(',', 1)
        return trigger if len(remainder) == 1 else f'{trigger},{remainder[1]}'
    return f'{trigger}, {caption}'


def _validate_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        if min(width, height) < 512:
            return False, f'{path.name}: smaller than 512 pixels on one side ({width}x{height})'
        return True, ''
    except Exception as exc:
        return False, f'{path.name}: unreadable image ({exc})'


def _write_text(path, value):
    path.write_text(str(value).rstrip() + '\n', encoding='utf-8')


def _validate_trainer_root(trainer_root):
    trainer_root = Path(trainer_root).expanduser().resolve()
    required = [
        trainer_root / '.venv' / 'Scripts' / 'python.exe',
        trainer_root / '.venv' / 'Lib' / 'site-packages' / 'accelerate' / '__init__.py',
        trainer_root / 'sd-scripts' / 'sdxl_train_network.py',
        trainer_root / 'models' / 'sdxl-base-1.0' / 'sd_xl_base_1.0.safetensors',
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError('The LoRA trainer is not ready. Missing: ' + '; '.join(missing))
    return trainer_root


def prepare_training_run(source_dir, lora_name, trigger='', trainer_root=DEFAULT_TRAINER_ROOT,
                         epochs=DEFAULT_EPOCHS, rank=DEFAULT_RANK, preview_root=None):
    source_dir = Path(str(source_dir or '').strip().strip('"')).expanduser().resolve()
    if not source_dir.is_dir():
        raise ValueError(f'Image folder does not exist: {source_dir}')

    lora_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(lora_name or '').strip()).strip(' .')
    if not lora_name:
        lora_name = source_dir.name
    output_slug = slugify(lora_name)
    if not output_slug:
        raise ValueError('LoRA name must contain at least one letter or number.')

    trigger = normalize_trigger(trigger, lora_name)
    if not trigger:
        raise ValueError('A trigger word could not be derived from the LoRA name.')

    trainer_root = _validate_trainer_root(trainer_root)
    epochs = max(1, min(30, int(epochs)))
    rank = int(rank)
    if rank not in (8, 16, 32, 64):
        raise ValueError('Network rank must be 8, 16, 32, or 64.')

    image_paths = sorted(
        path for path in source_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f'No supported images were found in {source_dir}.')

    run_id = f'{output_slug}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:6]}'
    run_dir = trainer_root / 'fooocus_runs' / run_id
    dataset_dir = run_dir / 'dataset'
    config_dir = run_dir / 'config'
    output_dir = trainer_root / 'output' / 'fooocus' / run_id
    log_dir = trainer_root / 'logs' / 'fooocus' / run_id
    if preview_root is None:
        preview_root = Path(getattr(__import__('modules.config', fromlist=['path_outputs']), 'path_outputs'))
    preview_dir = Path(preview_root).resolve() / 'lora_training' / run_id
    for directory in (dataset_dir, config_dir, output_dir, log_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=False)

    usable = []
    skipped = []
    for path in image_paths:
        valid, reason = _validate_image(path)
        if valid:
            usable.append(path)
        else:
            skipped.append(reason)
    if not usable:
        raise ValueError('No usable training images remained after validation. ' + '; '.join(skipped))

    repeats = calculate_repeats(len(usable), epochs)
    subset_dir = dataset_dir / f'{repeats}_{trigger}'
    subset_dir.mkdir(parents=True, exist_ok=False)
    generated_captions = 0
    for index, source_image in enumerate(usable, start=1):
        destination_image = subset_dir / f'{index:04d}{source_image.suffix.lower()}'
        shutil.copy2(source_image, destination_image)
        source_caption = source_image.with_suffix('.txt')
        caption = ''
        if source_caption.is_file():
            caption = source_caption.read_text(encoding='utf-8-sig', errors='replace').strip()
        if not caption:
            generated_captions += 1
        _write_text(destination_image.with_suffix('.txt'), _caption_with_trigger(caption, trigger))

    dataset_config = config_dir / 'dataset.toml'
    _write_text(dataset_config, f'''[general]
shuffle_caption = true
caption_extension = ".txt"
keep_tokens = 1

[[datasets]]
resolution = [1024, 1024]
batch_size = 1
enable_bucket = true
bucket_reso_steps = 64
bucket_no_upscale = true

  [[datasets.subsets]]
  image_dir = "{_toml_path(subset_dir)}"
  num_repeats = {repeats}
  class_tokens = "{trigger}"
''')

    sample_prompts = config_dir / 'sample_prompts.txt'
    negative = 'low quality, blurry, distorted face, bad anatomy, text, watermark, logo, signature'
    _write_text(sample_prompts, '\n'.join([
        f'{trigger}, adult woman, close-up portrait, realistic photo, neutral background --n {negative} --w 1024 --h 1024 --d 24701 --l 6.5 --s 28',
        f'{trigger}, adult woman, casual candid portrait, sitting indoors, realistic photo --n {negative} --w 1024 --h 1024 --d 24702 --l 6.5 --s 28',
        f'{trigger}, adult woman, outdoor portrait, natural daylight, realistic photo --n {negative} --w 1024 --h 1024 --d 24703 --l 6.5 --s 28',
        f'{trigger}, adult woman, half body portrait, different outfit, soft indoor light, realistic photo --n {negative} --w 1024 --h 1280 --d 24704 --l 6.5 --s 28',
        f'{trigger}, adult woman, full body portrait, standing outdoors, natural daylight, realistic photo --n {negative} --w 832 --h 1216 --d 24705 --l 6.5 --s 28',
    ]))

    return PreparedTrainingRun(
        run_id=run_id,
        source_dir=source_dir,
        trainer_root=trainer_root,
        run_dir=run_dir,
        dataset_dir=subset_dir,
        output_dir=output_dir,
        log_dir=log_dir,
        preview_dir=preview_dir,
        dataset_config=dataset_config,
        sample_prompts=sample_prompts,
        log_file=log_dir / 'training.log',
        lora_name=lora_name,
        output_slug=output_slug,
        trigger=trigger,
        image_count=len(usable),
        repeats=repeats,
        epochs=epochs,
        rank=rank,
        estimated_steps=len(usable) * repeats * epochs,
        skipped_images=skipped,
        generated_captions=generated_captions,
    )


def build_training_command(run):
    trainer_python = run.trainer_root / '.venv' / 'Scripts' / 'python.exe'
    training_script = run.trainer_root / 'sd-scripts' / 'sdxl_train_network.py'
    model_path = run.trainer_root / 'models' / 'sdxl-base-1.0' / 'sd_xl_base_1.0.safetensors'
    alpha = max(1, run.rank // 2)
    return [
        str(trainer_python), '-m', 'accelerate.commands.launch',
        '--num_processes=1', '--num_machines=1', '--mixed_precision=fp16',
        '--dynamo_backend=no', '--num_cpu_threads_per_process=2',
        str(training_script),
        f'--pretrained_model_name_or_path={model_path}',
        f'--dataset_config={run.dataset_config}',
        f'--output_dir={run.output_dir}',
        f'--output_name={run.output_slug}',
        f'--logging_dir={run.log_dir}',
        '--save_model_as=safetensors', '--save_every_n_epochs=1',
        f'--max_train_epochs={run.epochs}',
        '--network_module=networks.lora', f'--network_dim={run.rank}', f'--network_alpha={alpha}',
        '--learning_rate=6e-5', '--unet_lr=6e-5', '--text_encoder_lr', '5e-6', '5e-6',
        '--optimizer_type=AdamW8bit', '--lr_scheduler=cosine', '--lr_warmup_steps=50',
        '--mixed_precision=fp16', '--no_half_vae', '--save_precision=fp16', '--seed=24601',
        '--max_token_length=225', '--gradient_checkpointing', '--cache_latents',
        '--cache_latents_to_disk', '--xformers', '--max_data_loader_n_workers=0',
        '--noise_offset=0.03', '--min_snr_gamma=5', '--sample_sampler=euler_a',
        '--sample_every_n_epochs=1', f'--sample_prompts={run.sample_prompts}',
    ]


class TrainingController:
    def __init__(self):
        self.lock = threading.Lock()
        self.job_id = None
        self.process = None
        self.stop_requested = False

    def reserve(self, job_id):
        with self.lock:
            if self.job_id is not None:
                return False
            self.job_id = job_id
            self.process = None
            self.stop_requested = False
            return True

    def attach(self, process):
        with self.lock:
            self.process = process

    def finish(self, job_id):
        with self.lock:
            if self.job_id == job_id:
                self.job_id = None
                self.process = None
                self.stop_requested = False

    def stop(self):
        with self.lock:
            process = self.process
            active = self.job_id is not None
            self.stop_requested = active
        if not active:
            return False, 'No LoRA training job is running.'
        if process is None or process.poll() is not None:
            return True, 'Stop requested. The current setup step will finish first.'
        try:
            if os.name == 'nt':
                subprocess.run(
                    ['taskkill', '/PID', str(process.pid), '/T', '/F'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                    check=False,
                )
            else:
                process.terminate()
            return True, 'Stopping the LoRA training process…'
        except Exception as exc:
            return False, f'Could not stop training: {exc}'

    def was_stopped(self):
        with self.lock:
            return self.stop_requested


CONTROLLER = TrainingController()


def _release_fooocus_vram():
    try:
        import ldm_patched.modules.model_management as model_management
        model_management.unload_all_models()
        model_management.soft_empty_cache(True)
        return ''
    except Exception as exc:
        return f'Fooocus VRAM cleanup warning: {exc}'


def _tail_log(path, limit=24000):
    if not path.is_file():
        return ''
    with path.open('rb') as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(max(0, size - limit))
        data = file.read()
    return data.decode('utf-8', errors='replace').replace('\r', '\n').strip()


def _progress_from_log(log_text):
    percentages = re.findall(r'(\d{1,3})%\|', log_text)
    if percentages:
        return min(100, int(percentages[-1]))
    steps = re.findall(r'(\d+)\s*/\s*(\d+)', log_text)
    if steps:
        current, total = map(int, steps[-1])
        if total > 0:
            return min(100, int(current * 100 / total))
    return 0


def _sync_sample_images(run):
    samples = []
    for source in sorted(run.output_dir.rglob('*.png')):
        destination = run.preview_dir / source.name
        if not destination.exists() or destination.stat().st_mtime_ns != source.stat().st_mtime_ns:
            shutil.copy2(source, destination)
        samples.append(str(destination))
    return samples[-20:]


def _versioned_destination(directory, filename):
    destination = directory / filename
    if not destination.exists():
        return destination
    stem = destination.stem
    for version in range(2, 1000):
        candidate = directory / f'{stem}_v{version}{destination.suffix}'
        if not candidate.exists():
            return candidate
    raise RuntimeError(f'Could not find a free output filename for {filename}.')


def _install_finished_lora(run):
    model = run.expected_model
    if not model.is_file():
        candidates = sorted(run.output_dir.glob('*.safetensors'), key=lambda path: path.stat().st_mtime)
        if not candidates:
            raise RuntimeError('Training completed but no LoRA file was produced.')
        model = candidates[-1]

    import modules.config
    destination_dir = Path(modules.config.paths_loras[0]).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = _versioned_destination(destination_dir, f'{run.output_slug}.safetensors')
    shutil.copy2(model, destination)
    import modules.lora_notes
    modules.lora_notes.save_lora_note(
        destination.name,
        f'Trigger: {run.trigger}\nSource: {run.source_dir}\nRun: {run.run_id}',
    )
    modules.config.update_files()
    return destination


def _status_markdown(title, run=None, progress=None, detail=''):
    lines = [f'### {title}']
    if run is not None:
        lines.append(
            f'**{run.lora_name}** · `{run.trigger}` · {run.image_count} images · '
            f'{run.repeats} repeats · {run.epochs} epochs · approximately {run.estimated_steps:,} steps'
        )
    if progress is not None:
        progress = max(0, min(100, int(progress)))
        lines.append(f'`[{"█" * (progress // 5)}{"░" * (20 - progress // 5)}] {progress}%`')
    if detail:
        lines.append(detail)
    return '\n\n'.join(lines)


def run_training_ui(source_dir, lora_name, trigger, trainer_root, epochs, rank):
    import gradio as gr

    job_id = uuid.uuid4().hex
    if not CONTROLLER.reserve(job_id):
        yield (
            _status_markdown('A LoRA training job is already running'), '', [],
            gr.update(interactive=False), gr.update(interactive=True),
        )
        return

    run = None
    try:
        yield (
            _status_markdown('Checking the dataset…'), '', [],
            gr.update(interactive=False), gr.update(interactive=True),
        )
        run = prepare_training_run(source_dir, lora_name, trigger, trainer_root, epochs, rank)
        warning_parts = []
        if run.generated_captions:
            warning_parts.append(f'Created simple fallback captions for {run.generated_captions} images without TXT files.')
        if run.skipped_images:
            warning_parts.append(f'Skipped {len(run.skipped_images)} invalid or undersized images.')
        vram_warning = _release_fooocus_vram()
        if vram_warning:
            warning_parts.append(vram_warning)

        if CONTROLLER.was_stopped():
            yield (
                _status_markdown('Training cancelled before launch', run), '', [],
                gr.update(interactive=True), gr.update(interactive=False),
            )
            return

        command = build_training_command(run)
        environment = os.environ.copy()
        environment['PYTHONUTF8'] = '1'
        environment['HF_HOME'] = str(run.trainer_root / '.hf_cache')
        creation_flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0
        with run.log_file.open('wb') as log_handle:
            command_text = subprocess.list2cmdline(command) if os.name == 'nt' else ' '.join(command)
            log_handle.write(f'Fooocus LoRA training run: {run.run_id}\nCommand: {command_text}\n\n'.encode('utf-8'))
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=str(run.trainer_root),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
            CONTROLLER.attach(process)
            while process.poll() is None:
                log_text = _tail_log(run.log_file)
                progress = _progress_from_log(log_text)
                samples = _sync_sample_images(run)
                detail = ' '.join(warning_parts)
                yield (
                    _status_markdown('Training in progress', run, progress, detail),
                    log_text,
                    samples,
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                )
                time.sleep(1)
            exit_code = process.returncode

        log_text = _tail_log(run.log_file)
        samples = _sync_sample_images(run)
        if CONTROLLER.was_stopped():
            yield (
                _status_markdown('Training stopped', run, _progress_from_log(log_text), 'Partial checkpoints remain in the trainer output folder.'),
                log_text, samples, gr.update(interactive=True), gr.update(interactive=False),
            )
            return
        if exit_code != 0:
            if not log_text.strip():
                log_text = f'The trainer exited with code {exit_code} before producing log output.'
            yield (
                _status_markdown('Training failed', run, _progress_from_log(log_text), f'The trainer exited with code {exit_code}. See the log below.'),
                log_text, samples, gr.update(interactive=True), gr.update(interactive=False),
            )
            return

        installed_model = _install_finished_lora(run)
        yield (
            _status_markdown(
                'Training complete', run, 100,
                f'Installed **{installed_model.name}** in Fooocus. Use **Refresh All Files** in the Models panel if it is not already listed.'
            ),
            log_text,
            samples,
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
    except Exception as exc:
        error_text = f'{type(exc).__name__}: {exc}'
        existing_log = _tail_log(run.log_file) if run is not None else ''
        visible_log = f'{existing_log}\n\n{error_text}'.strip()
        yield (
            _status_markdown('Could not start LoRA training', run, detail=str(exc)),
            visible_log,
            _sync_sample_images(run) if run is not None else [],
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
    finally:
        CONTROLLER.finish(job_id)


def stop_training_ui():
    stopped, message = CONTROLLER.stop()
    title = 'Stop requested' if stopped else 'Nothing to stop'
    return _status_markdown(title, detail=message)


def build_lora_training_ui():
    import gradio as gr

    default_dataset = ROOT / 'datasets' / 'Luna-LoRA-2026-08-25'
    if not default_dataset.is_dir():
        default_dataset = ROOT / 'datasets'

    gr.Markdown(
        'Choose a folder of training images, give the LoRA a name, and press **Train**. '
        'Matching `.txt` captions are preserved; simple captions are created when they are missing.'
    )
    with gr.Row():
        source_dir = gr.Textbox(
            label='Image folder', value=str(default_dataset),
            placeholder=r'D:\path\to\training-images', scale=4,
        )
        lora_name = gr.Textbox(label='LoRA name', value='Luna', scale=1)

    with gr.Accordion(label='Advanced settings', open=False):
        trigger = gr.Textbox(
            label='Trigger word', value='girl_named_luna', placeholder='Automatic, for example person_luna',
            info='Leave blank to derive a unique trigger from the LoRA name. It is kept first in every caption.',
        )
        trainer_root = gr.Textbox(label='Trainer location', value=str(DEFAULT_TRAINER_ROOT))
        with gr.Row():
            epochs = gr.Slider(label='Epochs', minimum=1, maximum=20, step=1, value=DEFAULT_EPOCHS)
            rank = gr.Dropdown(label='Network rank', choices=[8, 16, 32, 64], value=DEFAULT_RANK)

    with gr.Row():
        train_button = gr.Button(value='Train LoRA', variant='primary')
        stop_button = gr.Button(value='Stop', variant='stop', interactive=False)

    status = gr.Markdown(value='### Ready to train')
    samples = gr.Gallery(
        label='Epoch samples', columns=5, rows=2, object_fit='contain',
        height=520, preview=True,
    )
    with gr.Accordion(label='Training log', open=False):
        log = gr.Textbox(label='Live log', lines=20, max_lines=30, interactive=False)

    train_button.click(
        fn=run_training_ui,
        inputs=[source_dir, lora_name, trigger, trainer_root, epochs, rank],
        outputs=[status, log, samples, train_button, stop_button],
        queue=True,
        show_progress=False,
    )
    stop_button.click(
        fn=stop_training_ui,
        outputs=status,
        queue=False,
        show_progress=False,
    )

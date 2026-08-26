import tempfile
import unittest
from pathlib import Path

from PIL import Image

from modules import lora_training


class TestLoraTraining(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / 'source'
        self.trainer = self.root / 'trainer'
        self.previews = self.root / 'previews'
        self.source.mkdir()
        for required in [
            self.trainer / '.venv' / 'Scripts' / 'python.exe',
            self.trainer / '.venv' / 'Lib' / 'site-packages' / 'accelerate' / '__init__.py',
            self.trainer / 'sd-scripts' / 'sdxl_train_network.py',
            self.trainer / 'models' / 'sdxl-base-1.0' / 'sd_xl_base_1.0.safetensors',
        ]:
            required.parent.mkdir(parents=True, exist_ok=True)
            required.write_bytes(b'test')

    def tearDown(self):
        self.temp_dir.cleanup()

    def _image(self, name, size=(1024, 1024)):
        path = self.source / name
        Image.new('RGB', size, '#886644').save(path)
        return path

    def test_slug_trigger_and_repeat_defaults(self):
        self.assertEqual('luna_v2', lora_training.slugify(' Luna V2! '))
        self.assertEqual('person_luna_v2', lora_training.normalize_trigger('', 'Luna V2'))
        self.assertEqual(6, lora_training.calculate_repeats(43, epochs=8))

    def test_existing_caption_keeps_exact_normalized_trigger(self):
        caption = lora_training._caption_with_trigger('Luna, adult woman, red dress', 'luna')
        self.assertEqual('luna, adult woman, red dress', caption)

    def test_prepare_run_preserves_captions_and_creates_fallbacks(self):
        first = self._image('first.png')
        first.with_suffix('.txt').write_text('Luna, adult woman, green dress', encoding='utf-8')
        self._image('second.jpg', (832, 1216))
        self._image('too-small.png', (256, 256))

        run = lora_training.prepare_training_run(
            self.source,
            'Luna Test',
            'luna_test',
            self.trainer,
            epochs=8,
            rank=16,
            preview_root=self.previews,
        )

        self.assertEqual(2, run.image_count)
        self.assertEqual(1, run.generated_captions)
        self.assertEqual(1, len(run.skipped_images))
        self.assertEqual(30, run.repeats)
        captions = sorted(run.dataset_dir.glob('*.txt'))
        self.assertEqual(2, len(captions))
        self.assertTrue(all(path.read_text(encoding='utf-8').startswith('luna_test,') for path in captions))
        self.assertIn('keep_tokens = 1', run.dataset_config.read_text(encoding='utf-8'))

    def test_training_command_uses_isolated_trainer_and_safe_defaults(self):
        self._image('portrait.png')
        run = lora_training.prepare_training_run(
            self.source,
            'Command Test',
            'command_test',
            self.trainer,
            preview_root=self.previews,
        )
        command = lora_training.build_training_command(run)

        self.assertEqual(str(self.trainer / '.venv' / 'Scripts' / 'python.exe'), command[0])
        self.assertEqual(['-m', 'accelerate.commands.launch'], command[1:3])
        self.assertIn('--network_dim=16', command)
        self.assertIn('--learning_rate=6e-5', command)
        self.assertIn('--text_encoder_lr', command)
        self.assertIn(f'--dataset_config={run.dataset_config}', command)

    def test_progress_parser_handles_tqdm_and_step_counts(self):
        self.assertEqual(37, lora_training._progress_from_log('steps: 37%|###| 740/2000'))
        self.assertEqual(25, lora_training._progress_from_log('training 500 / 2000'))


if __name__ == '__main__':
    unittest.main()

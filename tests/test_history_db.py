import os
import pathlib
import shutil
import sys
import tempfile
import time
import urllib.parse
import unittest
from types import SimpleNamespace

from PIL import Image
from PIL.PngImagePlugin import PngInfo

sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
sys.argv = sys.argv[:1]

import modules.config
import modules.history_db as history_db


class TestHistoryDb(unittest.TestCase):
    def setUp(self):
        self.old_outputs = modules.config.path_outputs
        self.temp_dir = tempfile.mkdtemp(prefix='fooocus_history_test_')
        modules.config.path_outputs = self.temp_dir
        history_db._initialized = False

    def tearDown(self):
        modules.config.path_outputs = self.old_outputs
        history_db._initialized = False
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_image(self, path, timestamp):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.new('RGB', (8, 8), (20, 30, 40)).save(path)
        os.utime(path, (timestamp, timestamp))

    def _image_count(self):
        with history_db._connect() as conn:
            return conn.execute('SELECT COUNT(*) AS count FROM images').fetchone()['count']

    def _batch_count(self):
        with history_db._connect() as conn:
            return conn.execute('SELECT COUNT(*) AS count FROM batches').fetchone()['count']

    def _write_log_html(self, folder, filename, config):
        os.makedirs(folder, exist_ok=True)
        payload = urllib.parse.quote(history_db._json_dumps(config), safe='')
        with open(os.path.join(folder, 'log.html'), 'w', encoding='utf-8') as f:
            f.write(
                '<!DOCTYPE html><html><body><!--fooocus-log-split-->'
                f'<div id="{filename.replace(".", "_")}" class="image-container">'
                f'<hr><table><tr><td><a href="{filename}" target="_blank">'
                f'<img src="{filename}" loading="lazy"/></a><div>{filename}</div></td>'
                '<td><table class="metadata">'
                '<tr><td class="label">Prompt</td><td class="value">log prompt</td></tr>'
                '</table><br/>'
                f'<button onclick="to_clipboard(\'{payload}\')">Copy to Clipboard</button>'
                '</td></tr></table></div>'
                '<!--fooocus-log-split--></body></html>'
            )

    def test_schema_initializes_expected_tables_and_indexes(self):
        history_db.init_db()

        with history_db._connect() as conn:
            tables = {
                row['name']
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            indexes = {
                row['name']
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                ).fetchall()
            }
            image_columns = {
                row['name']
                for row in conn.execute('PRAGMA table_info(images)').fetchall()
            }

        self.assertTrue({
            'batches',
            'images',
            'image_loras',
            'tags',
            'image_tags',
            'batch_tags',
            'notes',
            'prompt_configs',
        }.issubset(tables))
        self.assertTrue({
            'idx_images_created_at',
            'idx_images_batch_id',
            'idx_images_checkpoint',
            'idx_images_seed',
            'idx_image_loras_name',
            'idx_batches_created_at',
            'idx_images_thumbnail_hidden',
        }.issubset(indexes))
        self.assertIn('thumbnail_hidden', image_columns)

    def test_reconcile_outputs_groups_new_images_and_keeps_existing_rows(self):
        output_folder = os.path.join(self.temp_dir, 'outputs')
        base_time = time.time() - 7200
        close_first = os.path.join(output_folder, 'image_1.png')
        close_second = os.path.join(output_folder, 'image_2.png')
        later = os.path.join(output_folder, 'image_3.png')

        self._create_image(close_first, base_time)
        self._create_image(close_second, base_time + 60)
        self._create_image(later, base_time + 3600)

        first = history_db.reconcile_outputs_folder(output_folder)

        self.assertEqual(3, first['added'])
        self.assertEqual(0, first['unchanged'])
        self.assertEqual(2, first['imported_batches'])
        self.assertEqual(3, self._image_count())
        self.assertEqual(2, self._batch_count())

        second = history_db.reconcile_outputs_folder(output_folder)

        self.assertEqual(0, second['added'])
        self.assertEqual(3, second['unchanged'])
        self.assertEqual(0, second['imported_batches'])
        self.assertEqual(3, self._image_count())
        self.assertEqual(2, self._batch_count())

        os.remove(later)
        third = history_db.reconcile_outputs_folder(output_folder)

        self.assertEqual(1, third['removed'])
        self.assertEqual(1, third['removed_batches'])
        self.assertEqual(2, self._image_count())
        self.assertEqual(1, self._batch_count())

    def test_batch_curation_and_filters(self):
        output_folder = os.path.join(self.temp_dir, 'outputs')
        image_path = os.path.join(output_folder, 'image.png')
        self._create_image(image_path, time.time())
        history_db.reconcile_outputs_folder(output_folder)
        batch = history_db.list_batches()[0]

        saved = history_db.update_batch_curation(
            batch['id'],
            favorite=True,
            rating=4,
            review_status='keeper',
            tags='portrait, lora test',
            note='good comparison batch'
        )

        self.assertTrue(saved)
        curation = history_db.get_batch_curation(batch['id'])
        self.assertTrue(curation['favorite'])
        self.assertEqual(4, curation['rating'])
        self.assertEqual('keeper', curation['review_status'])
        self.assertEqual('lora test, portrait', curation['tags'])
        self.assertEqual('good comparison batch', curation['note'])
        self.assertEqual(1, len(history_db.list_batches(favorite_only=True)))
        self.assertEqual(1, len(history_db.list_batches(review_status='keeper')))
        self.assertEqual(1, len(history_db.list_batches(tag='portrait')))

    def test_batch_comparison_rows_include_checkpoint_seed_and_testing_lora(self):
        output_folder = os.path.join(self.temp_dir, 'outputs')
        paths = [os.path.join(output_folder, f'image_{index}.png') for index in range(4)]
        for index, path in enumerate(paths):
            self._create_image(path, time.time() + index)

        task = SimpleNamespace(
            prompt='comparison prompt',
            negative_prompt='',
            style_selections=[],
            wildprompt_selections=[],
            wildprompt_generate_all=False,
            wildprompt_line_selections={},
            performance_selection=SimpleNamespace(value='Quality'),
            overwrite_step=0,
            overwrite_switch=0,
            cfg_scale=7,
            sharpness=2,
            adm_scaler_positive=1.5,
            adm_scaler_negative=0.8,
            adm_scaler_end=0.3,
            refiner_swap_method='joint',
            adaptive_cfg=7,
            clip_skip=2,
            base_model_name='checkpoint_a.safetensors',
            refiner_model_name='None',
            refiner_switch=0.5,
            sampler_name='dpmpp_2m_sde_gpu',
            scheduler_name='karras',
            vae_name='Default',
            seed=100,
            aspect_ratios_selection='1024 1024',
            quick_preview=False,
            training_mode=False,
            testing_mode=True,
            testing_loras=['lora_a.safetensors', 'lora_b.safetensors'],
            loras=[],
            image_number=2,
            multi_checkpoint_model_names=[]
        )
        batch_id = history_db.create_batch_from_task(task)

        metadata_rows = [
            ('checkpoint_a.safetensors', '100', 'lora_a.safetensors', 0),
            ('checkpoint_a.safetensors', '100', 'lora_b.safetensors', 1),
            ('checkpoint_a.safetensors', '101', 'lora_a.safetensors', 2),
            ('checkpoint_a.safetensors', '101', 'lora_b.safetensors', 3),
        ]
        for path, (checkpoint, seed, testing_lora, image_index) in zip(paths, metadata_rows):
            history_db.record_image(
                batch_id,
                path,
                [
                    ('Prompt', 'prompt', 'comparison prompt'),
                    ('Negative Prompt', 'negative_prompt', ''),
                    ('Seed', 'seed', seed),
                    ('Base Model', 'base_model', checkpoint),
                    ('Sampler', 'sampler', 'dpmpp_2m_sde_gpu'),
                    ('Scheduler', 'scheduler', 'karras'),
                    ('Testing LoRA', 'testing_lora', testing_lora),
                ],
                loras=[(testing_lora, 1.0)],
                width=8,
                height=8,
                image_index=image_index
            )

        rows = history_db.list_batch_comparison_rows(batch_id)

        self.assertEqual(4, len(rows))
        self.assertEqual(['100', '100', '101', '101'], [str(row['seed']) for row in rows])
        self.assertEqual(
            ['lora_a.safetensors', 'lora_b.safetensors', 'lora_a.safetensors', 'lora_b.safetensors'],
            [row['testing_lora'] for row in rows]
        )
        self.assertEqual('checkpoint_a.safetensors', rows[0]['checkpoint'])

    def test_list_images_can_filter_by_output_day_folder(self):
        output_folder = self.temp_dir
        first_day = '2026-08-13'
        second_day = '2026-08-14'
        first_path = os.path.join(output_folder, first_day, 'image_1.png')
        second_path = os.path.join(output_folder, second_day, 'image_2.png')
        self._create_image(first_path, time.time())
        self._create_image(second_path, time.time() + 1)

        history_db.reconcile_outputs_folder(output_folder)

        days = history_db.list_output_days()
        first_day_images = history_db.list_images(days=[first_day])
        second_day_images = history_db.list_images(days=[second_day])
        all_images = history_db.list_images()

        self.assertIn(first_day, days)
        self.assertIn(second_day, days)
        self.assertEqual(1, len(first_day_images))
        self.assertEqual(1, len(second_day_images))
        self.assertEqual(2, len(all_images))
        self.assertTrue(first_day_images[0]['path'].endswith(os.path.join(first_day, 'image_1.png')))

    def test_list_images_can_filter_by_checkpoint_and_lora(self):
        output_folder = os.path.join(self.temp_dir, 'outputs')
        first_path = os.path.join(output_folder, 'image_1.png')
        second_path = os.path.join(output_folder, 'image_2.png')
        self._create_image(first_path, time.time())
        self._create_image(second_path, time.time() + 1)
        task = SimpleNamespace(
            prompt='filter prompt',
            negative_prompt='',
            style_selections=[],
            wildprompt_selections=[],
            wildprompt_generate_all=False,
            wildprompt_line_selections={},
            performance_selection=SimpleNamespace(value='Quality'),
            overwrite_step=0,
            overwrite_switch=0,
            cfg_scale=7,
            sharpness=2,
            adm_scaler_positive=1.5,
            adm_scaler_negative=0.8,
            adm_scaler_end=0.3,
            refiner_swap_method='joint',
            adaptive_cfg=7,
            clip_skip=2,
            base_model_name='checkpoint_a.safetensors',
            refiner_model_name='None',
            refiner_switch=0.5,
            sampler_name='dpmpp_2m_sde_gpu',
            scheduler_name='karras',
            vae_name='Default',
            seed=100,
            aspect_ratios_selection='1024 1024',
            quick_preview=False,
            training_mode=False,
            testing_mode=False,
            testing_loras=[],
            loras=[],
            image_number=2,
            multi_checkpoint_model_names=[]
        )
        batch_id = history_db.create_batch_from_task(task)
        for path, checkpoint, lora_name in [
            (first_path, 'checkpoint_a.safetensors', 'portrait_lora.safetensors'),
            (second_path, 'checkpoint_b.safetensors', 'style_lora.safetensors')
        ]:
            history_db.record_image(
                batch_id,
                path,
                [
                    ('Prompt', 'prompt', 'filter prompt'),
                    ('Negative Prompt', 'negative_prompt', ''),
                    ('Seed', 'seed', '100'),
                    ('Base Model', 'base_model', checkpoint),
                ],
                loras=[(lora_name, 1.0)],
                width=8,
                height=8
            )

        checkpoints = history_db.list_images(checkpoints=['checkpoint_a.safetensors'])
        loras = history_db.list_images(loras=['style_lora.safetensors'])
        filter_values = history_db.list_filter_values()

        self.assertEqual(1, len(checkpoints))
        self.assertTrue(checkpoints[0]['path'].endswith('image_1.png'))
        self.assertEqual(1, len(loras))
        self.assertTrue(loras[0]['path'].endswith('image_2.png'))
        self.assertIn('checkpoint_a.safetensors', filter_values['checkpoints'])
        self.assertIn('portrait_lora.safetensors', filter_values['loras'])

    def test_get_image_id_by_path(self):
        output_folder = os.path.join(self.temp_dir, 'outputs')
        image_path = os.path.join(output_folder, 'image.png')
        self._create_image(image_path, time.time())
        history_db.reconcile_outputs_folder(output_folder)

        image_id = history_db.get_image_id_by_path(image_path)

        self.assertIsNotNone(image_id)
        self.assertEqual(image_path, history_db.get_image_path(image_id))

    def test_delete_image_removes_file_and_record(self):
        output_folder = os.path.join(self.temp_dir, 'outputs')
        image_path = os.path.join(output_folder, 'image.png')
        self._create_image(image_path, time.time())
        history_db.reconcile_outputs_folder(output_folder)
        image_id = history_db.get_image_id_by_path(image_path)

        deleted, deleted_path = history_db.delete_image(image_id, delete_file=True)

        self.assertTrue(deleted)
        self.assertEqual(image_path, deleted_path)
        self.assertFalse(os.path.exists(image_path))
        self.assertIsNone(history_db.get_image_id_by_path(image_path))

    def test_seed_stacks_group_same_prompt_and_seed(self):
        output_folder = os.path.join(self.temp_dir, 'outputs')
        paths = [os.path.join(output_folder, f'image_{index}.png') for index in range(3)]
        for index, path in enumerate(paths):
            self._create_image(path, time.time() + index)
        task = SimpleNamespace(
            prompt='same test prompt',
            negative_prompt='',
            style_selections=[],
            wildprompt_selections=[],
            wildprompt_generate_all=False,
            wildprompt_line_selections={},
            performance_selection=SimpleNamespace(value='Quality'),
            overwrite_step=0,
            overwrite_switch=0,
            cfg_scale=7,
            sharpness=2,
            adm_scaler_positive=1.5,
            adm_scaler_negative=0.8,
            adm_scaler_end=0.3,
            refiner_swap_method='joint',
            adaptive_cfg=7,
            clip_skip=2,
            base_model_name='checkpoint_a.safetensors',
            refiner_model_name='None',
            refiner_switch=0.5,
            sampler_name='dpmpp_2m_sde_gpu',
            scheduler_name='karras',
            vae_name='Default',
            seed=100,
            aspect_ratios_selection='1024 1024',
            quick_preview=False,
            training_mode=False,
            testing_mode=True,
            testing_loras=['lora_a.safetensors', 'lora_b.safetensors'],
            loras=[],
            image_number=2,
            multi_checkpoint_model_names=[]
        )
        batch_id = history_db.create_batch_from_task(task)
        rows = [
            (paths[0], 'checkpoint_a.safetensors', '100', 'lora_a.safetensors'),
            (paths[1], 'checkpoint_b.safetensors', '100', 'lora_b.safetensors'),
            (paths[2], 'checkpoint_a.safetensors', '101', 'lora_a.safetensors'),
        ]
        for path, checkpoint, seed, lora_name in rows:
            history_db.record_image(
                batch_id,
                path,
                [
                    ('Prompt', 'prompt', 'same test prompt'),
                    ('Negative Prompt', 'negative_prompt', ''),
                    ('Seed', 'seed', seed),
                    ('Base Model', 'base_model', checkpoint),
                ],
                loras=[(lora_name, 1.0)],
                width=8,
                height=8
            )

        stacks = history_db.list_seed_stacks()
        seed, prompt = history_db.get_seed_stack_key(stacks[0]['id'])
        stack_images = history_db.list_seed_stack_images(seed, prompt)

        self.assertEqual(1, len(stacks))
        self.assertEqual(100, seed)
        self.assertEqual('same test prompt', prompt)
        self.assertEqual(2, len(stack_images))

    def test_thumbnail_visibility_filters_images_and_seed_stacks(self):
        output_folder = os.path.join(self.temp_dir, 'outputs')
        paths = [os.path.join(output_folder, f'image_{index}.png') for index in range(2)]
        for index, path in enumerate(paths):
            self._create_image(path, time.time() + index)
        task = SimpleNamespace(
            prompt='visibility prompt',
            negative_prompt='',
            style_selections=[],
            wildprompt_selections=[],
            wildprompt_generate_all=False,
            wildprompt_line_selections={},
            performance_selection=SimpleNamespace(value='Quality'),
            overwrite_step=0,
            overwrite_switch=0,
            cfg_scale=7,
            sharpness=2,
            adm_scaler_positive=1.5,
            adm_scaler_negative=0.8,
            adm_scaler_end=0.3,
            refiner_swap_method='joint',
            adaptive_cfg=7,
            clip_skip=2,
            base_model_name='checkpoint_a.safetensors',
            refiner_model_name='None',
            refiner_switch=0.5,
            sampler_name='dpmpp_2m_sde_gpu',
            scheduler_name='karras',
            vae_name='Default',
            seed=100,
            aspect_ratios_selection='1024 1024',
            quick_preview=False,
            training_mode=False,
            testing_mode=False,
            testing_loras=[],
            loras=[],
            image_number=2,
            multi_checkpoint_model_names=[]
        )
        batch_id = history_db.create_batch_from_task(task)
        image_ids = []
        for path in paths:
            image_ids.append(history_db.record_image(
                batch_id,
                path,
                [
                    ('Prompt', 'prompt', 'visibility prompt'),
                    ('Negative Prompt', 'negative_prompt', ''),
                    ('Seed', 'seed', '100'),
                    ('Base Model', 'base_model', 'checkpoint_a.safetensors'),
                ],
                width=8,
                height=8
            ))

        self.assertTrue(history_db.set_image_thumbnail_hidden(image_ids[0], True))

        visible = history_db.list_images(thumbnail_visibility='visible')
        all_images = history_db.list_images(thumbnail_visibility='all')
        hidden = history_db.list_images(thumbnail_visibility='hidden')
        visible_stacks = history_db.list_seed_stacks(thumbnail_visibility='visible')
        all_stacks = history_db.list_seed_stacks(thumbnail_visibility='all')

        self.assertEqual([image_ids[1]], [row['id'] for row in visible])
        self.assertEqual({image_ids[0], image_ids[1]}, {row['id'] for row in all_images})
        self.assertEqual([image_ids[0]], [row['id'] for row in hidden])
        self.assertEqual(0, len(visible_stacks))
        self.assertEqual(1, len(all_stacks))

        self.assertTrue(history_db.set_image_thumbnail_hidden(image_ids[0], False))
        visible_again = history_db.list_images(thumbnail_visibility='visible')

        self.assertEqual({image_ids[0], image_ids[1]}, {row['id'] for row in visible_again})

    def test_record_image_stores_reloadable_config_and_loras(self):
        output_folder = os.path.join(self.temp_dir, 'outputs')
        image_path = os.path.join(output_folder, 'recorded.png')
        self._create_image(image_path, time.time())
        task = SimpleNamespace(
            prompt='record prompt',
            negative_prompt='record negative',
            style_selections=['Fooocus V2'],
            wildprompt_selections=[],
            wildprompt_generate_all=False,
            wildprompt_line_selections={},
            performance_selection=SimpleNamespace(value='Quality'),
            overwrite_step=30,
            overwrite_switch=0.5,
            cfg_scale=7,
            sharpness=2,
            adm_scaler_positive=1.5,
            adm_scaler_negative=0.8,
            adm_scaler_end=0.3,
            refiner_swap_method='joint',
            adaptive_cfg=7,
            clip_skip=2,
            base_model_name='checkpoint_a.safetensors',
            refiner_model_name='None',
            refiner_switch=0.5,
            sampler_name='dpmpp_2m_sde_gpu',
            scheduler_name='karras',
            vae_name='Default',
            seed=123,
            aspect_ratios_selection='1024 1024',
            quick_preview=False,
            training_mode=False,
            testing_mode=False,
            testing_loras=[],
            loras=[('portrait_lora.safetensors', 0.75)],
            image_number=1,
            multi_checkpoint_model_names=[]
        )
        batch_id = history_db.create_batch_from_task(task)

        image_id = history_db.record_image(
            batch_id,
            image_path,
            [
                ('Prompt', 'prompt', 'record prompt'),
                ('Negative Prompt', 'negative_prompt', 'record negative'),
                ('Seed', 'seed', '123'),
                ('Base Model', 'base_model', 'checkpoint_a.safetensors'),
                ('Sampler', 'sampler', 'dpmpp_2m_sde_gpu'),
                ('Scheduler', 'scheduler', 'karras'),
                ('Steps', 'steps', 30),
            ],
            loras=[('portrait_lora.safetensors', 0.75)],
            width=8,
            height=8,
            image_index=0
        )

        config = history_db.get_config_by_path(image_path)
        summary = history_db.get_image_summary(image_id)
        with history_db._connect() as conn:
            lora_rows = conn.execute(
                'SELECT name, weight, role FROM image_loras WHERE image_id = ?',
                (image_id,)
            ).fetchall()
            image_row = conn.execute(
                'SELECT width, height FROM images WHERE id = ?',
                (image_id,)
            ).fetchone()

        self.assertEqual('record prompt', config['prompt'])
        self.assertEqual('record negative', config['negative_prompt'])
        self.assertEqual(30, config['steps'])
        self.assertEqual('checkpoint_a.safetensors', summary['checkpoint'])
        self.assertEqual(123, summary['seed'])
        self.assertEqual(8, image_row['width'])
        self.assertEqual(8, image_row['height'])
        self.assertEqual(1, len(lora_rows))
        self.assertEqual('portrait_lora.safetensors', lora_rows[0]['name'])
        self.assertEqual(0.75, lora_rows[0]['weight'])
        self.assertEqual('active', lora_rows[0]['role'])

    def test_reconcile_imports_reloadable_config_from_embedded_metadata(self):
        output_folder = os.path.join(self.temp_dir, 'outputs')
        image_path = os.path.join(output_folder, 'embedded.png')
        os.makedirs(output_folder, exist_ok=True)
        metadata = PngInfo()
        metadata.add_text('parameters', history_db._json_dumps({
            'prompt': 'embedded prompt',
            'negative_prompt': 'embedded negative',
            'seed': '456',
            'steps': 10,
            'sampler': 'dpmpp_2m_sde_gpu',
            'scheduler': 'karras',
            'guidance_scale': 7,
        }))
        Image.new('RGB', (12, 10), (30, 40, 50)).save(image_path, pnginfo=metadata)

        result = history_db.reconcile_outputs_folder(output_folder)
        image_id = history_db.get_image_id_by_path(image_path)
        config = history_db.get_config_by_path(image_path)
        summary = history_db.get_image_summary(image_id)
        with history_db._connect() as conn:
            image_row = conn.execute(
                'SELECT width, height FROM images WHERE id = ?',
                (image_id,)
            ).fetchone()

        self.assertEqual(1, result['added'])
        self.assertEqual('embedded prompt', config['prompt'])
        self.assertEqual('embedded negative', config['negative_prompt'])
        self.assertEqual('456', str(config['seed']))
        self.assertEqual(10, config['steps'])
        self.assertEqual('embedded prompt', summary['prompt'])
        self.assertEqual(456, summary['seed'])
        self.assertEqual(12, image_row['width'])
        self.assertEqual(10, image_row['height'])

    def test_reconcile_imports_missing_metadata_from_log_html(self):
        output_folder = os.path.join(self.temp_dir, 'outputs')
        image_path = os.path.join(output_folder, 'old_image.png')
        self._create_image(image_path, time.time())
        self._write_log_html(output_folder, 'old_image.png', {
            'prompt': 'log prompt',
            'negative_prompt': 'log negative',
            'seed': '789',
            'base_model': 'log_checkpoint.safetensors',
            'sampler': 'dpmpp_2m_sde_gpu',
            'scheduler': 'karras',
            'steps': 30,
            'lora_combined_1': 'log_lora.safetensors : 0.65',
        })

        result = history_db.reconcile_outputs_folder(output_folder)
        image_id = history_db.get_image_id_by_path(image_path)
        config = history_db.get_config_by_path(image_path)
        summary = history_db.get_image_summary(image_id)
        with history_db._connect() as conn:
            lora = conn.execute(
                'SELECT name, weight FROM image_loras WHERE image_id = ?',
                (image_id,)
            ).fetchone()

        self.assertEqual(1, result['added'])
        self.assertEqual('log prompt', config['prompt'])
        self.assertEqual('log negative', config['negative_prompt'])
        self.assertEqual(30, config['steps'])
        self.assertEqual('log_checkpoint.safetensors', summary['checkpoint'])
        self.assertEqual(789, summary['seed'])
        self.assertEqual('log_lora.safetensors', lora['name'])
        self.assertEqual(0.65, lora['weight'])

    def test_log_html_does_not_override_embedded_metadata(self):
        output_folder = os.path.join(self.temp_dir, 'outputs')
        image_path = os.path.join(output_folder, 'embedded_with_log.png')
        os.makedirs(output_folder, exist_ok=True)
        metadata = PngInfo()
        metadata.add_text('parameters', history_db._json_dumps({
            'prompt': 'embedded wins',
            'negative_prompt': 'embedded negative',
            'seed': '111',
            'base_model': 'embedded_checkpoint.safetensors',
        }))
        Image.new('RGB', (12, 10), (30, 40, 50)).save(image_path, pnginfo=metadata)
        self._write_log_html(output_folder, 'embedded_with_log.png', {
            'prompt': 'log should not replace',
            'negative_prompt': 'log negative',
            'seed': '222',
            'base_model': 'log_checkpoint.safetensors',
            'sampler': 'log_sampler',
        })

        result = history_db.reconcile_outputs_folder(output_folder)
        config = history_db.get_config_by_path(image_path)
        summary = history_db.get_image_summary(history_db.get_image_id_by_path(image_path))

        self.assertEqual(1, result['added'])
        self.assertEqual('embedded wins', config['prompt'])
        self.assertEqual('embedded negative', config['negative_prompt'])
        self.assertEqual('111', str(config['seed']))
        self.assertEqual('log_checkpoint.safetensors', summary['checkpoint'])
        self.assertEqual('log_sampler', config['sampler'])

    def test_reconcile_enriches_existing_file_only_rows_from_log_html(self):
        output_folder = os.path.join(self.temp_dir, 'outputs')
        image_path = os.path.join(output_folder, 'existing_old_image.png')
        self._create_image(image_path, time.time())

        first = history_db.reconcile_outputs_folder(output_folder)
        image_id = history_db.get_image_id_by_path(image_path)
        self._write_log_html(output_folder, 'existing_old_image.png', {
            'prompt': 'late log prompt',
            'negative_prompt': 'late log negative',
            'seed': '321',
            'base_model': 'late_checkpoint.safetensors',
        })
        second = history_db.reconcile_outputs_folder(output_folder)
        config = history_db.get_config_by_path(image_path)
        summary = history_db.get_image_summary(image_id)

        self.assertEqual(1, first['added'])
        self.assertEqual(1, second['updated'])
        self.assertEqual('late log prompt', config['prompt'])
        self.assertEqual('late log negative', config['negative_prompt'])
        self.assertEqual(321, summary['seed'])
        self.assertEqual('late_checkpoint.safetensors', summary['checkpoint'])


if __name__ == '__main__':
    unittest.main()

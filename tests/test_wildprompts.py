import os
import tempfile
import unittest
from random import Random

from modules import sdxl_styles
from modules import wildprompt_sorter


class TestWildprompts(unittest.TestCase):
    def setUp(self):
        self.original_wildprompts_path = sdxl_styles.wildprompts_path
        self.original_all_wildprompts = wildprompt_sorter.all_wildprompts
        self.temp_dir = tempfile.TemporaryDirectory()
        sdxl_styles.wildprompts_path = self.temp_dir.name

        self._write('Clothes/Dress', ['red dress', 'green dress'])
        self._write('Locations/Places', ['garden', 'rooftop'])
        self._write('Poses/Poses', ['looking back', 'seated'])
        self._write('Root Prompt', ['portrait'])
        wildprompt_sorter.all_wildprompts = sdxl_styles.get_legal_wildprompt_names()

    def tearDown(self):
        sdxl_styles.wildprompts_path = self.original_wildprompts_path
        wildprompt_sorter.all_wildprompts = self.original_all_wildprompts
        self.temp_dir.cleanup()

    def _write(self, name, lines):
        path = os.path.join(self.temp_dir.name, *name.split('/')) + '.txt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(lines))

    def test_nested_files_are_discovered_with_portable_names(self):
        self.assertEqual(
            [
                'Clothes/Dress',
                'Locations/Places',
                'Poses/Poses',
                'Root Prompt',
            ],
            sdxl_styles.get_legal_wildprompt_names(),
        )

    def test_random_mix_uses_one_row_from_every_selected_file(self):
        result = sdxl_styles.apply_wildprompts(
            ['Clothes/Dress', 'Locations/Places', 'Poses/Poses'],
            Random(7),
        )

        parts = result.split(', ')
        self.assertEqual(3, len(parts))
        self.assertIn(parts[0], ['red dress', 'green dress'])
        self.assertIn(parts[1], ['garden', 'rooftop'])
        self.assertIn(parts[2], ['looking back', 'seated'])

    def test_resolved_wildprompts_keep_each_file_name_and_chosen_row(self):
        result = sdxl_styles.resolve_wildprompts(
            ['Clothes/Dress', 'Locations/Places', 'Poses/Poses'],
            Random(7),
        )

        self.assertEqual(
            ['Clothes/Dress', 'Locations/Places', 'Poses/Poses'],
            [item['name'] for item in result],
        )
        self.assertEqual(
            sdxl_styles.apply_wildprompts(
                ['Clothes/Dress', 'Locations/Places', 'Poses/Poses'],
                Random(7),
            ),
            ', '.join(item['prompt'] for item in result),
        )

    def test_generate_all_builds_cartesian_product_in_selection_order(self):
        result = sdxl_styles.get_all_wildprompts(
            ['Clothes/Dress', 'Locations/Places', 'Poses/Poses']
        )

        self.assertEqual(8, len(result))
        self.assertEqual('red dress, garden, looking back', result[0])
        self.assertEqual('green dress, rooftop, seated', result[-1])

    def test_generate_all_respects_selected_rows(self):
        result = sdxl_styles.get_all_wildprompts(
            ['Clothes/Dress', 'Locations/Places', 'Poses/Poses'],
            {
                'Clothes/Dress': ['green dress'],
                'Locations/Places': ['garden', 'rooftop'],
                'Poses/Poses': ['looking back'],
            },
        )

        self.assertEqual(
            [
                'green dress, garden, looking back',
                'green dress, rooftop, looking back',
            ],
            result,
        )

    def test_generate_all_can_expand_only_one_selected_file(self):
        combinations = sdxl_styles.get_wildprompt_fixed_combinations(
            ['Clothes/Dress', 'Locations/Places', 'Poses/Poses'],
            ['Poses/Poses'],
        )

        self.assertEqual(
            [
                {'Poses/Poses': 'looking back'},
                {'Poses/Poses': 'seated'},
            ],
            combinations,
        )
        prompts = [
            sdxl_styles.apply_wildprompts(
                ['Clothes/Dress', 'Locations/Places', 'Poses/Poses'],
                Random(index),
                fixed_lines=fixed_lines,
            )
            for index, fixed_lines in enumerate(combinations)
        ]
        self.assertTrue(prompts[0].endswith('looking back'))
        self.assertTrue(prompts[1].endswith('seated'))
        self.assertEqual(3, len(prompts[0].split(', ')))

    def test_legacy_generate_all_true_expands_every_selected_file(self):
        self.assertEqual(
            ['Clothes/Dress', 'Locations/Places'],
            sdxl_styles.normalize_wildprompt_generate_all_files(
                ['Clothes/Dress', 'Locations/Places'],
                True,
            ),
        )

    def test_empty_row_selection_skips_that_ingredient(self):
        result = sdxl_styles.get_all_wildprompts(
            ['Clothes/Dress', 'Locations/Places'],
            {'Locations/Places': []},
        )

        self.assertEqual(['red dress', 'green dress'], result)

    def test_wildprompt_paths_cannot_escape_the_library(self):
        with self.assertRaises(ValueError):
            sdxl_styles._wildprompt_path('../outside')

    def test_folder_chips_use_top_level_folder_names(self):
        self.assertEqual(
            ['Clothes', 'Locations', 'Poses', 'Uncategorized'],
            wildprompt_sorter.get_wildprompt_folder_names(),
        )

    def test_no_folder_chips_shows_every_wildprompt(self):
        update = wildprompt_sorter.filter_wildprompts_by_folders([], [], '')

        self.assertEqual(
            wildprompt_sorter.all_wildprompts,
            update['choices'],
        )

    def test_folder_chips_hide_selected_files_outside_the_filter(self):
        update = wildprompt_sorter.filter_wildprompts_by_folders(
            ['Clothes/Dress'],
            ['Locations', 'Poses'],
            '',
        )

        self.assertEqual(
            ['Locations/Places', 'Poses/Poses'],
            update['choices'],
        )
        self.assertEqual(['Clothes/Dress'], update['value'])

    def test_selected_files_remain_in_alphabetical_folder_and_name_order(self):
        update = wildprompt_sorter.filter_wildprompts_by_folders(
            ['Poses/Poses', 'Clothes/Dress'],
            [],
            '',
        )

        self.assertEqual(
            ['Clothes/Dress', 'Locations/Places', 'Poses/Poses', 'Root Prompt'],
            update['choices'],
        )
        self.assertEqual(['Poses/Poses', 'Clothes/Dress'], update['value'])

    def test_folder_chips_and_search_filter_together(self):
        update = wildprompt_sorter.filter_wildprompts_by_folders(
            [],
            ['Locations', 'Poses'],
            'places',
        )

        self.assertEqual(['Locations/Places'], update['choices'])

    def test_reset_clears_folder_chips_search_and_selected_prompts(self):
        chips, search, prompts = wildprompt_sorter.reset_wildprompt_browser()

        self.assertEqual([], chips['value'])
        self.assertEqual(wildprompt_sorter.get_wildprompt_folder_names(), chips['choices'])
        self.assertEqual('', search)
        self.assertEqual([], prompts['value'])
        self.assertEqual(wildprompt_sorter.all_wildprompts, prompts['choices'])

    def test_combination_summary_shows_cartesian_count(self):
        summary = wildprompt_sorter.build_wildprompt_combination_summary(
            ['Clothes/Dress', 'Locations/Places', 'Poses/Poses'],
            True,
            '{}',
        )

        self.assertIn('8 combination(s)', summary)
        self.assertIn('2 &times; 2 &times; 2', summary)

    def test_combination_summary_only_multiplies_files_marked_generate_all(self):
        summary = wildprompt_sorter.build_wildprompt_combination_summary(
            ['Clothes/Dress', 'Locations/Places', 'Poses/Poses'],
            ['Poses/Poses'],
            '{}',
            wildprompt_sorter.build_generation_factors(image_number=4),
        )

        self.assertIn('8 total images', summary)
        self.assertIn('2 combination(s) &times; 4 Image Number', summary)
        self.assertIn('other 2 file(s)', summary)

    def test_generate_all_choices_only_include_applied_prompts(self):
        update = wildprompt_sorter.sync_wildprompt_generate_all_files(
            ['Clothes/Dress', 'Poses/Poses'],
            ['Locations/Places', 'Poses/Poses'],
        )

        self.assertEqual(['Clothes/Dress', 'Poses/Poses'], update['choices'])
        self.assertEqual(['Poses/Poses'], update['value'])

    def test_total_summary_multiplies_prompt_rows_by_image_number(self):
        self._write('Scenes/Three Prompts', ['one', 'two', 'three'])
        wildprompt_sorter.all_wildprompts = sdxl_styles.get_legal_wildprompt_names()
        factors = wildprompt_sorter.build_generation_factors(image_number=4)

        summary = wildprompt_sorter.build_wildprompt_combination_summary(
            ['Scenes/Three Prompts'],
            True,
            '{}',
            factors,
        )

        self.assertIn('12 total images', summary)
        self.assertIn('3 combination(s) &times; 4 Image Number', summary)

    def test_total_summary_includes_checkpoint_and_testing_lora_multipliers(self):
        self._write('Scenes/Three Prompts', ['one', 'two', 'three'])
        wildprompt_sorter.all_wildprompts = sdxl_styles.get_legal_wildprompt_names()
        factors = wildprompt_sorter.build_generation_factors(
            image_number=4,
            multi_checkpoint_enabled=True,
            multi_checkpoint_models=['a', 'b', 'c', 'd', 'e'],
            testing_mode=True,
            testing_loras=['one', 'two', 'three', 'four'],
        )

        summary = wildprompt_sorter.build_wildprompt_combination_summary(
            ['Scenes/Three Prompts'],
            True,
            '{}',
            factors,
        )

        self.assertIn('240 total images', summary)
        self.assertIn('5 checkpoints', summary)
        self.assertIn('4 testing LoRAs', summary)
        self.assertIn('Large queue', summary)

    def test_random_mix_does_not_multiply_selected_files(self):
        factors = wildprompt_sorter.build_generation_factors(image_number=4)
        summary = wildprompt_sorter.build_wildprompt_combination_summary(
            ['Clothes/Dress', 'Locations/Places', 'Poses/Poses'],
            False,
            '{}',
            factors,
        )

        self.assertIn('4 total images', summary)
        self.assertIn('Random mix', summary)

    def test_explicit_empty_row_selection_survives_section_refresh(self):
        updates = wildprompt_sorter.update_wildprompt_line_sections(
            ['Clothes/Dress'],
            '{"Clothes/Dress": []}',
        )

        self.assertEqual([], updates[2]['value'])


if __name__ == '__main__':
    unittest.main()

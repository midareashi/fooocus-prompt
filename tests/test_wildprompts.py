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

    def test_empty_row_selection_skips_that_ingredient(self):
        result = sdxl_styles.get_all_wildprompts(
            ['Clothes/Dress', 'Locations/Places'],
            {'Locations/Places': []},
        )

        self.assertEqual(['red dress', 'green dress'], result)

    def test_wildprompt_paths_cannot_escape_the_library(self):
        with self.assertRaises(ValueError):
            sdxl_styles._wildprompt_path('../outside')

    def test_combination_summary_shows_cartesian_count(self):
        summary = wildprompt_sorter.build_wildprompt_combination_summary(
            ['Clothes/Dress', 'Locations/Places', 'Poses/Poses'],
            True,
            '{}',
        )

        self.assertIn('8 combination(s)', summary)
        self.assertIn('2 &times; 2 &times; 2', summary)

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

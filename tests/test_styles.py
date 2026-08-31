import json
import os
import tempfile
import unittest

from modules import sdxl_styles
from modules import style_sorter


class TestStyleLayers(unittest.TestCase):
    def setUp(self):
        self.original_all_styles = style_sorter.all_styles
        style_sorter.try_load_sorted_styles(default_selected=[])

    def tearDown(self):
        style_sorter.all_styles = self.original_all_styles

    def test_folder_catalog_loads_named_prompt_layers(self):
        with tempfile.TemporaryDirectory() as folder:
            category_folder = os.path.join(folder, 'Time of Day')
            os.makedirs(category_folder)
            with open(os.path.join(category_folder, 'styles.json'), 'w', encoding='utf-8') as file:
                json.dump({
                    'selection_mode': 'single',
                    'styles': [
                        {'name': 'Golden Hour', 'prompt': 'warm low-angle sunlight'},
                        {'name': 'Night', 'prompt': 'nighttime atmosphere', 'negative_prompt': 'daylight'},
                    ],
                }, file)

            styles, categories, modes = sdxl_styles.load_style_layers(folder)

        self.assertEqual(
            ('warm low-angle sunlight', ''),
            styles['Time of Day/Golden Hour'],
        )
        self.assertEqual('Time of Day', categories['Time of Day/Night'])
        self.assertEqual('single', modes['Time of Day'])

    def test_catalog_choices_are_alphabetical_by_folder_and_name(self):
        choices = sdxl_styles.get_legal_style_layer_names()

        self.assertEqual(sorted(choices, key=str.casefold), choices)
        self.assertIn('Shot Distance/Head to Knees', choices)
        self.assertIn('Time of Day/Golden Hour', choices)
        self.assertIn('Fooocus V2', choices)

    def test_single_choice_folders_keep_only_latest_selection(self):
        selected = sdxl_styles.normalize_style_layer_selections([
            'Time of Day/Golden Hour',
            'Lighting/Soft Window Light',
            'Time of Day/Night',
            'Lighting/Rim Light',
        ])

        self.assertEqual([
            'Lighting/Soft Window Light',
            'Time of Day/Night',
            'Lighting/Rim Light',
        ], selected)

    def test_aesthetic_layer_replaces_fooocus_photograph(self):
        selected = sdxl_styles.normalize_style_layer_selections([
            'Fooocus V2',
            'Fooocus Photograph',
            'Aesthetic/Anime',
            'Fooocus Negative',
        ])

        self.assertEqual([
            'Fooocus V2',
            'Aesthetic/Anime',
            'Fooocus Negative',
        ], selected)

    def test_folder_filter_hides_outside_choices_without_deselecting_them(self):
        update = style_sorter.filter_styles_by_folders(
            ['Time of Day/Golden Hour', 'Lighting/Rim Light'],
            ['Lighting'],
            '',
        )

        self.assertTrue(all(name.startswith('Lighting/') for name in update['choices']))
        self.assertEqual(
            ['Time of Day/Golden Hour', 'Lighting/Rim Light'],
            update['value'],
        )

    def test_reset_clears_filters_search_and_layers(self):
        folders, search, styles = style_sorter.reset_style_browser()

        self.assertEqual(style_sorter.get_style_categories(), folders['choices'])
        self.assertEqual([], folders['value'])
        self.assertEqual('', search)
        self.assertEqual([], styles['value'])

    def test_atomic_style_adds_separate_positive_and_negative_fragments(self):
        positive, negative, replaced = sdxl_styles.apply_style(
            'Shot Distance/Full Body',
            'portrait subject',
        )

        self.assertFalse(replaced)
        self.assertIn('entire figure visible from head to feet', positive[0])
        self.assertIn('cropped feet', negative[0])


if __name__ == '__main__':
    unittest.main()

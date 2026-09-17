import json
import os
import unittest
from unittest import mock

import httpx

from modules import prompt_assistant


class TestPromptAssistant(unittest.TestCase):
    def setUp(self):
        self.checkpoints = ['realvisXL.safetensors', 'ponyRealism.safetensors']
        self.loras = ['luna_sdxl.safetensors', 'skin_pony.safetensors']

    @mock.patch.dict(os.environ, {
        'FOOOCUS_PROMPT_ASSISTANT_PROVIDER': 'openai',
        'OPENAI_API_KEY': 'test-key',
    }, clear=False)
    @mock.patch('modules.prompt_assistant.httpx.post')
    def test_creates_and_validates_plan(self, post):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            'output': [{
                'type': 'message',
                'content': [{
                    'type': 'output_text',
                    'text': json.dumps({
                        'prompt': 'portrait of Luna, window light',
                        'negative_prompt': 'distorted hands',
                        'checkpoint': 'realvisXL.safetensors',
                        'loras': [{'filename': 'luna_sdxl.safetensors', 'weight': 0.72}],
                        'summary': 'Natural portrait setup.',
                    }),
                }],
            }],
        }
        post.return_value = response

        plan = prompt_assistant.create_prompt_plan(
            'A natural portrait of Luna', self.checkpoints, self.loras
        )

        self.assertEqual('realvisXL.safetensors', plan['checkpoint'])
        self.assertEqual(0.72, plan['loras'][0]['weight'])
        payload = post.call_args.kwargs['json']
        self.assertFalse(payload['store'])
        self.assertEqual(self.checkpoints, payload['text']['format']['schema']['properties']['checkpoint']['enum'])

    @mock.patch.dict(os.environ, {
        'FOOOCUS_PROMPT_ASSISTANT_PROVIDER': 'openai',
        'OPENAI_API_KEY': 'test-key',
    }, clear=False)
    @mock.patch('modules.prompt_assistant.httpx.post')
    def test_rejects_hallucinated_lora(self, post):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            'output': [{
                'type': 'message',
                'content': [{
                    'type': 'output_text',
                    'text': json.dumps({
                        'prompt': 'portrait',
                        'negative_prompt': '',
                        'checkpoint': 'realvisXL.safetensors',
                        'loras': [{'filename': 'invented.safetensors', 'weight': 1.0}],
                        'summary': '',
                    }),
                }],
            }],
        }
        post.return_value = response

        with self.assertRaisesRegex(prompt_assistant.PromptAssistantError, 'unavailable LoRA'):
            prompt_assistant.create_prompt_plan('portrait', self.checkpoints, self.loras)

    def test_requires_installed_checkpoint_before_api_key(self):
        with self.assertRaisesRegex(prompt_assistant.PromptAssistantError, 'Install at least one checkpoint'):
            prompt_assistant.create_prompt_plan('portrait', [], [])

    def test_http_error_message_is_preserved(self):
        request = httpx.Request('POST', prompt_assistant.RESPONSES_URL)
        response = httpx.Response(401, request=request, json={'error': {'message': 'Bad API key'}})
        with mock.patch.dict(os.environ, {
                    'FOOOCUS_PROMPT_ASSISTANT_PROVIDER': 'openai',
                    'OPENAI_API_KEY': 'bad-key',
                }, clear=False), \
                mock.patch('modules.prompt_assistant.httpx.post') as post:
            post.return_value = response
            with self.assertRaisesRegex(prompt_assistant.PromptAssistantError, 'Bad API key'):
                prompt_assistant.create_prompt_plan('portrait', self.checkpoints, self.loras)

    @mock.patch.dict(os.environ, {'FOOOCUS_PROMPT_ASSISTANT_PROVIDER': 'codex'}, clear=False)
    @mock.patch('modules.prompt_assistant.subprocess.run')
    @mock.patch('modules.prompt_assistant._find_codex', return_value='/path/to/codex')
    def test_codex_provider_uses_desktop_login_and_schema(self, find_codex, run):
        plan = {
            'prompt': 'cinematic portrait of Luna',
            'negative_prompt': 'distortion',
            'checkpoint': 'realvisXL.safetensors',
            'loras': [{'filename': 'luna_sdxl.safetensors', 'weight': 0.7}],
            'summary': 'Identity-focused portrait.',
        }

        def complete(command, **kwargs):
            output_path = command[command.index('--output-last-message') + 1]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(plan, f)
            return mock.Mock(returncode=0, stdout='', stderr='')

        run.side_effect = complete
        result = prompt_assistant.create_prompt_plan('portrait', self.checkpoints, self.loras)

        self.assertEqual('realvisXL.safetensors', result['checkpoint'])
        command = run.call_args.args[0]
        self.assertIn('--ephemeral', command)
        self.assertIn('read-only', command)
        self.assertIn('--output-schema', command)


if __name__ == '__main__':
    unittest.main()

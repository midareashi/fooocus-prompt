# AI Prompt Director

The AI Prompt Director turns a plain-language image request into a complete Fooocus setup. It selects from the checkpoint and LoRA files currently installed, chooses LoRA weights, writes positive and negative prompts, disables multi-checkpoint mode, and clears the refiner.

## Setup

1. Sign in to the Codex or ChatGPT desktop app on this Mac.
2. Start Fooocus with `run-mac.command`.
3. Install checkpoint files under `models/checkpoints` and LoRA files under `models/loras`.
4. In Fooocus, click **Refresh All Files** after adding models.

The default provider invokes the Codex CLI bundled with the desktop app. It uses an ephemeral, read-only run with a strict output schema and your existing Codex sign-in. It does not require an OpenAI API key. Prompt Director requests consume normal Codex usage.

When you click **Choose Models and Build Prompt**, the director sends your description, the installed checkpoint and LoRA filenames, and saved LoRA notes through your Codex account. It does not upload checkpoint files, LoRA files, generated images, or person-likeness reference images.

## Better model selection

The director reads the saved prompt note for every installed LoRA. Use the note button beside a LoRA to record:

- its base family, such as SDXL, Pony XL, or Illustrious XL;
- trigger words;
- the effect it was trained for;
- a useful weight range.

For example:

```text
SDXL identity LoRA for Luna. Trigger: ohwx_luna. Usually works at 0.65-0.8.
```

The director treats these notes as catalog data and includes required trigger words when it selects a LoRA. Every returned filename is constrained to the current Fooocus inventory and validated again before the interface is updated.

## Configuration

No configuration file is required. To change the timeout, copy `prompt-assistant.env.example` to `prompt-assistant.env` and edit:

```text
FOOOCUS_PROMPT_ASSISTANT_PROVIDER=codex
FOOOCUS_PROMPT_ASSISTANT_TIMEOUT=120
```

An API-backed provider remains available for people with separate API access. Set the provider to `openai`, then configure `OPENAI_API_KEY` and optionally `FOOOCUS_PROMPT_ASSISTANT_MODEL`. API usage is separate from a ChatGPT or Codex subscription.

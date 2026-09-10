# llama.cpp Model Selector

A keyboard-driven Windows terminal UI for launching local GGUF models with `llama-server`, with per-model settings, search, sorting, template selection, and optional MCP integrations.

## Setup

Requires Python 3.12+ and a compatible local llama-server build:

```powershell
pip install windows-curses python-dotenv
python model_selector.py
```

Edit `LLAMA_SERVER`, `MODEL_DIRS`, and `TEMPLATES_DIR` near the top of `model_selector.py` for your machine. The configured model directories are `C:\llm`, `E:\llm`, `K:\models`, and `M:\models`. Missing directories are skipped. Only the first shard of a split GGUF is listed; its displayed size includes sibling shards.

Copy `.env.example` to `.env` to configure `LLAMA_HOST` and the default `LLAMA_PORT`. A saved per-model port takes precedence. This launcher uses flags supported by the installed build at `C:\tools\llamacpp\llama-server.exe`; check your build's `--help` if using another version.

## Navigation

| Key | Action |
|-----|--------|
| Up / Down | Select model |
| PgUp / PgDn | Jump 10 models |
| Enter | Launch selected model and exit the selector when the server stops |
| s | Open settings |
| c | Copy settings from another saved model |
| / | Filter by name or path; Enter or Esc keeps the filter |
| o | Toggle name / last-launch sorting |
| r | Rescan model directories |
| d | Clear selected model's saved settings and launch timestamp |
| q / Esc | Quit |

Open `/` and press Enter with an empty query to clear the filter. `*` marks saved settings, `>` marks a previously launched model, and `[V]` marks a configured vision projector (or one found in auto mode).

## Settings

The Main tab contains context size, cache types, sampling, reasoning effort, MCP selection, and settings copying. Press Tab / Shift-Tab for Advanced settings: threads, port, Flash Attention, RAM cache, tensor placement, unified KV, parallel slots, batching, load mode, Jinja/templates, reasoning controls, MTP drafting, draft-cache types, and vision.

Use Up / Down to select a field and Left / Right or + / - to cycle values. Fields marked `[*]` accept text with Enter; blank or `none` selects the default, and Esc cancels editing. Enter on `[>]` opens a submenu. Press q, s, or Esc to close settings and save; Enter on other fields also closes and saves. The settings list scrolls to keep the selected field visible.

The launcher always passes `-ngl -1`; GPU layer count is not a menu setting. Most optional settings omit their flag when set to `default`. Jinja is forced on when required by template matching, reasoning-effort kwargs, or enabled MCP definitions.

Copying uses the source model's resolved settings. Vision and draft file paths stay unchanged unless you toggle their inclusion with Tab in the copy menu. Host is never copied.

## Vision and templates

Vision is **disabled by default**. Set Advanced → Visual model to auto to search the model's folder, or select a projector explicitly. Auto mode selects the first sorted matching file; verify it belongs to your model if the directory contains several projectors. Projector detection uses filename patterns and, for some vision-tower names, a size limit.

Auto-match selects a local `.jinja` template by filename heuristics and otherwise uses the GGUF's embedded template. Explicit `TEMPLATE_OVERRIDES` mappings apply even when auto-match is off. Inspect the command preview to see the selected template. Reasoning-effort hints are inferred from the embedded template and can differ from a local override.

## Optional MCP integrations

The registry contains search, Godot, Playwright, Blender, and Unreal integrations. Their dependencies live under the ignored `mcp/` directory and are not included in this repository. Configure their paths and endpoints in `model_selector.py`. An enabled entry is skipped if its declared prerequisite is missing; this does not verify all dependencies or service availability. HTTP services require the local stdio bridge, and the remote service/editor must be running.

## Saved settings and checks

`model_settings.json` stores differences from each model's defaults plus last-used/launch metadata. Host is environment-managed and is not saved. Saves replace the file atomically. Invalid JSON stops loading rather than silently discarding settings; repair the file or move it aside to reset it. Press d to reset one model, or remove the file to reset all settings and history.

Run regression checks without loading a model:

```powershell
python -m unittest discover -s tests -v
```

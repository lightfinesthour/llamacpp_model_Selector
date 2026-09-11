# llama.cpp Model Selector

A keyboard-driven Windows terminal UI for launching local GGUF models with `llama-server`, with per-model settings, search, sorting, template selection, and optional MCP integrations.

## Setup

Requires Python 3.12+ and a compatible local llama-server build:

```powershell
pip install windows-curses python-dotenv huggingface_hub markdown
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
| h | Search Hugging Face and download GGUF quants / companion files |
| x | Delete selected model files (review and confirmation required) |
| d | Clear selected model's saved settings and launch timestamp |
| q / Esc | Quit |

Open `/` and press Enter with an empty query to clear the filter. `*` marks saved settings, `>` marks a previously launched model, and `[V]` marks a configured vision projector (or one found in auto mode).

## Hugging Face downloads and deletion

Press **h** to open the graphical model browser, even when the library is empty. It uses a local browser interface with a dark two-pane layout: search results on the left, model details and downloads on the right. You can also launch it directly:

```powershell
python model_browser.py
```

Search updates as you type. Enter a model name, author, multiple terms such as `qwen unsl`, an exact `owner/repository`, or a Hugging Face repository/file URL. Sort by **Best Match**, **Most Likes**, **Most Downloads**, or **Recently Updated**. Recently Updated sorts by repository modification time, not creation date. Results load in pages of 50; Show more models continues the search. Best Match uses Hugging Face's default ordering, not LM Studio's curated ranking. The browser uses live Hugging Face data and does not include LM Studio's proprietary Staff Picks.

Selecting a result shows downloads, likes, update date, the model card, and a quant dropdown with exact file sizes and combined shard sizes. Separate groups distinguish model quants, vision projectors, and MTP/draft modules. Incomplete shard sets are disabled. Weight-size badges compare files with detected system RAM / NVIDIA VRAM; these are estimates only and exclude context, KV cache and runtime memory. Unavailable GPU memory is not guessed.

Choose the main quant, compatible vision/MTP files if needed, and a configured download location. Review the file list and download total. The official `hf download` CLI downloads all selected shards at the listed repository revision to `<model root>/<owner>/<repository>/`, preserving repository subfolders. The browser remains usable during downloads and shows a Cancel action, live speed in MB/s (1 MB = 1,000,000 bytes), estimated time remaining for the selected files, and byte-based progress. Estimates use HF transfer readings; before readings arrive or when they become stale, speed/time show Waiting or Calculating. The raw HF log remains available. Re-selecting the same files lets HF reuse/resume downloads. Free disk space is checked against known sizes.

Selected companion paths are saved explicitly for the launcher, including files in other subfolders; selecting a draft enables the existing MTP setting. Roles are inferred from filenames, so verify compatibility. Embedded MTP needs no separate file; enable it in Advanced settings when the model supports it. Incomplete local shard sets are hidden from the launcher.

**My Models** lists the local library and lets you review and permanently delete a model and its shards. Companion files remain because other quants may share them. Use **Return to launcher** when done; the terminal reloads models and saved settings. If you close the browser tab directly, the service stays alive; use the printed URL to reopen it or Ctrl+C in its terminal to stop it. While a download runs, finish or cancel it before returning to the launcher.

If `hf` is missing, run `python -m pip install -U huggingface_hub` and ensure its Scripts directory is on PATH. Install `markdown` for formatted README rendering. For gated/private repositories, run `hf auth login` and obtain access on Hugging Face. Authentication uses HF's saved credentials or `HF_TOKEN`. The browser service binds only to localhost and requires a per-session token for its API. See the [official HF CLI documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/cli).

Press **x** to preview and permanently delete the selected GGUF and its shards. Configured companion files are kept by default because other quants may share them; you can explicitly include them. The final confirmation lists the exact paths. Only files inside configured model roots can be deleted; repository folders, HF cache metadata, and unrelated quants remain. Successful deletion also clears that model's settings/history. **d** still only resets settings.

## Settings

The Main tab contains context size, cache types, sampling, reasoning effort, MCP selection, and settings copying. Press Tab / Shift-Tab for Advanced settings: threads, port, Flash Attention, RAM cache, tensor placement, unified KV, parallel slots, batching, load mode, Jinja/templates, reasoning controls, MTP drafting, draft-cache types, and vision.

Use Up / Down to select a field and Left / Right or + / - to cycle values. Fields marked `[*]` accept text with Enter; blank or `none` selects the default, and Esc cancels editing. Enter on `[>]` opens a submenu. Press q, s, or Esc to close settings and save; Enter on other fields also closes and saves. The settings list scrolls to keep the selected field visible.

The launcher always passes `-ngl -1`; GPU layer count is not a menu setting. Most optional settings omit their flag when set to `default`. Jinja is forced on when required by template matching, reasoning-effort kwargs, or enabled MCP definitions.

Copying uses the source model's resolved settings. Vision and draft file paths stay unchanged unless you toggle their inclusion with Tab in the copy menu. Host is never copied.

## Vision and templates

Vision is **disabled by default**. Set Advanced → Visual model to auto to search the model's folder, or select a projector explicitly. Auto mode selects the first sorted matching file; verify it belongs to your model if the directory contains several projectors. Projector detection uses filename patterns and, for some vision-tower names, a size limit.

Auto-match selects a local `.jinja` template by filename heuristics and otherwise uses the GGUF's embedded template. Explicit `TEMPLATE_OVERRIDES` mappings apply even when auto-match is off. Inspect the command preview to see the selected template. Reasoning-effort hints are inferred from the embedded template and can differ from a local override.

## Model-specific builds

`MODEL_FIXES` selects a separate server for paths containing
`qwen3.8-flash-next-uncensored`:
`C:\tools\llamacpp\patches\qwen38-flash-next-mtp\llama-server.exe`.
This build includes [llama.cpp PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243)
for Qwen3.8 Flash Next MTP. Keep Draft MTP enabled and select the matching
`MTP-draft.gguf` in Advanced settings to use the separate draft head.
Other models use the normal server. The patch directory contains build instructions
and the pinned source revision. If the patched executable is missing, the existing
exception mechanism falls back to the normal server, which may not support this MTP
head; disable Draft MTP or restore the patched build before launching.

## Optional MCP integrations

The registry contains search, Godot, Playwright, Blender, and Unreal integrations. Their dependencies live under the ignored `mcp/` directory and are not included in this repository. Configure their paths and endpoints in `model_selector.py`. An enabled entry is skipped if its declared prerequisite is missing; this does not verify all dependencies or service availability. HTTP services require the local stdio bridge, and the remote service/editor must be running.

## Saved settings and checks

`model_settings.json` stores differences from each model's defaults plus last-used/launch metadata. Host is environment-managed and is not saved. Saves replace the file atomically. Invalid JSON stops loading rather than silently discarding settings; repair the file or move it aside to reset it. Press d to reset one model, or remove the file to reset all settings and history.

Run regression checks without loading a model:

```powershell
python -m unittest discover -s tests -v
```

# llama.cpp Model Selector

A keyboard-driven terminal UI for launching `llama-server` from your GGUF model library. Automatically detects vision models, remembers per-model settings across sessions, and lets you search and sort your collection.

## Requirements

- Python 3.12+
- `windows-curses` (Windows only)

```
pip install windows-curses
```

## Usage

```
python model_selector.py
```

## Navigation

| Key | Action |
|-----|--------|
| `↑ / ↓` | Move through model list |
| `PgUp / PgDn` | Jump 10 models |
| `Enter` | Launch selected model |
| `s` | Open settings menu |
| `/` | Search / filter models |
| `o` | Toggle sort: name ↔ last launched |
| `r` | Rescan model directories |
| `d` | Delete saved settings for selected model |
| `q` / `Esc` | Quit |

## Model List Indicators

```
 >[V]* 12.3GB  subfolder\model-name.gguf
 ^  ^  ^
 |  |  └─ file size
 |  └──── * = has saved settings
 └─────── > = previously launched  [V] = vision capable (mmproj found)
```

## Settings (`s`)

| Setting | Flag | Description |
|---------|------|-------------|
| Context length | `-c` | Token context window size |
| Threads | `--threads` | CPU threads |
| GPU layers | `-ngl` | Layers offloaded to GPU (999 = all) |
| Port | `--port` | Server port |
| Flash attention | `--flash-attn` | Enable flash attention |
| Cache K | `-ctk` | KV cache quantisation for keys |
| Cache V | `-ctv` | KV cache quantisation for values |
| Verbosity | `--verbosity` | Log verbosity level (0–5) |
| Parallel slots | `-np` | Number of simultaneous request slots |
| Batch size | `-b` | Prompt processing batch size |
| Micro-batch | `-ub` | Micro-batch size for pipeline |
| mlock | `--mlock` | Pin model in RAM (prevent swapping) |

Navigate fields with `↑ / ↓`, change values with `← / →` or `+` / `-`. Settings are **saved per model** when you press `Enter` to launch or `s` to close the menu.

## Vision Models

If a `mmproj-*.gguf` file is found in the same directory as the selected model, the selector automatically adds `--mmproj` and `--image-min-tokens 1024` to the launch command. Vision-capable models are tagged `[V]` in the list.

## Model Directories

The selector recursively scans these directories:

```
C:\llm
E:\llm
```

Edit `MODEL_DIRS` at the top of `model_selector.py` to change them.

## Settings File

Per-model settings are stored in `model_settings.json` next to the script. Only values that differ from defaults are written. Delete the file to reset everything, or press `d` on a model to reset just that one.

## llama-server Path

The path to `llama-server.exe` is set at the top of the script:

```python
LLAMA_SERVER = r"C:\tools\llamacpp\llama-server.exe"
```

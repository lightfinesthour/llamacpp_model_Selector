#!/usr/bin/env python3
r"""
llama.cpp model selector - keyboard-driven TUI
Scans C:\llm and E:\llm recursively for GGUF files, detects mmproj (vision),
and launches llama-server with appropriate settings.
Per-model settings are saved to model_settings.json next to this script.
"""

import sys
import os
import json
import subprocess
import curses
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

LLAMA_SERVER = r"C:\tools\llamacpp\llama-server.exe"
MODEL_DIRS   = [r"C:\llm", r"E:\llm"]
SETTINGS_FILE = Path(__file__).parent / "model_settings.json"

DEFAULTS = {
    "ngl":          999,
    "threads":      16,
    "context":      32768,
    "host":         os.getenv("LLAMA_HOST", "127.0.0.1"),
    "port":         int(os.getenv("LLAMA_PORT", "8080")),
    "flash_attn":   True,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "verbosity":    3,
    "batch":        512,
    "ubatch":       None,
    "mlock":           False,
    "temp":            0.6,
    "top_p":           0.95,
    "top_k":           20,
    "thinking":         None,
    "thinking_budget":  None,
    "reasoning_format": None,
    "repeat_penalty":   1.02,
    "jinja":            False,
    "gemma4_template_fix": False,   # Temp fix: adds --chat-template-file for Gemma 4 models
    "visual_model":     None,       # None=auto (same folder), "none"=disabled, or path to mmproj
}

CONTEXT_OPTIONS  = [4096, 8192, 16384, 32768, 49152, 65536, 72000, 80000, 90000, 131072, 200000, 262144]
THREAD_OPTIONS   = [4, 8, 12, 16, 20, 24, 32]
CACHE_OPTIONS    = [None, "f16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl"]
BATCH_OPTIONS    = [None, 256, 512, 1024, 2048, 4096]
TEMP_OPTIONS     = [None, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.2, 1.5]
TOP_P_OPTIONS    = [None, 0.1, 0.5, 0.8, 0.9, 0.95, 1.0]
TOP_K_OPTIONS    = [None, 0, 10, 20, 40, 80, 100]
THINKING_OPTIONS         = [None, True, False]    # None=default, True=on, False=off
THINKING_BUDGET_OPTIONS  = [None, 256, 1024, 4096, 8192, 16384, 32768]
REASONING_FORMAT_OPTIONS = [None, "deepseek", "deepseek-legacy", "none"]
# deepseek        → extracts thinking into reasoning_content (Open WebUI shows collapsible dropdown)
# deepseek-legacy → keeps <think> tags in content but also populates reasoning_content
# none            → strips all thinking tags from output entirely
REPEAT_PENALTY_OPTIONS   = [None, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5]

# Fields that support direct text entry for precision
EDITABLE_FIELDS = {"temp", "top_p", "top_k", "repeat_penalty"}


# ── Persistence ───────────────────────────────────────────────────────────────

def load_saved_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_settings(all_saved: dict):
    SETTINGS_FILE.write_text(json.dumps(all_saved, indent=2))


def cfg_for_model(model: Path, all_saved: dict) -> dict:
    saved = all_saved.get(str(model), {})
    cfg = dict(DEFAULTS)
    cfg.update(saved)
    return cfg


def persist_cfg(model: Path, cfg: dict, all_saved: dict, is_launch: bool = False):
    # Only save keys that differ from DEFAULTS; skip host (env-managed)
    skip = {"host"}
    delta = {k: v for k, v in cfg.items()
             if k not in skip and v != DEFAULTS.get(k)}
    if delta:
        all_saved[str(model)] = delta
    elif str(model) in all_saved:
        del all_saved[str(model)]
    # record per-model launch time and the last-used model
    meta = all_saved.setdefault("__meta__", {})
    entry = meta.setdefault(str(model), {})
    if is_launch:
        entry["last_launch"] = time.time()
    meta["__last_model__"] = str(model)
    save_settings(all_saved)


def delete_cfg(model: Path, all_saved: dict):
    all_saved.pop(str(model), None)
    all_saved.get("__meta__", {}).pop(str(model), None)
    save_settings(all_saved)


def last_launch_time(model: Path, all_saved: dict):
    return all_saved.get("__meta__", {}).get(str(model), {}).get("last_launch", 0)


# ── Model discovery ───────────────────────────────────────────────────────────

def find_models():
    models = []
    for base in MODEL_DIRS:
        p = Path(base)
        if not p.exists():
            continue
        for f in sorted(p.rglob("*.gguf")):
            name = f.name.lower()
            if "mmproj" in name or "projector" in name:
                continue
            models.append(f)
    return models


def find_mmproj(model_path: Path):
    for f in model_path.parent.glob("*.gguf"):
        n = f.name.lower()
        if "mmproj" in n or "projector" in n:
            return f
    return None


def find_all_mmproj():
    """Recursively scan all model dirs for mmproj-*.gguf files."""
    found = []
    for base in MODEL_DIRS:
        p = Path(base)
        if not p.exists():
            continue
        for f in sorted(p.rglob("*.gguf")):
            n = f.name.lower()
            if n.startswith("mmproj-") or ("mmproj" in n or "projector" in n):
                found.append(f)
    return found


def fmt_size(path: Path) -> str:
    try:
        b = path.stat().st_size
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024:
                return f"{b:.0f}{unit}"
            b /= 1024
        return f"{b:.1f}TB"
    except OSError:
        return "?"


# ── Command builder ───────────────────────────────────────────────────────────

def build_command(model: Path, cfg: dict) -> list:
    cmd = [LLAMA_SERVER, "-m", str(model)]
    visual = cfg.get("visual_model")
    if visual == "none":
        pass  # explicitly disabled
    elif visual and visual != "none":
        vpath = Path(visual)
        if vpath.exists():
            cmd += ["--mmproj", str(vpath)]
    else:
        # auto-detect from same folder
        mmproj = find_mmproj(model)
        if mmproj:
            cmd += ["--mmproj", str(mmproj)]
    cmd += ["-ngl", str(cfg["ngl"])]
    cmd += ["-c",   str(cfg["context"])]
    cmd += ["--threads", str(cfg["threads"])]
    if cfg["flash_attn"]:
        cmd += ["--flash-attn", "on"]
    cmd += ["--host", cfg["host"], "--port", str(cfg["port"])]
    if cfg.get("cache_type_k"):
        cmd += ["-ctk", cfg["cache_type_k"]]
    if cfg.get("cache_type_v"):
        cmd += ["-ctv", cfg["cache_type_v"]]
    if cfg.get("verbosity") is not None:
        cmd += ["--verbosity", str(cfg["verbosity"])]
    if cfg.get("batch"):
        cmd += ["-b", str(cfg["batch"])]
    if cfg.get("ubatch"):
        cmd += ["-ub", str(cfg["ubatch"])]
    if cfg.get("mlock"):
        cmd += ["--mlock"]
    if cfg.get("temp") is not None:
        cmd += ["--temp", str(cfg["temp"])]
    if cfg.get("top_p") is not None:
        cmd += ["--top-p", str(cfg["top_p"])]
    if cfg.get("top_k") is not None:
        cmd += ["--top-k", str(cfg["top_k"])]
    if cfg.get("thinking") is True:
        cmd += ["--reasoning", "on"]
    elif cfg.get("thinking") is False:
        cmd += ["--reasoning", "off"]
    if cfg.get("thinking_budget") is not None:
        cmd += ["--reasoning-budget", str(cfg["thinking_budget"])]
    if cfg.get("reasoning_format") is not None:
        cmd += ["--reasoning-format", cfg["reasoning_format"]]
    cmd += ["--parallel", "1"]
    if cfg.get("jinja"):
        cmd += ["--jinja"]
    if cfg.get("gemma4_template_fix"):
        cmd += ["--chat-template-file", r"C:\tools\llamacpp\templates\google-gemma-4-31B-it-interleaved.jinja"]
    if cfg.get("repeat_penalty") is not None:
        cmd += ["--repeat-penalty", str(cfg["repeat_penalty"])]
    return cmd


# ── Helpers ───────────────────────────────────────────────────────────────────

def short_label(model: Path, base_dirs):
    for b in base_dirs:
        try:
            return str(model.relative_to(b))
        except ValueError:
            pass
    return str(model)


def apply_sort(models, sort_mode, all_saved):
    if sort_mode == "recent":
        return sorted(models, key=lambda m: last_launch_time(m, all_saved), reverse=True)
    return models  # "name" — already sorted alphabetically from find_models()


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_list(stdscr, models, sel, cfg, base_dirs, all_saved, sort_mode,
              filter_str="", status=""):
    stdscr.clear()
    h, w = stdscr.getmaxyx()

    # Header
    header = (
        " llama.cpp Model Selector  |  "
        "arrows=navigate  enter=launch  s=settings  "
        "/=search  o=sort  d=del-settings  r=rescan  q=quit"
    )
    stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
    stdscr.addstr(0, 0, header[:w-1].ljust(w-1))
    stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

    # Settings bar
    if models:
        visual = cfg.get("visual_model")
        if visual == "none":
            vision_tag = " [V:off]"
        elif visual:
            vision_tag = f" [V:{Path(visual).name}]"
        else:
            mmproj = find_mmproj(models[sel])
            vision_tag = " [V]" if mmproj else ""
        ctk         = cfg.get("cache_type_k") or "-"
        ctv         = cfg.get("cache_type_v") or "-"
        verb        = cfg.get("verbosity")
        bat         = cfg.get("batch")
        ubat        = cfg.get("ubatch")
        saved_mark  = " [saved]" if str(models[sel]) in {k for k in all_saved if k != "__meta__"} else ""
        parts = [
            f"ctx={cfg['context']}",
            f"threads={cfg['threads']}",
            f"ngl={cfg['ngl']}",
            f"port={cfg['port']}",
            f"fa={'on' if cfg['flash_attn'] else 'off'}",
            f"ctk={ctk}",
            f"ctv={ctv}",
        ]
        if verb is not None:  parts.append(f"verb={verb}")
        if bat  is not None:  parts.append(f"b={bat}")
        if ubat is not None:  parts.append(f"ub={ubat}")
        if cfg.get("mlock"):  parts.append("mlock")
        if cfg.get("temp")  is not None: parts.append(f"temp={cfg['temp']}")
        if cfg.get("top_p") is not None: parts.append(f"top_p={cfg['top_p']}")
        if cfg.get("top_k") is not None: parts.append(f"top_k={cfg['top_k']}")
        thinking = cfg.get("thinking")
        if thinking is not None:
            parts.append(f"think={'on' if thinking else 'off'}")
        if cfg.get("thinking_budget") is not None:
            parts.append(f"budget={cfg['thinking_budget']}")
        if cfg.get("reasoning_format") is not None:
            parts.append(f"rfmt={cfg['reasoning_format']}")
        if cfg.get("jinja"):
            parts.append("jinja")
        if cfg.get("gemma4_template_fix"):
            parts.append("g4fix")
        if cfg.get("repeat_penalty") is not None:
            parts.append(f"rep={cfg['repeat_penalty']}")
        settings_str = " " + "  ".join(parts) + vision_tag + saved_mark
    else:
        settings_str = " (no models)"

    stdscr.attron(curses.color_pair(3))
    stdscr.addstr(1, 0, settings_str[:w-1].ljust(w-1))
    stdscr.attroff(curses.color_pair(3))

    # Filter bar
    sort_label = "recent" if sort_mode == "recent" else "name"
    filter_bar = f" Filter: {filter_str}_  [sort:{sort_label}]  {len(models)} models"
    stdscr.attron(curses.color_pair(6))
    stdscr.addstr(2, 0, filter_bar[:w-1].ljust(w-1))
    stdscr.attroff(curses.color_pair(6))

    # Model list
    list_start = 3
    list_h     = h - list_start - 2
    offset     = max(0, sel - list_h + 1) if sel >= list_h else 0

    for i, model in enumerate(models[offset:offset + list_h]):
        idx       = i + offset
        label     = short_label(model, base_dirs)
        size_str  = fmt_size(model)
        m_cfg = cfg_for_model(model, all_saved)
        m_visual = m_cfg.get("visual_model")
        if m_visual == "none":
            has_vision = False
        elif m_visual:
            has_vision = True
        else:
            has_vision = find_mmproj(model) is not None
        has_saved  = str(model) in {k for k in all_saved if k != "__meta__"}
        launched   = last_launch_time(model, all_saved)
        recency    = ">" if launched else " "
        vtag       = "[V]" if has_vision else "   "
        stag       = "*" if has_saved else " "
        line       = f" {recency}{vtag}{stag} {size_str:>7}  {label}"
        y = list_start + i
        if idx == sel:
            stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
            stdscr.addstr(y, 0, line[:w-1].ljust(w-1))
            stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
        else:
            stdscr.addstr(y, 0, line[:w-1])

    # Status / command preview
    if models and sel < len(models):
        preview = " ".join(build_command(models[sel], cfg))
        stdscr.attron(curses.color_pair(4))
        stdscr.addstr(h-1, 0, (" CMD: " + preview)[:w-1].ljust(w-1))
        stdscr.attroff(curses.color_pair(4))

    if status:
        stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
        stdscr.addstr(h-1, 0, status[:w-1].ljust(w-1))
        stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)

    stdscr.refresh()


# ── Inline value editor ───────────────────────────────────────────────────────

def inline_edit(stdscr, label, current_val):
    """Show a bottom-bar text input; returns parsed value or None on cancel."""
    h, w = stdscr.getmaxyx()
    curses.curs_set(1)
    buf = "" if current_val is None else str(current_val)
    while True:
        prompt = f" Enter {label} (blank=default, Esc=cancel): {buf}_"
        stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        stdscr.addstr(h - 1, 0, prompt[:w - 1].ljust(w - 1))
        stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
        stdscr.refresh()
        ch = stdscr.getch()
        if ch in (10, 13):          # Enter — confirm
            curses.curs_set(0)
            s = buf.strip()
            if s == "" or s.lower() == "none":
                return None
            return s
        elif ch == 27:              # Esc — cancel
            curses.curs_set(0)
            return current_val      # unchanged
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            buf = buf[:-1]
        elif 32 <= ch <= 126:
            buf += chr(ch)


# ── Settings menu ─────────────────────────────────────────────────────────────

def settings_menu(stdscr, cfg):
    # Build visual model options: None (auto), "none" (disabled), then all found mmproj files
    all_mmproj = find_all_mmproj()
    visual_options = [None, "none"] + [str(f) for f in all_mmproj]
    visual_labels = {None: "auto (same folder)", "none": "disabled"}
    for f in all_mmproj:
        visual_labels[str(f)] = f"{f.name}  ({f.parent})"

    fields = [
        ("context",      "Context length",     CONTEXT_OPTIONS),
        ("threads",      "Threads",            THREAD_OPTIONS),
        ("ngl",          "GPU layers (-ngl)",  None),
        ("port",         "Port",               None),
        ("flash_attn",   "Flash attention",    [True, False]),
        ("cache_type_k", "Cache K (-ctk)",     CACHE_OPTIONS),
        ("cache_type_v", "Cache V (-ctv)",     CACHE_OPTIONS),
        ("verbosity",    "Verbosity",          [None, 0, 1, 2, 3, 4, 5]),
        ("batch",        "Batch size (-b)",    BATCH_OPTIONS),
        ("ubatch",       "Micro-batch (-ub)",  BATCH_OPTIONS),
        ("mlock",          "mlock (pin in RAM)",      [False, True]),
        ("temp",           "Temperature (--temp)",    TEMP_OPTIONS),
        ("top_p",          "Top-P (--top-p)",         TOP_P_OPTIONS),
        ("top_k",          "Top-K (--top-k)",         TOP_K_OPTIONS),
        ("thinking",         "Thinking (on/off)",       THINKING_OPTIONS),
        ("thinking_budget",  "Thinking budget (tokens)", THINKING_BUDGET_OPTIONS),
        ("reasoning_format", "Reasoning format",        REASONING_FORMAT_OPTIONS),
        ("jinja",            "Jinja templates (--jinja)", [False, True]),
        ("gemma4_template_fix", "Gemma 4 template fix",  [False, True]),
        ("repeat_penalty", "Repeat penalty",           REPEAT_PENALTY_OPTIONS),
        ("visual_model",   "Visual model (mmproj)",    visual_options),
    ]
    sel = 0

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        stdscr.addstr(0, 0,
            " Settings  |  up/down=field  left/right or +/-=value  enter=edit[*]  q=back"
            .ljust(w-1))
        stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

        for i, (key, label, options) in enumerate(fields):
            val = cfg.get(key)
            if key == "thinking":
                display = {None: "default", True: "on", False: "off"}.get(val, str(val))
            elif key == "visual_model":
                display = visual_labels.get(val, Path(val).name if val else "auto (same folder)")
            else:
                display = str(val) if val is not None else "default"
            editable_marker = "[*]" if key in EDITABLE_FIELDS else "   "
            line = f"  {editable_marker} {label:<26} {display}"
            if i == sel:
                stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                stdscr.addstr(2 + i, 0, line[:w-1].ljust(w-1))
                stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            else:
                stdscr.addstr(2 + i, 0, line[:w-1])

        stdscr.refresh()
        key = stdscr.getch()

        if key in (ord('q'), ord('s'), 27):
            break
        elif key in (10, 13):
            fkey, flabel, _ = fields[sel]
            if fkey in EDITABLE_FIELDS:
                raw = inline_edit(stdscr, flabel, cfg.get(fkey))
                if raw is None:
                    cfg[fkey] = None
                else:
                    try:
                        cfg[fkey] = int(raw) if fkey == "top_k" else float(raw)
                    except ValueError:
                        pass  # leave unchanged on bad input
            else:
                break
        elif key == curses.KEY_UP:
            sel = (sel - 1) % len(fields)
        elif key == curses.KEY_DOWN:
            sel = (sel + 1) % len(fields)
        elif key in (curses.KEY_LEFT, curses.KEY_RIGHT, ord('+'), ord('-')):
            fkey, _, options = fields[sel]
            direction = 1 if key in (curses.KEY_RIGHT, ord('+')) else -1
            if options is not None:
                cur = cfg.get(fkey)
                idx = options.index(cur) if cur in options else 0
                cfg[fkey] = options[(idx + direction) % len(options)]
            else:
                if fkey == "ngl":
                    cfg[fkey] = max(0, cfg[fkey] + direction)
                elif fkey == "port":
                    cfg[fkey] = max(1024, cfg[fkey] + direction)


# ── Search / filter ───────────────────────────────────────────────────────────

def filter_models(all_models, query: str):
    if not query:
        return all_models
    q = query.lower()
    return [m for m in all_models if q in m.name.lower() or q in str(m).lower()]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(stdscr):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE,  curses.COLOR_BLUE)    # header
    curses.init_pair(2, curses.COLOR_BLACK,  curses.COLOR_YELLOW)  # selected
    curses.init_pair(3, curses.COLOR_CYAN,   -1)                   # settings bar
    curses.init_pair(4, curses.COLOR_WHITE,  curses.COLOR_BLACK)   # cmd preview
    curses.init_pair(5, curses.COLOR_GREEN,  -1)                   # status
    curses.init_pair(6, curses.COLOR_YELLOW, -1)                   # filter bar

    base_dirs = [Path(d) for d in MODEL_DIRS]

    stdscr.addstr(0, 0, "Scanning for models...")
    stdscr.refresh()

    all_models = find_models()
    all_saved  = load_saved_settings()

    if not all_models:
        stdscr.clear()
        stdscr.addstr(0, 0, "No GGUF models found in " + ", ".join(MODEL_DIRS))
        stdscr.addstr(1, 0, "Press any key to exit.")
        stdscr.getch()
        return

    sort_mode  = "name"
    filter_str = ""
    models     = apply_sort(filter_models(all_models, filter_str), sort_mode, all_saved)

    # Restore cursor to last used model
    last_model_str = all_saved.get("__meta__", {}).get("__last_model__")
    sel = 0
    if last_model_str:
        last_path = Path(last_model_str)
        if last_path in models:
            sel = models.index(last_path)

    cfg        = cfg_for_model(models[sel], all_saved)
    status     = f"Found {len(all_models)} models.   [V]=vision  *=saved  >=launched"

    while True:
        draw_list(stdscr, models, sel, cfg, base_dirs, all_saved,
                  sort_mode, filter_str, status)
        key = stdscr.getch()
        status = ""

        # ── Quit ──
        if key in (ord('q'), ord('Q'), 27):
            break

        # ── Navigation ──
        elif key == curses.KEY_UP:
            sel = (sel - 1) % len(models) if models else 0
            cfg = cfg_for_model(models[sel], all_saved)

        elif key == curses.KEY_DOWN:
            sel = (sel + 1) % len(models) if models else 0
            cfg = cfg_for_model(models[sel], all_saved)

        elif key == curses.KEY_PPAGE:
            sel = max(0, sel - 10)
            cfg = cfg_for_model(models[sel], all_saved)

        elif key == curses.KEY_NPAGE:
            sel = min(len(models) - 1, sel + 10)
            cfg = cfg_for_model(models[sel], all_saved)

        # ── Settings ──
        elif key in (ord('s'), ord('S')):
            settings_menu(stdscr, cfg)
            persist_cfg(models[sel], cfg, all_saved, is_launch=False)
            status = "Settings saved."

        # ── Delete saved settings ──
        elif key in (ord('d'), ord('D')):
            if models:
                delete_cfg(models[sel], all_saved)
                cfg = cfg_for_model(models[sel], all_saved)
                status = "Saved settings cleared for this model."

        # ── Sort toggle ──
        elif key in (ord('o'), ord('O')):
            sort_mode = "recent" if sort_mode == "name" else "name"
            cur_model = models[sel] if models else None
            models = apply_sort(filter_models(all_models, filter_str), sort_mode, all_saved)
            sel = models.index(cur_model) if cur_model in models else 0
            status = f"Sorted by {'last launched' if sort_mode == 'recent' else 'name'}."

        # ── Rescan ──
        elif key in (ord('r'), ord('R')):
            stdscr.addstr(0, 0, "Rescanning...")
            stdscr.refresh()
            cur_model  = models[sel] if models else None
            all_models = find_models()
            models     = apply_sort(filter_models(all_models, filter_str), sort_mode, all_saved)
            sel        = models.index(cur_model) if cur_model in models else 0
            cfg        = cfg_for_model(models[sel], all_saved) if models else dict(DEFAULTS)
            status     = f"Rescan complete. {len(all_models)} models found."

        # ── Search / filter ──
        elif key == ord('/'):
            # Enter filter mode — read characters until Enter/Esc
            curses.curs_set(1)
            filter_str = ""
            cur_model  = models[sel] if models else None
            while True:
                models = apply_sort(filter_models(all_models, filter_str), sort_mode, all_saved)
                sel    = 0
                if cur_model in models:
                    sel = models.index(cur_model)
                cfg = cfg_for_model(models[sel], all_saved) if models else dict(DEFAULTS)
                draw_list(stdscr, models, sel, cfg, base_dirs, all_saved,
                          sort_mode, filter_str, f"Filter mode — type to search, Enter/Esc to confirm")
                fkey = stdscr.getch()
                if fkey in (10, 13, 27):
                    break
                elif fkey in (curses.KEY_BACKSPACE, 127, 8):
                    filter_str = filter_str[:-1]
                elif 32 <= fkey <= 126:
                    filter_str += chr(fkey)
            curses.curs_set(0)
            status = f"Filter: '{filter_str}'  ({len(models)} results)" if filter_str else "Filter cleared."

        # ── Launch ──
        elif key in (10, 13, curses.KEY_ENTER):
            if not models:
                status = "No models to launch."
                continue
            model = models[sel]
            persist_cfg(model, cfg, all_saved, is_launch=True)
            cmd = build_command(model, cfg)
            curses.endwin()
            print("Launching:")
            print(" ".join(cmd))
            print()
            subprocess.run(cmd)
            return


if __name__ == "__main__":
    # Request terminal resize via VT escape sequence (works in Windows Terminal)
    sys.stdout.write("\033[8;50;220t")
    sys.stdout.flush()
    curses.wrapper(main)

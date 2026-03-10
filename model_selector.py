#!/usr/bin/env python3
r"""
llama.cpp model selector - keyboard-driven TUI
Scans C:\llm and E:\llm recursively for GGUF files, detects mmproj (vision),
and launches llama-server with appropriate settings.
Per-model settings are saved to model_settings.json next to this script.
"""

import sys
import json
import subprocess
import curses
from pathlib import Path

LLAMA_SERVER = r"C:\tools\llamacpp\llama-server.exe"
MODEL_DIRS = [r"C:\llm", r"E:\llm"]
SETTINGS_FILE = Path(__file__).parent / "model_settings.json"

DEFAULTS = {
    "ngl": 999,
    "threads": 12,
    "context": 32768,
    "host": "REDACTED_HOST",
    "port": 8080,
    "flash_attn": True,
    "cache_type_k": None,
    "cache_type_v": None,
    "verbosity": None,
}

CONTEXT_OPTIONS = [4096, 8192, 16384, 32768, 49152, 65536, 72000, 80000, 90000, 131072, 200000]
THREAD_OPTIONS  = [12, 16, 20, 24, 32]
CACHE_OPTIONS   = [None, "q8_0", "q6_0", "q5_0", "q4_0", "q3_0"]


# ── Persistence ──────────────────────────────────────────────────────────────

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
    key = str(model)
    saved = all_saved.get(key, {})
    cfg = dict(DEFAULTS)
    cfg.update(saved)
    return cfg


def persist_cfg(model: Path, cfg: dict, all_saved: dict):
    # Only save keys that differ from defaults (keeps file tidy)
    delta = {k: v for k, v in cfg.items() if v != DEFAULTS.get(k)}
    all_saved[str(model)] = delta
    save_settings(all_saved)


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


# ── Command builder ───────────────────────────────────────────────────────────

def build_command(model: Path, cfg: dict) -> list:
    cmd = [LLAMA_SERVER, "-m", str(model)]
    mmproj = find_mmproj(model)
    if mmproj:
        cmd += ["--mmproj", str(mmproj), "--image-min-tokens", "1024"]
    cmd += ["-ngl", str(cfg["ngl"])]
    cmd += ["-c", str(cfg["context"])]
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
    if cfg.get("np"):
        cmd += ["-np", str(cfg["np"])]
    return cmd


# ── Helpers ───────────────────────────────────────────────────────────────────

def short_label(model: Path, base_dirs):
    for b in base_dirs:
        try:
            return str(model.relative_to(b))
        except ValueError:
            pass
    return str(model)


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_list(stdscr, models, sel, cfg, base_dirs, all_saved, status=""):
    stdscr.clear()
    h, w = stdscr.getmaxyx()

    header = " llama.cpp Model Selector  |  arrows=navigate  enter=launch  s=settings  q=quit"
    stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
    stdscr.addstr(0, 0, header[:w-1].ljust(w-1))
    stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

    mmproj = find_mmproj(models[sel]) if models else None
    vision_tag = " [VISION]" if mmproj else ""
    ctk = cfg.get("cache_type_k") or "default"
    ctv = cfg.get("cache_type_v") or "default"
    verb = cfg.get("verbosity")
    verb_str = f"  verb={verb}" if verb is not None else ""
    saved_marker = " [saved]" if str(models[sel]) in all_saved else ""
    settings_str = (
        f" ctx={cfg['context']}  threads={cfg['threads']}  "
        f"ngl={cfg['ngl']}  port={cfg['port']}  "
        f"fa={'on' if cfg['flash_attn'] else 'off'}  "
        f"ctk={ctk}  ctv={ctv}{verb_str}"
        f"{vision_tag}{saved_marker}"
    )
    stdscr.attron(curses.color_pair(3))
    stdscr.addstr(1, 0, settings_str[:w-1].ljust(w-1))
    stdscr.attroff(curses.color_pair(3))

    list_start = 3
    list_h = h - list_start - 3
    offset = max(0, sel - list_h + 1) if sel >= list_h else 0

    for i, model in enumerate(models[offset:offset+list_h]):
        idx = i + offset
        label = short_label(model, base_dirs)
        has_vision = find_mmproj(model) is not None
        has_saved  = str(model) in all_saved
        tag = ("[V]" if has_vision else "   ") + ("*" if has_saved else " ")
        line = f" {tag} {label}"
        y = list_start + i
        if idx == sel:
            stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
            stdscr.addstr(y, 0, line[:w-1].ljust(w-1))
            stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
        else:
            stdscr.addstr(y, 0, line[:w-1])

    if models:
        preview = " ".join(build_command(models[sel], cfg))
        stdscr.attron(curses.color_pair(4))
        stdscr.addstr(h-2, 0, (" CMD: " + preview)[:w-1].ljust(w-1))
        stdscr.attroff(curses.color_pair(4))

    if status:
        stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
        stdscr.addstr(h-1, 0, status[:w-1].ljust(w-1))
        stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)

    stdscr.refresh()


# ── Settings menu ─────────────────────────────────────────────────────────────

def settings_menu(stdscr, cfg):
    fields = [
        ("context",      "Context length",    CONTEXT_OPTIONS),
        ("threads",      "Threads",           THREAD_OPTIONS),
        ("ngl",          "GPU layers (-ngl)", None),
        ("port",         "Port",              None),
        ("flash_attn",   "Flash attention",   [True, False]),
        ("cache_type_k", "Cache K (-ctk)",    CACHE_OPTIONS),
        ("cache_type_v", "Cache V (-ctv)",    CACHE_OPTIONS),
        ("verbosity",    "Verbosity",         [None, 0, 1, 2, 3, 4, 5]),
    ]
    sel = 0

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        stdscr.addstr(0, 0, " Settings  |  up/down=field  left/right=value  enter/q=back".ljust(w-1))
        stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

        for i, (key, label, options) in enumerate(fields):
            val = cfg.get(key)
            display = str(val) if val is not None else "default"
            line = f"  {label:<22} {display}"
            if i == sel:
                stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                stdscr.addstr(2 + i, 0, line[:w-1].ljust(w-1))
                stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            else:
                stdscr.addstr(2 + i, 0, line[:w-1])

        stdscr.addstr(2 + len(fields) + 1, 2,
                      "left/right or +/- to change value   q/enter to go back")
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord('q'), ord('s'), 27, 10, 13):
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
                step = 1
                if fkey == "ngl":
                    cfg[fkey] = max(0, cfg[fkey] + direction * step)
                elif fkey == "port":
                    cfg[fkey] = max(1024, cfg[fkey] + direction * step)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(stdscr):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_YELLOW)
    curses.init_pair(3, curses.COLOR_CYAN,  -1)
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_GREEN, -1)

    base_dirs = [Path(d) for d in MODEL_DIRS]
    stdscr.addstr(0, 0, "Scanning for models...")
    stdscr.refresh()

    models = find_models()
    all_saved = load_saved_settings()

    if not models:
        stdscr.clear()
        stdscr.addstr(0, 0, "No GGUF models found in " + ", ".join(MODEL_DIRS))
        stdscr.addstr(1, 0, "Press any key to exit.")
        stdscr.getch()
        return

    sel = 0
    cfg = cfg_for_model(models[sel], all_saved)
    status = f"Found {len(models)} models.  * = saved settings"

    while True:
        draw_list(stdscr, models, sel, cfg, base_dirs, all_saved, status)
        key = stdscr.getch()

        if key in (ord('q'), ord('Q'), 27):
            break

        elif key == curses.KEY_UP:
            sel = (sel - 1) % len(models)
            cfg = cfg_for_model(models[sel], all_saved)
            status = ""

        elif key == curses.KEY_DOWN:
            sel = (sel + 1) % len(models)
            cfg = cfg_for_model(models[sel], all_saved)
            status = ""

        elif key == curses.KEY_PPAGE:
            sel = max(0, sel - 10)
            cfg = cfg_for_model(models[sel], all_saved)
            status = ""

        elif key == curses.KEY_NPAGE:
            sel = min(len(models) - 1, sel + 10)
            cfg = cfg_for_model(models[sel], all_saved)
            status = ""

        elif key in (ord('s'), ord('S')):
            settings_menu(stdscr, cfg)
            persist_cfg(models[sel], cfg, all_saved)
            status = "Settings saved."

        elif key in (10, 13, curses.KEY_ENTER):
            model = models[sel]
            persist_cfg(model, cfg, all_saved)
            cmd = build_command(model, cfg)
            curses.endwin()
            print("Launching:")
            print(" ".join(cmd))
            print()
            subprocess.run(cmd)
            return


if __name__ == "__main__":
    curses.wrapper(main)

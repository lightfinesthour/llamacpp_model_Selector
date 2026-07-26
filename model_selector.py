#!/usr/bin/env python3
r"""
llama.cpp model selector - keyboard-driven TUI
Scans C:\llm and E:\llm recursively for GGUF files, detects mmproj (vision),
and launches llama-server with appropriate settings.
Per-model settings are saved to model_settings.json next to this script.
"""

import sys
import os
import re
import json
import subprocess
import curses
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

LLAMA_SERVER = r"C:\tools\llamacpp\llama-server.exe"
MODEL_DIRS   = [r"C:\llm", r"E:\llm", r"K:\models", r"M:\models"]
TEMPLATES_DIR = Path(r"C:\tools\llamacpp\templates")
SETTINGS_FILE = Path(__file__).parent / "model_settings.json"

DEFAULTS = {
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
    "no_mmap":         True,
    "mlock":           True,
    "temp":            0.6,
    "top_p":           0.95,
    "top_k":           20,
    "min_p":           None,
    # thinking=True emits --reasoning on, which makes the SERVER DEFAULT
    # thinking-on. Note: the built-in WebUI can still override this per-request
    # if its custom-JSON field sends chat_template_kwargs {"enable_thinking":
    # false}. Keep that WebUI field EMPTY (or set it to enable_thinking: true)
    # so the server default wins. The launcher cannot control that field; it is
    # a per-request browser setting.
    "thinking":         None,
    "thinking_budget":  None,
    # deepseek puts thoughts in message.reasoning_content (clean separation the
    # WebUI renders as a Thinking block). 'auto' also works; 'none' leaves
    # thoughts inline in content.
    "reasoning_format": "deepseek",
    # Chat-template kwarg for templates that gate thinking on reasoning_effort
    # instead of enable_thinking (Hunyuan V3: no_think/low/high, defaults
    # no_think; gpt-oss: low/medium/high). Sent as
    # --chat-template-kwargs '{"reasoning_effort": ...}'; forces --jinja.
    # None = flag omitted, template default applies.
    "reasoning_effort": None,
    # Keep thinking blocks from earlier assistant turns in the history instead
    # of only the last one. Only works on templates that read a preserve_thinking
    # kwarg (llama-server logs "chat template supports preserving reasoning" at
    # startup when it applies). Better multi-turn coherence, costs context.
    # None = flag omitted, template default applies.
    "reasoning_preserve": None,
    "repeat_penalty":   1.05,
    "jinja":            False,
    "auto_template":    False,   # Auto-match a curated .jinja template by model name
    "draft_mtp":        False,       # Enable Multi-Token Prediction speculative decoding
    "visual_model":     "none",     # None=auto (same folder), "none"=disabled, or path to mmproj
}

CONTEXT_OPTIONS  = [4096, 8192, 16384, 32768, 49152, 65536, 72000, 80000, 90000, 131072, 150000, 196608, 200000, 262144]
THREAD_OPTIONS   = [4, 8, 12, 16, 20, 24, 32]
CACHE_OPTIONS    = [None, "f16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl"]
BATCH_OPTIONS    = [None, 256, 512, 1024, 2048, 4096]
TEMP_OPTIONS     = [None, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.2, 1.5]
TOP_P_OPTIONS    = [None, 0.1, 0.5, 0.8, 0.9, 0.95, 1.0]
TOP_K_OPTIONS    = [None, 0, 10, 20, 40, 80, 100]
MIN_P_OPTIONS    = [None, 0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
THINKING_OPTIONS         = [None, True, False]    # None=default, True=on, False=off
THINKING_BUDGET_OPTIONS  = [None, 0, 256, 1024, 4096, 8192, 16384, 32768]
REASONING_FORMAT_OPTIONS = [None, "deepseek", "deepseek-legacy", "none"]
# deepseek        → extracts thinking into reasoning_content (Open WebUI shows collapsible dropdown)
# deepseek-legacy → keeps <think> tags in content but also populates reasoning_content
# none            → strips all thinking tags from output entirely
# Union of values seen across templates; the settings menu narrows this to the
# values the selected model's embedded template actually accepts.
REASONING_EFFORT_OPTIONS = [None, "no_think", "low", "medium", "high"]
REASONING_PRESERVE_OPTIONS = [None, True, False]  # None=template default
REPEAT_PENALTY_OPTIONS   = [None, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5]

# Fields that support direct text entry for precision
EDITABLE_FIELDS = {"temp", "top_p", "top_k", "min_p", "repeat_penalty"}


# ── Hard-coded per-model fixes ────────────────────────────────────────────────
# Applied automatically in build_command when the model path contains "match"
# (case-insensitive). Overrides win over saved and default settings; no user
# action needed. The settings bar shows FIX:<name> when one is active.
#
# deepseek-v4-flash (C:\llm\unsloth\DeepSeek-V4-Flash-GGUF):
#   - The unsloth GGUF embeds a fixed DSML jinja template that is only honored
#     on the --jinja code path, so jinja is forced on.
#   - DeepSeek recommends temp=1.0, top_p=1.0, min_p=0.0 with no top-k and no
#     repeat penalty; the launcher defaults (0.6 / 0.95 / 20 / 1.05) degrade it.
#   - llama.cpp context checkpointing corrupts multi-turn output for this arch
#     (gibberish from turn 2; fix PR #25402 not merged as of build 9898). A
#     llama-server built from that PR lives in patches\deepseekv4flash and is
#     used via "server" when present. If it is missing, the stock server runs
#     with checkpoints disabled via "fallback_args" instead.
MODEL_FIXES = [
    {
        "name": "dsv4-flash",
        "match": "deepseek-v4-flash",
        "overrides": {
            "jinja":          True,
            "temp":           1.0,
            "top_p":          1.0,
            "top_k":          0,      # 0 = top-k disabled in llama.cpp
            "min_p":          0.0,
            "repeat_penalty": None,   # flag omitted; server default 1.0 = off
        },
        "server": r"C:\tools\llamacpp\patches\deepseekv4flash\llama-server.exe",
        "fallback_args": ["--ctx-checkpoints", "0"],
    },
]


def find_model_fix(model: Path):
    p = str(model).lower()
    for fix in MODEL_FIXES:
        if fix["match"] in p:
            return fix
    return None


# ── Persistence ───────────────────────────────────────────────────────────────

def load_saved_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text())
            return _migrate_settings(data)
        except Exception:
            pass
    return {}


def _migrate_settings(data: dict) -> dict:
    """Rename the retired gemma4_template_fix key to auto_template in place."""
    for key, val in data.items():
        if key == "__meta__" or not isinstance(val, dict):
            continue
        if "gemma4_template_fix" in val:
            val["auto_template"] = val.pop("gemma4_template_fix")
    return data


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
            # Split GGUFs: only the first shard is launchable; llama-server
            # picks up the rest of the -NNNNN-of-NNNNN set automatically.
            shard = re.search(r"-(\d{5})-of-\d{5}\.gguf$", name)
            if shard and shard.group(1) != "00001":
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


def model_total_bytes(path: Path) -> int:
    """Size of the model on disk; for split GGUFs, the sum of all shards."""
    m = re.match(r"(.+)-\d{5}-of-(\d{5})\.gguf$", path.name, re.IGNORECASE)
    if not m:
        return path.stat().st_size
    prefix, count = m.group(1), m.group(2)
    shard_re = re.compile(re.escape(prefix) + r"-\d{5}-of-" + count + r"\.gguf$",
                          re.IGNORECASE)
    total = 0
    for f in path.parent.iterdir():
        if shard_re.match(f.name):
            total += f.stat().st_size
    return total


def fmt_size(path: Path) -> str:
    try:
        b = model_total_bytes(path)
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024:
                return f"{b:.0f}{unit}"
            b /= 1024
        return f"{b:.1f}TB"
    except OSError:
        return "?"


# ── Chat template matching ────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    """Lowercase and strip separators so 'Qwen3.5-4B' ~ 'qwen3-5-4b' ~ 'qwen354b'."""
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
    return "".join(out)


def _template_tokens(stem: str) -> set:
    """Split a template/model name into comparable family+version tokens."""
    norm = stem.lower()
    for sep in ("-", "_", ".", " "):
        norm = norm.replace(sep, " ")
    return {t for t in norm.split() if t}


# Explicit model-name -> template overrides, checked BEFORE the fuzzy matcher.
# Each entry is (substring matched case-insensitively against the model
# filename, template filename in TEMPLATES_DIR). Use this when the right
# template does not share a name with the model, which the token scorer below
# can never work out on its own.
TEMPLATE_OVERRIDES = [
    # Laguna S 2.1 (and derivatives like -CRACK) shipped with a template that
    # stopped reasoning after turn 1. Poolside published a fixed one in
    # poolside/Laguna-S-2.1-GGUF (chat_template.jinja): enable_thinking now
    # defaults true, preserve_thinking is honoured, and the generation prompt
    # ends on an open <think>. That is what Laguna-S-2.1.jinja holds. Pair it
    # with reasoning_preserve=true so prior turns carry real reasoning_content
    # instead of an empty <think></think>. Older GGUFs still embed the broken
    # template, so keep this override until the quants are re-uploaded.
    ("laguna", "Laguna-S-2.1.jinja"),
]


def find_template_override(model: Path):
    """Return the explicitly mapped template for this model, or None."""
    name = model.name.lower()
    for needle, tpl_name in TEMPLATE_OVERRIDES:
        if needle in name:
            tpl = TEMPLATES_DIR / tpl_name
            if tpl.exists():
                return tpl
    return None


def find_template_for_model(model: Path):
    """
    Best-effort match of a local model file to a curated .jinja template in
    TEMPLATES_DIR by name. Returns the template Path or None.

    Strategy: an explicit TEMPLATE_OVERRIDES hit wins outright; otherwise score
    each template by how many of its name tokens appear in the model filename,
    requiring the model-family token (first alpha token) to match. Returns the
    highest-scoring template above a small threshold.
    """
    if not TEMPLATES_DIR.exists():
        return None
    override = find_template_override(model)
    if override is not None:
        return override
    model_norm = _normalize(model.stem)
    model_tokens = _template_tokens(model.stem)

    best = None
    best_score = 0
    for tpl in TEMPLATES_DIR.glob("*.jinja"):
        tpl_tokens = _template_tokens(tpl.stem)
        # Drop generic role suffixes and org-prefix noise that don't identify
        # the model family (e.g. the "ai" in "deepseek-ai", "forai", "hf").
        core = tpl_tokens - {"it", "instruct", "tool", "use", "default",
                             "rag", "interleaved", "fixed", "bf16",
                             "ai", "org", "hf", "team", "research", "forai"}
        if not core:
            continue
        # Score by token overlap plus a strong bonus when the template's
        # normalized stem appears directly in the model filename.
        overlap = len(core & model_tokens)
        # Strongest signal: the template's normalized stem appears in the model.
        stem_hit = _normalize(tpl.stem.replace("interleaved", "")
                                       .replace("fixed", "")) in model_norm
        score = overlap + (5 if stem_hit else 0)
        # Require a real family-name match, not just a shared size/quant token
        # like "8b". The model must contain at least one alphabetic core token
        # (e.g. "gemma", "deepseek", "qwen"); org prefixes were already dropped.
        alpha_core = {t for t in core if any(c.isalpha() for c in t)}
        shared_alpha = {t for t in alpha_core if _normalize(t) in model_norm}
        if not shared_alpha:
            continue
        if score > best_score:
            best_score, best = score, tpl
    # Need the family token plus at least one more corroborating token.
    return best if best_score >= 2 else None


# ── GGUF template inspection ──────────────────────────────────────────────────
# Header-only read of the embedded jinja chat template, then a plain text scan
# for the thinking-related kwargs the template reads (enable_thinking vs
# reasoning_effort) and, when stated as a literal list, the values it accepts.
# No jinja execution, so it cannot follow computed logic; unknown templates
# just fall back to the full option list.

_GGUF_VALUE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_gguf_chat_template(model: Path):
    """Return tokenizer.chat_template from a GGUF header, or None."""
    try:
        with open(model, "rb") as f:
            def u32(): return int.from_bytes(f.read(4), "little")
            def u64(): return int.from_bytes(f.read(8), "little")

            def skip_value(vtype):
                if vtype == 8:                       # string
                    f.seek(u64(), 1)
                elif vtype == 9:                     # array
                    etype, n = u32(), u64()
                    if etype == 8:
                        for _ in range(n):
                            f.seek(u64(), 1)
                    elif etype == 9:
                        for _ in range(n):
                            skip_value(9)
                    else:
                        f.seek(_GGUF_VALUE_SIZES[etype] * n, 1)
                else:
                    f.seek(_GGUF_VALUE_SIZES[vtype], 1)

            if f.read(4) != b"GGUF":
                return None
            u32()                                    # version
            u64()                                    # tensor count
            for _ in range(u64()):
                key = f.read(u64()).decode("utf-8", "replace")
                vtype = u32()
                if key == "tokenizer.chat_template" and vtype == 8:
                    return f.read(u64()).decode("utf-8", "replace")
                skip_value(vtype)
    except Exception:
        return None
    return None


_template_info_cache = {}


def template_reasoning_info(model: Path) -> dict:
    """Which thinking switches the model's embedded template understands."""
    key = str(model)
    if key not in _template_info_cache:
        tpl = read_gguf_chat_template(model)
        info = {
            "has_template":     tpl is not None,
            "enable_thinking":  bool(tpl) and "enable_thinking" in tpl,
            "reasoning_effort": bool(tpl) and "reasoning_effort" in tpl,
            "effort_values":    None,
        }
        if info["reasoning_effort"]:
            # e.g. {%- elif reasoning_effort not in ['high', 'low', 'no_think'] %}
            m = re.search(r"reasoning_effort\s+(?:not\s+)?in\s*\[([^\]]*)\]", tpl)
            if m:
                vals = re.findall(r"""['"]([^'"]+)['"]""", m.group(1))
                if vals:
                    info["effort_values"] = vals
        _template_info_cache[key] = info
    return _template_info_cache[key]


# ── Command builder ───────────────────────────────────────────────────────────

def build_command(model: Path, cfg: dict) -> list:
    server = LLAMA_SERVER
    fix_args = []
    fix = find_model_fix(model)
    if fix:
        cfg = {**cfg, **fix["overrides"]}
        patched = fix.get("server")
        if patched and Path(patched).exists():
            server = patched
        else:
            fix_args = fix.get("fallback_args", [])
    cmd = [server, "-m", str(model)]
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
    cmd += ["-ngl", "-1"]
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
    if cfg.get("no_mmap"):
        cmd += ["--no-mmap"]
    if cfg.get("mlock"):
        cmd += ["--mlock"]
    if cfg.get("temp") is not None:
        cmd += ["--temp", str(cfg["temp"])]
    if cfg.get("top_p") is not None:
        cmd += ["--top-p", str(cfg["top_p"])]
    if cfg.get("top_k") is not None:
        cmd += ["--top-k", str(cfg["top_k"])]
    if cfg.get("min_p") is not None:
        cmd += ["--min-p", str(cfg["min_p"])]
    # --reasoning on/off sets the server-side thinking default. If reasoning
    # still does not appear in the built-in WebUI, check the WebUI's custom-JSON
    # field: an un-nested {"enable_thinking": true} does NOTHING (the template
    # only reads it nested as {"chat_template_kwargs": {"enable_thinking":
    # true}}), and the WebUI's default sends enable_thinking:false, which wins
    # over this flag per-request. Keeping that field EMPTY lets this flag win.
    if cfg.get("thinking") is True:
        cmd += ["--reasoning", "on"]
    elif cfg.get("thinking") is False:
        cmd += ["--reasoning", "off"]
    if cfg.get("thinking_budget") is not None:
        cmd += ["--reasoning-budget", str(cfg["thinking_budget"])]
    if cfg.get("reasoning_format") is not None:
        cmd += ["--reasoning-format", cfg["reasoning_format"]]
    # Thinking is driven by --reasoning on/off (emitted above). The current
    # llama.cpp build reads it from the template's thinking flag; the older
    # --chat-template-kwargs '{"enable_thinking": ...}' path is now deprecated
    # and warns at startup, so we no longer emit it. Templates that instead
    # gate thinking on a reasoning_effort kwarg (Hunyuan V3, gpt-oss) ignore
    # enable_thinking, so --reasoning on alone cannot enable thinking there;
    # the reasoning_effort setting reaches them via --chat-template-kwargs.
    if cfg.get("reasoning_effort") is not None:
        cmd += ["--chat-template-kwargs",
                json.dumps({"reasoning_effort": cfg["reasoning_effort"]})]
    if cfg.get("reasoning_preserve") is True:
        cmd += ["--reasoning-preserve"]
    elif cfg.get("reasoning_preserve") is False:
        cmd += ["--no-reasoning-preserve"]
    # Auto-match a curated chat template by model name. When auto_template is on,
    # find the best local .jinja for this model and pass it explicitly. Falls
    # back to the embedded template. TEMPLATE_OVERRIDES entries are explicit
    # per-model rules rather than guesses, so they apply even with auto_template
    # off: those models are known-broken on their embedded template.
    matched_template = None
    if cfg.get("auto_template"):
        matched_template = find_template_for_model(model)
    else:
        matched_template = find_template_override(model)
    # A custom --chat-template-file is only honored on the Jinja code path
    # (jinja is on by default in current builds, but force it to be explicit).
    # --chat-template-kwargs likewise only applies under Jinja.
    use_jinja = (cfg.get("jinja") or cfg.get("auto_template")
                 or cfg.get("reasoning_effort") is not None
                 or matched_template is not None)
    if use_jinja:
        cmd += ["--jinja"]
    if matched_template is not None:
        cmd += ["--chat-template-file", str(matched_template)]
    if cfg.get("draft_mtp"):
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    if cfg.get("repeat_penalty") is not None:
        cmd += ["--repeat-penalty", str(cfg["repeat_penalty"])]
    if fix_args:
        cmd += fix_args
    cmd += ["--parallel", "1"]
    #cmd += ["--no-warmup"]
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
            f"port={cfg['port']}",
            f"fa={'on' if cfg['flash_attn'] else 'off'}",
            f"ctk={ctk}",
            f"ctv={ctv}",
        ]
        if verb is not None:  parts.append(f"verb={verb}")
        if bat  is not None:  parts.append(f"b={bat}")
        if ubat is not None:  parts.append(f"ub={ubat}")
        if cfg.get("no_mmap"): parts.append("no-mmap")
        if cfg.get("mlock"):  parts.append("mlock")
        if cfg.get("temp")  is not None: parts.append(f"temp={cfg['temp']}")
        if cfg.get("top_p") is not None: parts.append(f"top_p={cfg['top_p']}")
        if cfg.get("top_k") is not None: parts.append(f"top_k={cfg['top_k']}")
        if cfg.get("min_p") is not None: parts.append(f"min_p={cfg['min_p']}")
        thinking = cfg.get("thinking")
        if thinking is not None:
            parts.append(f"think={'on' if thinking else 'off'}")
        if cfg.get("thinking_budget") is not None:
            parts.append(f"budget={cfg['thinking_budget']}")
        if cfg.get("reasoning_format") is not None:
            parts.append(f"rfmt={cfg['reasoning_format']}")
        if cfg.get("reasoning_effort") is not None:
            parts.append(f"reff={cfg['reasoning_effort']}")
        if cfg.get("reasoning_preserve") is not None:
            parts.append(f"preserve={'on' if cfg['reasoning_preserve'] else 'off'}")
        if (cfg.get("jinja") or cfg.get("auto_template") or thinking is not None
                or cfg.get("reasoning_effort") is not None):
            parts.append("jinja")
        if cfg.get("auto_template"):
            tpl = find_template_for_model(models[sel])
            parts.append(f"tpl={tpl.stem}" if tpl else "tpl=embedded")
        else:
            tpl = find_template_override(models[sel])
            if tpl is not None:
                parts.append(f"tpl={tpl.stem}*")
        if cfg.get("draft_mtp"):
            parts.append("mtp")
        if cfg.get("repeat_penalty") is not None:
            parts.append(f"rep={cfg['repeat_penalty']}")
        fix = find_model_fix(models[sel])
        if fix:
            parts.append(f"FIX:{fix['name']}")
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

def settings_menu(stdscr, cfg, model=None):
    # Build visual model options: None (auto), "none" (disabled), then all found mmproj files
    all_mmproj = find_all_mmproj()
    visual_options = [None, "none"] + [str(f) for f in all_mmproj]
    visual_labels = {None: "auto (same folder)", "none": "disabled"}
    for f in all_mmproj:
        visual_labels[str(f)] = f"{f.name}  ({f.parent})"

    # Narrow reasoning_effort choices to what this model's embedded template
    # actually reads, and build a hint line describing its thinking switches.
    effort_options = REASONING_EFFORT_OPTIONS
    tpl_hint = ""
    if model is not None:
        tinfo = template_reasoning_info(model)
        if not tinfo["has_template"]:
            tpl_hint = "no chat template embedded in GGUF"
        elif tinfo["reasoning_effort"]:
            vals = tinfo["effort_values"]
            if vals:
                effort_options = [None] + vals
                tpl_hint = f"template thinking switch: reasoning_effort ({'/'.join(vals)})"
            else:
                tpl_hint = "template thinking switch: reasoning_effort (values not detected)"
        elif tinfo["enable_thinking"]:
            tpl_hint = "template thinking switch: enable_thinking (use Thinking on/off, leave effort default)"
        else:
            tpl_hint = "template has no thinking switch (always-on or non-thinking model)"

    fields = [
        ("context",      "Context length",     CONTEXT_OPTIONS),
        ("threads",      "Threads",            THREAD_OPTIONS),
        ("port",         "Port",               None),
        ("flash_attn",   "Flash attention",    [True, False]),
        ("cache_type_k", "Cache K (-ctk)",     CACHE_OPTIONS),
        ("cache_type_v", "Cache V (-ctv)",     CACHE_OPTIONS),
        ("verbosity",    "Verbosity",          [None, 0, 1, 2, 3, 4, 5]),
        ("batch",        "Batch size (-b)",    BATCH_OPTIONS),
        ("ubatch",       "Micro-batch (-ub)",  BATCH_OPTIONS),
        ("no_mmap",        "No mmap (load to RAM)",   [False, True]),
        ("mlock",          "mlock (pin in RAM)",      [False, True]),
        ("temp",           "Temperature (--temp)",    TEMP_OPTIONS),
        ("top_p",          "Top-P (--top-p)",         TOP_P_OPTIONS),
        ("top_k",          "Top-K (--top-k)",         TOP_K_OPTIONS),
        ("min_p",          "Min-P (--min-p)",         MIN_P_OPTIONS),
        ("thinking",         "Thinking (on/off)",       THINKING_OPTIONS),
        ("thinking_budget",  "Thinking budget (tokens)", THINKING_BUDGET_OPTIONS),
        ("reasoning_format", "Reasoning format",        REASONING_FORMAT_OPTIONS),
        ("reasoning_effort", "Reasoning effort (kwarg)", effort_options),
        ("reasoning_preserve", "Preserve reasoning history", REASONING_PRESERVE_OPTIONS),
        ("jinja",            "Jinja templates (--jinja)", [False, True]),
        ("auto_template",    "Auto-match chat template",  [False, True]),
        ("draft_mtp",      "Draft MTP (--draft-mtp)",  [False, True]),
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
            if key in ("thinking", "reasoning_preserve"):
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

        if tpl_hint and 3 + len(fields) < h:
            stdscr.attron(curses.color_pair(6))
            stdscr.addstr(3 + len(fields), 0, f"  {tpl_hint}"[:w-1])
            stdscr.attroff(curses.color_pair(6))

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
                if fkey == "port":
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
            settings_menu(stdscr, cfg, models[sel])
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

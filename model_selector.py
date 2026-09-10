#!/usr/bin/env python3
r"""
llama.cpp model selector - keyboard-driven TUI
Scans the configured model directories recursively for GGUF files, detects
mmproj (vision), and launches llama-server with appropriate settings.
Per-model settings are saved to model_settings.json next to this script.
"""

import sys
import os
import re
import json
import subprocess
import curses
import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

LLAMA_SERVER = r"C:\tools\llamacpp\llama-server.exe"
MODEL_DIRS   = [r"C:\llm", r"E:\llm", r"K:\models", r"M:\models"]
TEMPLATES_DIR = Path(r"C:\tools\llamacpp\templates")
SETTINGS_FILE = Path(__file__).parent / "model_settings.json"

# Optional MCP integrations installed under mcp/. HTTP endpoints use the
# local stdio bridge; remote services must be running separately.
MCP_DIR = Path(__file__).parent / "mcp"

# Web-search MCP server (SearXNG-backed), with its own virtualenv.
MCP_SEARCH_PYTHON  = MCP_DIR / "search" / ".venv" / "Scripts" / "python.exe"
MCP_SEARCH_SERVER  = MCP_DIR / "search" / "server.py"

# Godot 4.x engine control MCP server (node running the prebuilt bundle).
# GODOT_PATH points the server at the Godot console build shipped alongside it.
MCP_GODOT_ENTRY    = MCP_DIR / "godot" / "server-new" / "build" / "index.js"
MCP_GODOT_EXE      = MCP_DIR / "godot" / "Godot_v4.7.2-stable_win64_console.exe"

MCP_HTTP_BRIDGE    = MCP_DIR / "http_bridge" / "mcp_http_bridge.py"
PLAYWRIGHT_MCP_URL = "http://100.92.156.106:8931/mcp"

# Playwright browser-automation MCP server, reached over streamable-HTTP.
# llama-server only speaks MCP over stdio, so the http_bridge relays between
# stdio (server-side) and the remote HTTP endpoint.

# Blender must be open with its MCP add-on server running.
MCP_BLENDER_PYTHON = MCP_DIR / "blender" / "venv" / "Scripts" / "python.exe"

# UnrealEditor hosts this endpoint; the local bridge connects over HTTP.
UNREAL_MCP_URL     = "http://127.0.0.1:8000/mcp"

MCP_SERVERS = {
    "search": {
        "label": "Search (SearXNG)",
        "note":  "web_search via local search MCP",
        "requires": MCP_SEARCH_PYTHON,
        "definition": {
            "command": str(MCP_SEARCH_PYTHON),
            "args": [str(MCP_SEARCH_SERVER)],
        },
    },
    "godot": {
        "label": "Godot (game engine)",
        "note":  "Godot scene/script tools",
        "requires": MCP_GODOT_ENTRY,
        "definition": {
            "type": "stdio",
            "command": "node",
            "args": [str(MCP_GODOT_ENTRY)],
            "env": {"GODOT_PATH": str(MCP_GODOT_EXE)},
        },
    },
    "playwright": {
        "label": "Playwright (browser)",
        "note":  "browser automation tools",
        "endpoint": PLAYWRIGHT_MCP_URL,
        "requires": MCP_HTTP_BRIDGE,
        "definition": {
            "command": sys.executable,
            "args": [str(MCP_HTTP_BRIDGE), PLAYWRIGHT_MCP_URL],
        },
    },
    "blender": {
        "label": "Blender (3D)",
        "note":  "needs Blender open on :9876",
        "requires": MCP_BLENDER_PYTHON,
        "definition": {
            "command": str(MCP_BLENDER_PYTHON),
            "args": ["-m", "blmcp"],
        },
    },
    "unreal581": {
        "label": "Unreal Engine 5.8",
        "note":  "needs UE 5.8 editor open, MCP server on :8000",
        "endpoint": UNREAL_MCP_URL,
        "requires": MCP_HTTP_BRIDGE,
        "definition": {
            "command": sys.executable,
            "args": [str(MCP_HTTP_BRIDGE), UNREAL_MCP_URL],
        },
    },
}

DEFAULTS = {
    "threads":      16,
    "context":      32768,
    "host":         os.getenv("LLAMA_HOST", "127.0.0.1"),
    "port":         int(os.getenv("LLAMA_PORT", "8080")),
    "flash_attn":   True,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    # Maximum server prompt cache size in MiB; None omits --cache-ram.
    "cache_ram":    16384,
    "verbosity":    3,
    # Share a KV buffer across sequences; False explicitly disables sharing.
    "kv_unified":   True,
    # Comma-separated tensor-regex=device rules. Manual placement can affect
    # automatic fitting; check memory use when overriding it. None omits -ot.
    "override_tensor": None,
    # Concurrent request slots; None leaves the server default.
    "parallel":     1,
    "batch":        512,
    "ubatch":       None,
    # --load-mode; None leaves the server default.
    "load_mode":       "mlock",
    "temp":            1.0,
    "top_p":           0.95,
    "top_k":           20,
    "min_p":           None,
    # Server reasoning default; individual requests may override it.
    "thinking":         True,
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
    # Preserve earlier reasoning where supported by the template.
    # None leaves the server default; False explicitly disables preservation.
    "reasoning_preserve": False,
    "repeat_penalty":   1.05,
    # --presence-penalty: flat penalty on any token already present in the
    # context (server default 0.0 = off). None = flag omitted.
    "presence_penalty": 1.5,
    "jinja":            True,
    "auto_template":    True,    # Auto-match a curated .jinja template by model name
    # MCP servers: names from MCP_SERVERS enabled for this model (e.g.
    # ["search", "playwright"]). Enabled servers are passed to llama-server
    # via --mcp-servers-json and force --jinja (MCP tool calls need the jinja
    # code path). Entries whose files are missing are skipped at launch.
    "mcp_enabled":      [],
    # Multi-Token Prediction (--spec-type draft-mtp); requires model support.
    "draft_mtp":        False,
    # Tokens drafted per step (--spec-draft-n-max). None = server default (3).
    # Lower it when acceptance is poor (less wasted work per rejection), raise
    # it when acceptance is very high.
    "draft_n_max":      None,
    # Optional external draft GGUF (--spec-draft-model).
    "draft_model":      None,
    # KV cache types for the DRAFT context only (--spec-draft-type-k /
    # --spec-draft-type-v, alias -ctkd/-ctvd). Same allowed values as the main
    # -ctk/-ctv. None = flag omitted, so the draft context uses llama.cpp's own
    # f16 default rather than inheriting the main cache_type_*. Only emitted
    # when draft_mtp is on, since there is no draft context otherwise.
    "draft_cache_type_k": None,
    "draft_cache_type_v": None,
    "visual_model":     "none",     # None=auto (same folder), "none"=disabled, or path to mmproj
}

CONTEXT_OPTIONS  = [4096, 8192, 16384, 32768, 49152, 65536, 72000, 80000, 90000, 131072, 150000, 196608, 200000, 262144]
THREAD_OPTIONS   = [4, 8, 12, 16, 20, 24, 32]
# Conservative cache presets. Kernel support and performance depend on the
# installed server build and hardware; None leaves the server default.
CACHE_OPTIONS    = [None, "f16", "q8_0", "q4_0"]
BATCH_OPTIONS    = [None, 256, 512, 1024, 2048, 4096]
CACHE_RAM_OPTIONS = [None, 0, 4096, 8192, 16384, 32768, 49152, 65536, 131072]
KV_UNIFIED_OPTIONS = [True, False]
PARALLEL_OPTIONS   = [None, 1, 2, 3, 4, 6, 8, 16]
# Tensor placement presets; arbitrary rules can also be entered as text.
OVERRIDE_TENSOR_OPTIONS = [
    None,
    r"per_layer_token_embd\.weight=CPU",   # Qwen3.8-Flash-Next / qwen4exp
    r"token_embd\.weight=CPU",             # generic: plain embedding table
]
LOAD_MODE_OPTIONS = [None, "none", "mmap", "mlock", "mmap+mlock", "dio"]
TEMP_OPTIONS     = [None, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.2, 1.5]
TOP_P_OPTIONS    = [None, 0.1, 0.5, 0.8, 0.9, 0.95, 1.0]
TOP_K_OPTIONS    = [None, 0, 10, 20, 40, 80, 100]
MIN_P_OPTIONS    = [None, 0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
THINKING_OPTIONS         = [None, True, False]    # None=default, True=on, False=off
THINKING_BUDGET_OPTIONS  = [None, 0, 256, 1024, 4096, 8192, 16384, 32768]
REASONING_FORMAT_OPTIONS = [None, "deepseek", "deepseek-legacy", "none"]
# deepseek        → extracts thinking into reasoning_content (Open WebUI shows collapsible dropdown)
# deepseek-legacy → keeps <think> tags in content but also populates reasoning_content
# none            → no special formatting; raw output stays in content (think tags included)
# Union of values seen across templates; the settings menu narrows this to the
# values the selected model's embedded template actually accepts.
REASONING_EFFORT_OPTIONS = [None, "no_think", "low", "medium", "high", "xhigh", "max"]
REASONING_PRESERVE_OPTIONS = [None, True, False]  # None=template default
DRAFT_N_MAX_OPTIONS      = [None, 1, 2, 3, 4, 5, 6, 8]  # None = server default (3)
REPEAT_PENALTY_OPTIONS   = [None, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5]
PRESENCE_PENALTY_OPTIONS = [None, 0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]

# Fields that support direct text entry for precision
# (draft_model and override_tensor are free-form text - a file path and a
#  tensor-placement regex - not numbers, so they skip the numeric parse below)
EDITABLE_FIELDS = {"context", "temp", "top_p", "top_k", "min_p",
                   "repeat_penalty", "presence_penalty", "draft_model",
                   "cache_ram", "parallel", "override_tensor"}
# Editable fields parsed as whole numbers rather than floats
INT_EDITABLE_FIELDS = {"context", "top_k", "cache_ram", "parallel"}


# Optional per-model overrides and soft defaults, matched against the path.
MODEL_FIXES = []


# Two kinds of per-model value live in MODEL_FIXES:
#   "overrides"     - correctness. Applied in build_command AFTER the saved
#                     settings, so they always win and cannot be edited away.
#   "soft_defaults" - a starting point. Applied in cfg_for_model BETWEEN the
#                     global DEFAULTS and the saved settings, so the settings
#                     menu shows them, and anything the user saves wins.
# Put a value in "soft_defaults" when it is a sizing/tuning choice rather than
# a "this model is broken without it" fact.


def find_model_fix(model: Path):
    p = str(model).lower()
    for fix in MODEL_FIXES:
        if fix["match"] in p:
            return fix
    return None


# ── Persistence ───────────────────────────────────────────────────────────────

def load_saved_settings() -> dict:
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict) or any(not isinstance(v, dict) for v in data.values()):
        raise ValueError(f"Invalid settings structure in {SETTINGS_FILE}")
    return _migrate_settings(data)


def _migrate_settings(data: dict) -> dict:
    """Retired-key fixups in place:
    gemma4_template_fix -> auto_template,
    no_mmap/mlock -> load_mode (old defaults no_mmap=True, mlock=True
    equal the new "mlock" mode).

    kv_unified used to be stripped here, back when the launcher hard-coded
    --kv-unified on. It is a real setting again (default True, i.e. the same
    behaviour), so saved values are now kept."""
    for key, val in data.items():
        if key == "__meta__" or not isinstance(val, dict):
            continue
        if "gemma4_template_fix" in val:
            val["auto_template"] = val.pop("gemma4_template_fix")
        if "live_search" in val:
            val["mcp_enabled"] = ["search"] if val.pop("live_search") else []
        if "no_mmap" in val or "mlock" in val:
            no_mmap = val.pop("no_mmap", True)
            mlock   = val.pop("mlock", True)
            # Entries matching the OLD defaults (no_mmap=True, mlock=True) are
            # dropped so the model follows the new default instead of being
            # pinned to the old behavior.
            if (no_mmap, mlock) != (True, True):
                if no_mmap and mlock:
                    val["load_mode"] = "mlock"
                elif not no_mmap and mlock:
                    val["load_mode"] = "mmap+mlock"
                elif no_mmap and not mlock:
                    val["load_mode"] = "none"
                else:
                    val["load_mode"] = "mmap"
    return data


def save_settings(all_saved: dict):
    # Replace only after the complete JSON has been written successfully.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=SETTINGS_FILE.parent,
                                         delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(json.dumps(all_saved, indent=2))
        os.replace(tmp_path, SETTINGS_FILE)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def cfg_for_model(model: Path, all_saved: dict) -> dict:
    saved = all_saved.get(str(model), {})
    cfg = dict(DEFAULTS)
    fix = find_model_fix(model)
    if fix:
        cfg.update(fix.get("soft_defaults", {}))
    cfg.update(saved)
    return cfg


def persist_cfg(model: Path, cfg: dict, all_saved: dict, is_launch: bool = False):
    # Compare against this model's defaults so explicit tuning survives reload.
    baseline = cfg_for_model(model, {})
    skip = {"host"}
    delta = {k: v for k, v in cfg.items()
             if k not in skip and v != baseline.get(k)}
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

# Vision towers usually announce themselves (mmproj-*, *-projector-*), but some
# repos ship one as a plain "-vision-<dtype>" sibling of the LM, e.g.
# Qwen3.8-27B-Uncensored-vision-f16.gguf next to ...-YMQ-XL.gguf. Match that
# form only at the tail of the name and only for a small file, so a real VL
# model (Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf) is not mistaken for one.
VISION_TOWER_RE = re.compile(
    r"[-_.](?:vision|visual|vit|clip)"
    r"(?:[-_.](?:f16|f32|bf16|fp16|fp32|q\d\w*))?\.gguf$", re.IGNORECASE)
VISION_TOWER_MAX_BYTES = 3 * 1024 ** 3


def is_mmproj(path: Path) -> bool:
    """True if this GGUF is a vision tower rather than a launchable model."""
    n = path.name.lower()
    if "mmproj" in n or "projector" in n:
        return True
    if not VISION_TOWER_RE.search(n):
        return False
    try:
        return path.stat().st_size <= VISION_TOWER_MAX_BYTES
    except OSError:
        return False


def find_models():
    models = []
    for base in MODEL_DIRS:
        p = Path(base)
        if not p.exists():
            continue
        for f in sorted(p.rglob("*.gguf")):
            name = f.name.lower()
            if is_mmproj(f) or is_draft(f):
                continue
            # Split GGUFs: only the first shard is launchable; llama-server
            # picks up the rest of the -NNNNN-of-NNNNN set automatically.
            shard = re.search(r"-(\d{5})-of-\d{5}\.gguf$", name)
            if shard and shard.group(1) != "00001":
                continue
            models.append(f)
    return models


def find_mmproj(model_path: Path):
    for f in sorted(model_path.parent.glob("*.gguf")):
        if f != model_path and is_mmproj(f):
            return f
    return None


def find_all_mmproj():
    """Recursively scan all model dirs for mmproj/projector GGUF files."""
    found = []
    for base in MODEL_DIRS:
        p = Path(base)
        if not p.exists():
            continue
        for f in sorted(p.rglob("*.gguf")):
            if is_mmproj(f):
                found.append(f)
    return found


# A separate drafter module, not a full MTP model: embedded-MTP quants draft
# from their own nextn layers and never need to be picked here.
DRAFT_NAME_RE = re.compile(
    r"draft|nextn|eagle|medusa|speculat|mtp[-_]?(module|head)|(?:^|[-_.])mtp(?=[-_.]|$)",
    re.IGNORECASE)


def is_draft(path: Path) -> bool:
    """Recognize separate draft modules, including suffix-named MTP shards."""
    return path.name.lower().startswith("mtp") or bool(DRAFT_NAME_RE.search(path.name))


def find_all_draft():
    """Recursively scan all model dirs for separate draft/MTP module GGUFs."""
    found = []
    for base in MODEL_DIRS:
        p = Path(base)
        if not p.exists():
            continue
        for f in sorted(p.rglob("*.gguf")):
            n = f.name.lower()
            if is_mmproj(f):
                continue
            shard = re.search(r"-(\d{5})-of-\d{5}\.gguf$", n)
            if shard and shard.group(1) != "00001":
                continue
            if is_draft(f):
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
    # Explicit local template mappings; more specific names come first.
    ("laguna", "Laguna-S-2.1.jinja"),
    ("qwen3.8-flash-next", "Qwen3.8-Flash-Next.jinja"),
    ("qwen3.8", "Qwen3.8.jinja"),
]

# Models whose GGUF-embedded template is correct but whose filename fools the
# token scorer into handing them someone else's. Matched case-insensitively
# against the model filename; a hit disables auto-matching for that model, so
# the server uses the embedded template. Substrings, same as the list above.
TEMPLATE_BLOCKLIST = []


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
    name = model.name.lower()
    if any(needle in name for needle in TEMPLATE_BLOCKLIST):
        return None
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
        # Penalise tokens the model name doesn't carry, including the variant
        # suffixes (interleaved, fixed) already dropped from core, so a vision
        # "-interleaved" template loses to the plain one for a text-only model
        # instead of ties being decided by glob() order. Generic role suffixes
        # stay exempt from the penalty.
        role_suffixes = {"it", "instruct", "tool", "use", "default", "rag",
                         "bf16", "ai", "org", "hf", "team", "research", "forai"}
        extra = len((tpl_tokens - role_suffixes) - model_tokens)
        score = overlap + (5 if stem_hit else 0) - 0.5 * extra
        # Require a real family-name match, not just a shared size/quant token
        # like "8b". The model must contain at least one alphabetic core token
        # (e.g. "gemma", "deepseek", "qwen"); org prefixes were already dropped.
        alpha_core = {t for t in core if t[0].isalpha() and len(t) > 2
                      and not re.fullmatch(r"(?:i?q\d+|f\d+|fp\d+)", t)}
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
            # Lists, tuples and sets all appear in the wild:
            #   {%- elif reasoning_effort not in ['high', 'low', 'no_think'] %}
            #   {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
            # Picking the wrong values here is not cosmetic: an effort value the
            # template rejects makes it raise, and every request 500s.
            m = re.search(r"reasoning_effort\s+(?:not\s+)?in\s*"
                          r"[\[({]([^\])}]*)[\])}]", tpl)
            if m:
                vals = re.findall(r"""['"]([^'"]+)['"]""", m.group(1))
                if vals:
                    info["effort_values"] = vals
        _template_info_cache[key] = info
    return _template_info_cache[key]


# ── Command builder ───────────────────────────────────────────────────────────

def collect_mcp_servers(cfg: dict) -> dict:
    """Definitions of the enabled MCP servers, keyed for --mcp-servers-json.

    Registry entries either point at a Cursor-format config file (its
    mcpServers definitions are merged in) or carry an inline definition.
    Entries whose config_file/requires path is missing are skipped, so a
    launch never breaks on a dead toggle.
    """
    defs = {}
    for name in cfg.get("mcp_enabled") or []:
        entry = MCP_SERVERS.get(name)
        if not entry:
            continue
        cfg_file = entry.get("config_file")
        if cfg_file is not None:
            try:
                data = json.loads(Path(cfg_file).read_text())
            except (OSError, ValueError):
                continue
            defs.update(data.get("mcpServers", {}))
        elif entry.get("definition"):
            req = entry.get("requires")
            if req is not None and not Path(req).exists():
                continue
            defs[name] = entry["definition"]
    return defs


def build_env(model: Path) -> dict | None:
    """Process environment for the server, or None to inherit unchanged."""
    fix = find_model_fix(model)
    extra = fix.get("env") if fix else None
    if not extra:
        return None
    return {**os.environ, **extra}


def build_command(model: Path, cfg: dict) -> list:
    server = LLAMA_SERVER
    fix_args = []
    fix = find_model_fix(model)
    if fix:
        cfg = {**cfg, **fix.get("overrides", {})}
        # "args" always apply; "fallback_args" only when the patched build is
        # missing and we are falling back to the stock server.
        fix_args = list(fix.get("args", []))
        patched = fix.get("server")
        if patched and Path(patched).exists():
            server = patched
        else:
            fix_args += fix.get("fallback_args", [])
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
    if cfg.get("context") is not None:
        cmd += ["-c", str(cfg["context"])]
    cmd += ["--threads", str(cfg["threads"])]
    cmd += ["--flash-attn", "on" if cfg["flash_attn"] else "off"]
    cmd += ["--host", cfg["host"], "--port", str(cfg["port"])]
    if cfg.get("cache_type_k"):
        cmd += ["-ctk", cfg["cache_type_k"]]
    if cfg.get("cache_type_v"):
        cmd += ["-ctv", cfg["cache_type_v"]]
    if cfg.get("cache_ram") is not None:
        cmd += ["--cache-ram", str(cfg["cache_ram"])]
    if cfg.get("kv_unified") is not None:
        cmd += ["--kv-unified" if cfg["kv_unified"] else "--no-kv-unified"]
    if cfg.get("verbosity") is not None:
        cmd += ["--verbosity", str(cfg["verbosity"])]
    if cfg.get("batch"):
        cmd += ["-b", str(cfg["batch"])]
    if cfg.get("ubatch"):
        cmd += ["-ub", str(cfg["ubatch"])]
    if cfg.get("load_mode"):
        cmd += ["--load-mode", cfg["load_mode"]]
    ot_rules = [r for r in (cfg.get("override_tensor") or "").split(",") if r.strip()]
    if ot_rules:
        cmd += ["-ot", ",".join(ot_rules)]
    if cfg.get("temp") is not None:
        cmd += ["--temp", str(cfg["temp"])]
    if cfg.get("top_p") is not None:
        cmd += ["--top-p", str(cfg["top_p"])]
    if cfg.get("top_k") is not None:
        cmd += ["--top-k", str(cfg["top_k"])]
    if cfg.get("min_p") is not None:
        cmd += ["--min-p", str(cfg["min_p"])]
    if cfg.get("thinking") is True:
        cmd += ["--reasoning", "on"]
    elif cfg.get("thinking") is False:
        cmd += ["--reasoning", "off"]
    if cfg.get("thinking_budget") is not None:
        cmd += ["--reasoning-budget", str(cfg["thinking_budget"])]
    if cfg.get("reasoning_format") is not None:
        cmd += ["--reasoning-format", cfg["reasoning_format"]]
    # Pass template-specific effort through Jinja kwargs.
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
    # off: these are explicit local mappings.
    matched_template = None
    if cfg.get("auto_template"):
        matched_template = find_template_for_model(model)
    else:
        matched_template = find_template_override(model)
    # A custom --chat-template-file is only honored on the Jinja code path
    # (jinja is on by default in current builds, but force it to be explicit).
    # --chat-template-kwargs likewise only applies under Jinja.
    mcp_defs = collect_mcp_servers(cfg)
    use_jinja = (cfg.get("jinja") or cfg.get("auto_template")
                 or cfg.get("reasoning_effort") is not None
                 or bool(mcp_defs)
                 or matched_template is not None)
    if use_jinja:
        cmd += ["--jinja"]
    else:
        cmd += ["--no-jinja"]
    if matched_template is not None:
        cmd += ["--chat-template-file", str(matched_template)]
    if mcp_defs:
        cmd += ["--mcp-servers-json", json.dumps({"mcpServers": mcp_defs})]
        # MCP/tools make the server lock CORS to localhost and warn about it.
        # The special value "localhost" IS that default, so stating it
        # explicitly keeps the same security posture and silences the warning.
        cmd += ["--cors-origins", "localhost"]
    if cfg.get("draft_mtp"):
        cmd += ["--spec-type", "draft-mtp"]
        if cfg.get("draft_n_max") is not None:
            cmd += ["--spec-draft-n-max", str(cfg["draft_n_max"])]
        # Only for models whose MTP heads live in a separate GGUF; embedded-MTP
        # models draft from their own nextn layers with no extra file.
        draft = cfg.get("draft_model")
        if draft and Path(draft).exists():
            cmd += ["--spec-draft-model", str(draft)]
        if cfg.get("draft_cache_type_k"):
            cmd += ["-ctkd", cfg["draft_cache_type_k"]]
        if cfg.get("draft_cache_type_v"):
            cmd += ["-ctvd", cfg["draft_cache_type_v"]]
    if cfg.get("repeat_penalty") is not None:
        cmd += ["--repeat-penalty", str(cfg["repeat_penalty"])]
    if cfg.get("presence_penalty") is not None:
        cmd += ["--presence-penalty", str(cfg["presence_penalty"])]
    if fix_args:
        cmd += fix_args
    if cfg.get("parallel") is not None:
        cmd += ["--parallel", str(cfg["parallel"])]
    return cmd


# ── Helpers ───────────────────────────────────────────────────────────────────

def draw_text(stdscr, y, x, text):
    """Clip writes to the current terminal, including during a resize."""
    h, w = stdscr.getmaxyx()
    if not (0 <= y < h and 0 <= x < w - 1):
        return
    try:
        stdscr.addstr(y, x, text[:w - x - 1])
    except curses.error:
        # A resize or a wide glyph can invalidate the measured bounds.
        pass


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
        "arrows=navigate  enter=launch  s=settings  c=copy-settings  "
        "/=search  o=sort  d=del-settings  r=rescan  q=quit"
    )
    stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
    draw_text(stdscr, 0, 0, header[:w-1].ljust(w-1))
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
            f"ctx={cfg['context'] if cfg.get('context') is not None else 'default'}",
            f"threads={cfg['threads']}",
            f"port={cfg['port']}",
            f"fa={'on' if cfg['flash_attn'] else 'off'}",
            f"ctk={ctk}",
            f"ctv={ctv}",
        ]
        if verb is not None:  parts.append(f"verb={verb}")
        if bat  is not None:  parts.append(f"b={bat}")
        if ubat is not None:  parts.append(f"ub={ubat}")
        if cfg.get("load_mode"): parts.append(f"load={cfg['load_mode']}")
        if cfg.get("cache_ram") is not None: parts.append(f"cram={cfg['cache_ram']}")
        if cfg.get("override_tensor"): parts.append(f"ot={cfg['override_tensor']}")
        parts.append(f"kvu={'on' if cfg.get('kv_unified') else 'off'}")
        if cfg.get("parallel") is not None: parts.append(f"par={cfg['parallel']}")
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
        if cfg.get("mcp_enabled"):
            parts.append("mcp=" + ",".join(cfg["mcp_enabled"]))
        if cfg.get("draft_mtp"):
            dm = cfg.get("draft_model")
            nmax = cfg.get("draft_n_max")
            tag = f"mtp:{Path(dm).name}" if dm else "mtp"
            parts.append(f"{tag}(n={nmax})" if nmax is not None else tag)
            dctk = cfg.get("draft_cache_type_k")
            dctv = cfg.get("draft_cache_type_v")
            if dctk: parts.append(f"ctkd={dctk}")
            if dctv: parts.append(f"ctvd={dctv}")
        if cfg.get("repeat_penalty") is not None:
            parts.append(f"rep={cfg['repeat_penalty']}")
        if cfg.get("presence_penalty") is not None:
            parts.append(f"pres={cfg['presence_penalty']}")
        fix = find_model_fix(models[sel])
        if fix:
            parts.append(f"FIX:{fix['name']}")
        settings_str = " " + "  ".join(parts) + vision_tag + saved_mark
    else:
        settings_str = " (no models)"

    stdscr.attron(curses.color_pair(3))
    draw_text(stdscr, 1, 0, settings_str[:w-1].ljust(w-1))
    stdscr.attroff(curses.color_pair(3))

    # Filter bar
    sort_label = "recent" if sort_mode == "recent" else "name"
    filter_bar = f" Filter: {filter_str}_  [sort:{sort_label}]  {len(models)} models"
    stdscr.attron(curses.color_pair(6))
    draw_text(stdscr, 2, 0, filter_bar[:w-1].ljust(w-1))
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
            draw_text(stdscr, y, 0, line[:w-1].ljust(w-1))
            stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
        else:
            draw_text(stdscr, y, 0, line[:w-1])

    # Status / command preview
    if models and sel < len(models):
        preview = " ".join(build_command(models[sel], cfg))
        stdscr.attron(curses.color_pair(4))
        draw_text(stdscr, h-1, 0, (" CMD: " + preview)[:w-1].ljust(w-1))
        stdscr.attroff(curses.color_pair(4))

    if status:
        stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
        draw_text(stdscr, h-1, 0, status[:w-1].ljust(w-1))
        stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)

    stdscr.refresh()


# ── Inline value editor ───────────────────────────────────────────────────────

def inline_edit(stdscr, label, current_val):
    """Show a bottom-bar text input. Returns the entered string, None for
    blank/"none" (meaning "use default"), or current_val if Esc is pressed
    (meaning "keep the current value")."""
    h, w = stdscr.getmaxyx()
    curses.curs_set(1)
    buf = "" if current_val is None else str(current_val)
    while True:
        prompt = f" Enter {label} (blank=default, Esc=cancel): {buf}_"
        stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        draw_text(stdscr, h - 1, 0, prompt[:w - 1].ljust(w - 1))
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


# ── MCP menu ──────────────────────────────────────────────────────────────────

def mcp_menu(stdscr, cfg):
    """Per-server MCP enable/disable, reached from the settings menu.

    Toggling always assigns a FRESH list to cfg["mcp_enabled"], rebuilt in
    registry order - never mutate in place: cfg_for_model shallow-copies
    DEFAULTS, so an in-place append/remove would edit the list inside
    DEFAULTS itself, leak into every model, and defeat persist_cfg's
    v != DEFAULTS.get(k) delta check.
    """
    names = list(MCP_SERVERS.keys())
    sel = 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        draw_text(stdscr, 0, 0,
            " MCP servers  |  up/down=server  space/enter=toggle  q=back"
            .ljust(w-1))
        stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

        enabled = cfg.get("mcp_enabled") or []
        row = 2
        for i, name in enumerate(names):
            entry = MCP_SERVERS[name]
            path = entry.get("config_file") or entry.get("requires")
            missing = path if (path is not None and not Path(path).exists()) else None
            mark = "[x]" if name in enabled else "[ ]"
            line = f"  {mark} {name:<12} {entry['label']:<24} {entry.get('note', '')}"
            if missing:
                line += f"  (missing: {missing})"
            if i == sel:
                stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                draw_text(stdscr, row, 0, line[:w-1].ljust(w-1))
                stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            else:
                draw_text(stdscr, row, 0, line[:w-1])
            row += 1
            if entry.get("endpoint"):
                stdscr.attron(curses.color_pair(6))
                draw_text(stdscr, row, 0, f"{'':20}{entry['endpoint']}"[:w-1])
                stdscr.attroff(curses.color_pair(6))
                row += 1

        stdscr.refresh()
        key = stdscr.getch()

        if key in (ord('q'), 27):
            break
        elif key == curses.KEY_UP:
            sel = (sel - 1) % len(names)
        elif key == curses.KEY_DOWN:
            sel = (sel + 1) % len(names)
        elif key in (ord(' '), 10, 13):
            name = names[sel]
            entry = MCP_SERVERS[name]
            path = entry.get("config_file") or entry.get("requires")
            if path is not None and not Path(path).exists():
                continue  # missing on disk: not toggleable
            current = set(cfg.get("mcp_enabled") or [])
            current.symmetric_difference_update({name})
            cfg["mcp_enabled"] = [n for n in names if n in current]


# ── Copy settings from another model ──────────────────────────────────────────
# Settings that name a file inside the SOURCE model's own folder. Copying them
# onto another model would point it at the wrong file, so they are left alone
# unless the user asks for them explicitly (p toggle).
PATH_BOUND_KEYS = {"visual_model", "draft_model"}


def saved_models(all_saved: dict, exclude: Path = None):
    """Models that have saved settings, most recently launched first."""
    keys = [k for k in all_saved if k != "__meta__"]
    if exclude is not None:
        keys = [k for k in keys if k != str(exclude)]
    keys.sort(key=lambda k: (-all_saved.get("__meta__", {})
                             .get(k, {}).get("last_launch", 0), k.lower()))
    return [Path(k) for k in keys]


def fmt_value(key, val):
    if key in ("thinking", "reasoning_preserve", "flash_attn",
               "jinja", "auto_template", "draft_mtp"):
        return {None: "default", True: "on", False: "off"}.get(val, str(val))
    if key == "mcp_enabled":
        return ", ".join(val) if val else "none"
    if key in PATH_BOUND_KEYS:
        if val is None:
            return "auto" if key == "visual_model" else "none"
        if val == "none":
            return "disabled"
        return Path(val).name
    return str(val) if val is not None else "default"


def copy_payload(src_cfg: dict, include_paths: bool) -> dict:
    """The settings that a copy would apply (host is env-managed, never copied)."""
    skip = {"host"} if include_paths else {"host"} | PATH_BOUND_KEYS
    return {k: (list(v) if isinstance(v, list) else v)
            for k, v in src_cfg.items() if k not in skip}


def copy_settings_menu(stdscr, cfg, target: Path, all_saved: dict):
    """Pick another model and copy its settings onto the current one.

    Returns (copied, status message). The full resolved config of the source is
    applied (defaults included), so the target ends up matching the source
    rather than merging with whatever it had before.
    """
    sources = saved_models(all_saved, exclude=target)
    if not sources:
        return False, "No other model has saved settings to copy from."

    base_dirs = [Path(d) for d in MODEL_DIRS]
    include_paths = False
    filter_str = ""
    sel = 0

    while True:
        shown = [m for m in sources
                 if not filter_str or filter_str.lower() in str(m).lower()]
        sel = max(0, min(sel, len(shown) - 1))

        stdscr.clear()
        h, w = stdscr.getmaxyx()
        stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        draw_text(stdscr, 0, 0,
            " Copy settings from another model  |  up/down=pick  enter=copy  "
            "type=filter  tab=include file paths  Esc=cancel".ljust(w-1))
        stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

        stdscr.attron(curses.color_pair(3))
        draw_text(stdscr, 1, 0, f" Onto: {short_label(target, base_dirs)}"[:w-1])
        stdscr.attroff(curses.color_pair(3))
        stdscr.attron(curses.color_pair(6))
        draw_text(stdscr, 2, 0,
            (f" Filter: {filter_str}_   {len(shown)} models with saved settings"
             f"   [file paths (mmproj/draft): "
             f"{'copied too' if include_paths else 'kept as-is'}]")[:w-1])
        stdscr.attroff(curses.color_pair(6))

        list_w = max(30, min(70, w // 2 - 2))
        list_h = h - 5
        offset = max(0, sel - list_h + 1) if sel >= list_h else 0

        for i, m in enumerate(shown[offset:offset + list_h]):
            idx = i + offset
            line = f" {short_label(m, base_dirs)}"[:list_w].ljust(list_w)
            if idx == sel:
                stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                draw_text(stdscr, 4 + i, 0, line)
                stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            else:
                draw_text(stdscr, 4 + i, 0, line)

        # Preview: what would change on the target
        if shown:
            src_cfg = cfg_for_model(shown[sel], all_saved)
            payload = copy_payload(src_cfg, include_paths)
            changes = [(k, cfg.get(k), v) for k, v in payload.items()
                       if v != cfg.get(k)]
            px = list_w + 3
            pw = max(10, w - px - 1)
            if not changes:
                stdscr.attron(curses.color_pair(5))
                draw_text(stdscr, 4, px, "Identical — nothing would change."[:pw])
                stdscr.attroff(curses.color_pair(5))
            else:
                stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
                draw_text(stdscr, 3, px, f"{len(changes)} setting(s) would change:"[:pw])
                stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)
                for i, (k, old, new) in enumerate(changes[:h - 6]):
                    txt = f"  {k:<20} {fmt_value(k, old)}  ->  {fmt_value(k, new)}"
                    draw_text(stdscr, 4 + i, px, txt[:pw])

        stdscr.refresh()
        key = stdscr.getch()

        if key == 27:                                   # Esc — cancel
            return False, "Copy cancelled."
        elif key == curses.KEY_UP:
            sel = (sel - 1) % len(shown) if shown else 0
        elif key == curses.KEY_DOWN:
            sel = (sel + 1) % len(shown) if shown else 0
        elif key == curses.KEY_PPAGE:
            sel = max(0, sel - 10)
        elif key == curses.KEY_NPAGE:
            sel = min(len(shown) - 1, sel + 10) if shown else 0
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            filter_str = filter_str[:-1]
            sel = 0
        elif key in (10, 13, curses.KEY_ENTER):
            if not shown:
                continue
            src = shown[sel]
            payload = copy_payload(cfg_for_model(src, all_saved), include_paths)
            changed = sum(1 for k, v in payload.items() if v != cfg.get(k))
            cfg.update(payload)
            return True, (f"Copied {changed} setting(s) from "
                          f"{short_label(src, base_dirs)}.")
        elif key == 9:                                  # Tab — path keys on/off
            include_paths = not include_paths
        elif 32 <= key <= 126:
            filter_str += chr(key)
            sel = 0


# ── Settings menu ─────────────────────────────────────────────────────────────

def settings_menu(stdscr, cfg, model=None, all_saved=None):
    # Build visual model options: None (auto), "none" (disabled), then all found mmproj files
    all_mmproj = find_all_mmproj()
    visual_options = [None, "none"] + [str(f) for f in all_mmproj]
    visual_labels = {None: "auto (same folder)", "none": "disabled"}
    for f in all_mmproj:
        visual_labels[str(f)] = f"{f.name}  ({f.parent})"

    # Draft/MTP module options, same shape as the visual model list: None
    # (no separate drafter) followed by every draft-looking GGUF found.
    all_draft = find_all_draft()
    draft_options = [None] + [str(f) for f in all_draft]
    draft_labels = {None: "none (embedded MTP)"}
    for f in all_draft:
        draft_labels[str(f)] = f"{f.name}  ({f.parent})"

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

    main_fields = [
        ("__copy__",     "Copy from other model",     None),
        ("context",      "Context length",     CONTEXT_OPTIONS),
        ("cache_type_k", "Cache K (-ctk)",     CACHE_OPTIONS),
        ("cache_type_v", "Cache V (-ctv)",     CACHE_OPTIONS),
        ("verbosity",    "Verbosity",          [None, 0, 1, 2, 3, 4, 5]),
        ("temp",           "Temperature (--temp)",    TEMP_OPTIONS),
        ("top_p",          "Top-P (--top-p)",         TOP_P_OPTIONS),
        ("top_k",          "Top-K (--top-k)",         TOP_K_OPTIONS),
        ("min_p",          "Min-P (--min-p)",         MIN_P_OPTIONS),
        ("reasoning_effort", "Reasoning effort (kwarg)", effort_options),
        ("__mcp__",          "MCP servers",               None),
        ("repeat_penalty", "Repeat penalty",           REPEAT_PENALTY_OPTIONS),
        ("presence_penalty", "Presence penalty",       PRESENCE_PENALTY_OPTIONS),
    ]
    # Rarely-touched knobs. Kept out of the main tab so the common settings fit
    # on one screen; same editing keys apply.
    advanced_fields = [
        ("threads",      "Threads",            THREAD_OPTIONS),
        ("port",         "Port",               None),
        ("flash_attn",   "Flash attention",    [True, False]),
        ("cache_ram",    "Cache RAM MB (--cache-ram)", CACHE_RAM_OPTIONS),
        ("override_tensor", "Tensor placement (-ot)", OVERRIDE_TENSOR_OPTIONS),
        ("kv_unified",   "KV unified (--kv-unified)", KV_UNIFIED_OPTIONS),
        ("parallel",     "Parallel slots (--parallel)", PARALLEL_OPTIONS),
        ("batch",        "Batch size (-b)",    BATCH_OPTIONS),
        ("ubatch",       "Micro-batch (-ub)",  BATCH_OPTIONS),
        ("load_mode",      "Load mode (--load-mode)", LOAD_MODE_OPTIONS),
        ("reasoning_preserve", "Preserve reasoning history", REASONING_PRESERVE_OPTIONS),
        ("jinja",            "Jinja templates (--jinja)", [False, True]),
        ("auto_template",    "Auto-match chat template",  [False, True]),
        ("reasoning_format", "Reasoning format",        REASONING_FORMAT_OPTIONS),
        ("thinking_budget",  "Thinking budget (tokens)", THINKING_BUDGET_OPTIONS),
        ("thinking",         "Thinking (on/off)",       THINKING_OPTIONS),
        ("draft_mtp",      "Draft MTP (--spec-type)",  [False, True]),
        ("draft_n_max",    "MTP draft tokens (n-max)", DRAFT_N_MAX_OPTIONS),
        ("draft_model",    "Draft GGUF (-md, optional)", draft_options),
        ("draft_cache_type_k", "Draft cache K (-ctkd)", CACHE_OPTIONS),
        ("draft_cache_type_v", "Draft cache V (-ctvd)", CACHE_OPTIONS),
        ("visual_model",   "Visual model (mmproj)",    visual_options),
    ]
    tabs = [("Main", main_fields), ("Advanced", advanced_fields)]
    tab = 0
    sel = 0
    menu_status = ""

    while True:
        fields = tabs[tab][1]
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        draw_text(stdscr, 0, 0,
            " Settings  |  up/down=field  left/right or +/-=value  enter=edit[*]  "
            "tab=switch tab  q=back".ljust(w-1))
        stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

        # Tab bar
        x = 1
        for i, (name, _) in enumerate(tabs):
            chunk = f" {name} "
            if i == tab:
                stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                draw_text(stdscr, 1, x, chunk[:max(0, w - 1 - x)])
                stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            else:
                stdscr.attron(curses.color_pair(6))
                draw_text(stdscr, 1, x, chunk[:max(0, w - 1 - x)])
                stdscr.attroff(curses.color_pair(6))
            x += len(chunk) + 1

        visible_rows = max(1, h - 5)
        offset = max(0, sel - visible_rows + 1)
        for i in range(offset, min(len(fields), offset + visible_rows)):
            key, label, options = fields[i]
            val = cfg.get(key)
            if key in ("thinking", "reasoning_preserve"):
                display = {None: "default", True: "on", False: "off"}.get(val, str(val))
            elif key == "__mcp__":
                display = ", ".join(cfg.get("mcp_enabled") or []) or "none"
            elif key == "__copy__":
                display = "enter = pick a model to copy its settings from"
            elif key == "draft_model":
                display = draft_labels.get(val, Path(val).name if val else "none")
            elif key == "visual_model":
                display = visual_labels.get(val, Path(val).name if val else "auto (same folder)")
            else:
                display = str(val) if val is not None else "default"
            if key in EDITABLE_FIELDS:
                editable_marker = "[*]"
            elif key in ("__mcp__", "__copy__"):
                editable_marker = "[>]"   # enter opens a submenu
            else:
                editable_marker = "   "
            line = f"  {editable_marker} {label:<26} {display}"
            if i == sel:
                stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                draw_text(stdscr, 3 + i - offset, 0, line[:w-1].ljust(w-1))
                stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            else:
                draw_text(stdscr, 3 + i - offset, 0, line[:w-1])

        if tpl_hint and 4 + len(fields) < h:
            stdscr.attron(curses.color_pair(6))
            draw_text(stdscr, 4 + len(fields), 0, f"  {tpl_hint}"[:w-1])
            stdscr.attroff(curses.color_pair(6))

        if menu_status:
            stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
            draw_text(stdscr, h - 1, 0, f" {menu_status}"[:w-1].ljust(w-1))
            stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)

        stdscr.refresh()
        key = stdscr.getch()

        if key in (ord('q'), ord('s'), 27):
            break
        elif key == 9:              # Tab — next tab
            tab = (tab + 1) % len(tabs)
            sel = 0
        elif key == curses.KEY_BTAB:  # Shift-Tab — previous tab
            tab = (tab - 1) % len(tabs)
            sel = 0
        elif key in (10, 13):
            fkey, flabel, _ = fields[sel]
            if fkey == "__mcp__":
                mcp_menu(stdscr, cfg)
                continue
            if fkey == "__copy__":
                if all_saved is None or model is None:
                    menu_status = "Copy unavailable here."
                else:
                    _, menu_status = copy_settings_menu(stdscr, cfg, model, all_saved)
                continue
            if fkey in EDITABLE_FIELDS:
                raw = inline_edit(stdscr, flabel, cfg.get(fkey))
                if raw is None:
                    cfg[fkey] = None
                elif fkey in ("draft_model", "override_tensor"):
                    cfg[fkey] = raw  # free-form path / regex, no numeric parsing
                else:
                    try:
                        cfg[fkey] = (int(raw) if fkey in INT_EDITABLE_FIELDS
                                     else float(raw))
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
                    cfg[fkey] = min(65535, max(1, cfg[fkey] + direction))


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

    draw_text(stdscr, 0, 0, "Scanning for models...")
    stdscr.refresh()

    all_models = find_models()
    all_saved  = load_saved_settings()

    if not all_models:
        stdscr.clear()
        draw_text(stdscr, 0, 0, "No GGUF models found in " + ", ".join(MODEL_DIRS))
        draw_text(stdscr, 1, 0, "Press any key to exit.")
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

        elif not models and key in (
                curses.KEY_UP, curses.KEY_DOWN, curses.KEY_PPAGE,
                curses.KEY_NPAGE, ord('s'), ord('S')):
            status = "No models selected. Change the filter or rescan."
            continue

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
            settings_menu(stdscr, cfg, models[sel], all_saved)
            persist_cfg(models[sel], cfg, all_saved, is_launch=False)
            status = "Settings saved."

        # ── Copy settings from another model ──
        elif key in (ord('c'), ord('C')):
            if models:
                copied, status = copy_settings_menu(stdscr, cfg, models[sel], all_saved)
                if copied:
                    persist_cfg(models[sel], cfg, all_saved, is_launch=False)
                    status += " Saved."

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
            draw_text(stdscr, 0, 0, "Rescanning...")
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
            env = build_env(model)
            curses.endwin()
            print("Launching:")
            fix = find_model_fix(model)
            for k, v in (fix.get("env") or {}).items() if fix else ():
                print(f"set {k}={v}")
            print(subprocess.list2cmdline(cmd))
            print()
            subprocess.run(cmd, env=env)
            return


if __name__ == "__main__":
    # Request terminal resize via VT escape sequence (works in Windows Terminal)
    sys.stdout.write("\033[8;50;220t")
    sys.stdout.flush()
    curses.wrapper(main)

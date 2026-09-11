"""Hugging Face GGUF catalog, exact download plans, and scoped deletion."""
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess

SHARD = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.I)
QUANT = re.compile(r"(?:^|[-_. /])((?:I?Q\d)[A-Z0-9_]*|BF16|FP16|FP32|F16|F32)(?=[-. /]|$)", re.I)


@dataclass(frozen=True)
class Variant:
    name: str
    files: tuple
    size: int | None
    kind: str
    complete: bool

    @property
    def quant(self):
        matches = list(QUANT.finditer(self.name))
        return matches[-1][1].upper() if matches else "unknown"


def format_bytes(size):
    if size is None:
        return "unknown size"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024


def safe_child(root, name):
    """Reject traversal and Windows device/alternate-stream paths from metadata."""
    parts = name.split("/")
    reserved = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)
    if any(not p or p in (".", "..") or p.endswith((" ", "."))
           or any(c in p for c in '\\:<>"|?*') or any(ord(c) < 32 for c in p) or reserved.match(p)
           for p in parts):
        raise ValueError(f"Unsafe repository path: {name}")
    root = Path(root).resolve()
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError(f"Path leaves model directory: {name}")
    return target


def catalog(siblings, classify):
    groups = {}
    for item in siblings:
        name = item.rfilename
        if not name.lower().endswith(".gguf"):
            continue
        safe_child(Path.cwd(), name)
        match = SHARD.match(name)
        key = (match[1] + ".gguf", match[3]) if match else (name, None)
        groups.setdefault(key, []).append(item)
    result = []
    for (name, count), items in groups.items():
        items.sort(key=lambda f: f.rfilename)
        complete = count is None or {int(SHARD.match(f.rfilename)[2]) for f in items} == set(range(1, int(count) + 1))
        sizes = [f.size for f in items]
        size = sum(sizes) if all(s is not None for s in sizes) else None
        result.append(Variant(name, tuple(f.rfilename for f in items), size,
                              classify(name, size), complete))
    return sorted(result, key=lambda v: (v.kind, v.size is None, v.size or 0, v.name))


def search(query):
    from huggingface_hub import HfApi
    return list(HfApi().list_models(search=query, filter="gguf", sort="downloads", direction=-1, limit=60))


def repo_catalog(repo, classify):
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True, timeout=30)
    return info.sha, catalog(info.siblings, classify)


def download(repo, revision, root, variants):
    executable = shutil.which("hf")
    if not executable:
        raise RuntimeError('hf is missing. Run: python -m pip install -U huggingface_hub')
    if not variants or any(not v.complete for v in variants):
        raise ValueError("Select complete GGUF variants before downloading.")
    destination = safe_child(root, repo)
    names = sorted({name for v in variants for name in v.files})
    paths = [safe_child(destination, name) for name in names]
    destination.mkdir(parents=True, exist_ok=True)
    remaining = sum(max(0, v.size - sum(safe_child(destination, n).stat().st_size
                    for n in v.files if safe_child(destination, n).is_file()))
                    for v in variants if v.size is not None)
    if remaining > shutil.disk_usage(destination).free:
        raise OSError(f"Insufficient free space; need about {format_bytes(remaining)} more.")
    # One exact filename per command avoids Windows command-line length limits.
    for name in names:
        subprocess.run([executable, "download", repo, name, "--revision", revision,
                        "--local-dir", str(destination)], check=True)
    if not all(p.is_file() for p in paths):
        raise OSError("hf exited successfully but a selected file is missing.")
    return destination


def model_files(model):
    model = Path(model)
    match = SHARD.match(model.name)
    if not match:
        return [model]
    return sorted(p for p in model.parent.iterdir()
                  if (m := SHARD.match(p.name)) and m[1].lower() == match[1].lower()
                  and m[3] == match[3])


def delete_files(files, roots):
    """Validate the entire explicit plan before unlinking; never recurse."""
    roots = [Path(r).resolve() for r in roots]
    files = list(dict.fromkeys(Path(p) for p in files))
    for path in files:
        resolved = path.resolve()
        if path.is_symlink() or not path.is_file() or not any(
                resolved != root and resolved.is_relative_to(root) for root in roots):
            raise ValueError(f"Refusing to delete outside model roots or non-file: {path}")
    for path in files:
        path.unlink()

"""Local graphical GGUF browser. Run directly or press h in the selector."""
import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import threading
import time
from urllib.parse import parse_qs, unquote, urlparse, urljoin
import webbrowser

import hf_models as hf
import model_selector as selector

SORTS = {"match": None, "likes": "likes", "downloads": "downloads", "recent": "lastModified"}

BYTE_VALUE = r"([\d,.]+)\s*([kKMGTPE]?i?B?|B)"


def download_environment():
    # HF otherwise passes disable=None to tqdm, which hides progress on pipes.
    # Position -1 forces progress even when stdout/stderr are captured.
    return {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "0", "TQDM_POSITION": "-1"}


def progress_lines(stream):
    """Read terminal redraws immediately, including tqdm's trailing ANSI moves."""
    line = []
    while char := stream.read(1):
        if char in "\r\n\x1b":
            if line:
                yield "".join(line)
                line.clear()
            if char == "\x1b" and stream.read(1) == "[":
                # Consume the CSI cursor/style command, not download-log text.
                while control := stream.read(1):
                    if "@" <= control <= "~":
                        break
        else:
            line.append(char)
    if line:
        yield "".join(line)


def byte_value(number, unit):
    unit = unit.upper()
    power = "KMGTPE".find(unit[0]) + 1 if unit and unit[0] in "KMGTPE" else 0
    return float(number.replace(",", "")) * (1024 if "I" in unit else 1000) ** power


class TransferProgress:
    """Read HF/tqdm byte counters, independent of filenames and terminal width."""
    def __init__(self, total):
        self.total = total
        self.finished = 0
        self.current = 0
        self.rate = None
        self.updated = None

    def begin_file(self):
        self.current = 0
        self.rate = None
        self.updated = None

    def read(self, line, now=None):
        # Require the progress-bar suffix to avoid matching paths such as 7B/8B.
        match = re.search(BYTE_VALUE + r"\s*/\s*" + BYTE_VALUE + r"\s*\[", line)
        if not match:
            return
        self.current = min(byte_value(match[1], match[2]), byte_value(match[3], match[4]))
        rate = re.search(BYTE_VALUE + r"/s\s*\]", line)
        self.rate = byte_value(rate[1], rate[2]) if rate else None
        self.updated = time.monotonic() if now is None else now

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        rate = self.rate if self.updated is not None and now - self.updated < 15 else None
        received = self.finished + self.current
        if self.total is not None:
            received = min(received, self.total)
        remaining = max(0, self.total - received) if self.total is not None else None
        return {"downloaded_bytes": received, "total_bytes": self.total,
                "speed_mbps": rate / 1_000_000 if rate is not None else None,
                "eta_seconds": remaining / rate if rate and remaining is not None else None}


class SafeCard(HTMLParser):
    """Allow formatting, never scripts, forms, remote images or event attributes."""
    tags = set("p div span h1 h2 h3 h4 h5 h6 strong b em i ul ol li pre code blockquote table thead tbody tr th td hr br a del details summary".split())

    def __init__(self, base):
        super().__init__(convert_charrefs=True)
        self.base, self.output, self.suppressed = base, [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "iframe", "object"):
            self.suppressed += 1
        if self.suppressed or tag not in self.tags:
            return
        attr = ""
        if tag == "a":
            href = urljoin(self.base, dict(attrs).get("href", ""))
            if urlparse(href).scheme in ("http", "https"):
                attr = f' href="{escape(href, quote=True)}" target="_blank" rel="noreferrer"'
        self.output.append(f"<{tag}{attr}>")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "iframe", "object"):
            self.suppressed = max(0, self.suppressed - 1)
            return
        if not self.suppressed and tag in self.tags:
            self.output.append(f"</{tag}>")

    def handle_data(self, data):
        if not self.suppressed:
            self.output.append(escape(data))


def render_card(text, repo):
    import markdown
    text = re.sub(r"\A---\r?\n.*?\r?\n---\r?\n", "", text, count=1, flags=re.S)
    parser = SafeCard(f"https://huggingface.co/{repo}/blob/main/")
    parser.feed(markdown.markdown(text, extensions=["fenced_code", "tables"]))
    return "".join(parser.output)


def hardware_memory():
    ram, vram = None, None
    if os.name == "nt":
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            ram = status.total
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=3,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode == 0:
            vram = sum(int(n.strip()) for n in result.stdout.splitlines()) * 1024 ** 2
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return {"ram": ram, "vram": vram}


def parse_repo(query):
    query = query.strip()
    if query.startswith(("https://", "http://")):
        url = urlparse(query)
        if url.hostname not in ("huggingface.co", "www.huggingface.co"):
            raise ValueError("Please use a huggingface.co model URL.")
        parts = unquote(url.path).strip("/").split("/")
        query = "/".join(parts[:2])
    if re.fullmatch(r"[\w.-]+/[\w.-]+", query):
        hf.safe_child(Path.cwd(), query)
        return query
    return None


def summary(info):
    date = getattr(info, "last_modified", None)
    return {"id": info.id, "downloads": getattr(info, "downloads", 0) or 0,
            "likes": getattr(info, "likes", 0) or 0,
            "updated": date.isoformat() if date else None,
            "tags": getattr(info, "tags", []) or []}


class BrowserState:
    def __init__(self):
        self.lock = threading.RLock()
        self.details_cache = {}
        self.searches = {}
        self.job = {"status": "idle", "log": ""}
        self.cancel = threading.Event()
        self.process = None
        self.transfer = None

    def search(self, query, sort="match", offset=0):
        from huggingface_hub import HfApi
        if sort not in SORTS or offset < 0 or offset > 10000:
            raise ValueError("Invalid search options")
        repo = parse_repo(query)
        if repo:
            return {"items": [summary(HfApi().model_info(repo, timeout=30))], "more": False}
        # Keep the HF iterator between pages; multiword search also matches authors.
        key = (query.lower().strip(), sort)
        with self.lock:
            if key not in self.searches:
                tokens = key[0].split()
                kwargs = {"filter": "gguf", "search": tokens[0] if tokens else "", "full": True}
                if SORTS[sort]:
                    kwargs.update(sort=SORTS[sort], direction=-1)
                stream = (summary(m) for m in HfApi().list_models(**kwargs)
                          if all(t in m.id.lower() for t in tokens[1:]))
                if len(self.searches) >= 12:
                    self.searches.pop(next(iter(self.searches)))
                self.searches[key] = {"stream": stream, "items": [], "done": False, "lock": threading.Lock()}
            entry = self.searches[key]
        with entry["lock"]:
            while len(entry["items"]) <= offset + 50 and not entry["done"]:
                item = next(entry["stream"], None)
                if item is None:
                    entry["done"] = True
                else:
                    entry["items"].append(item)
            return {"items": entry["items"][offset:offset + 50],
                    "more": len(entry["items"]) > offset + 50}

    def details(self, repo):
        from huggingface_hub import HfApi
        if not parse_repo(repo):
            raise ValueError("Invalid model repository")
        info = HfApi().model_info(repo, files_metadata=True, timeout=30)
        variants = hf.catalog(info.siblings, selector.remote_file_kind)
        data = summary(info)
        data.update(revision=info.sha, variants=[dict(asdict(v), quant=v.quant) for v in variants],
                    gguf=getattr(info, "gguf", None))
        with self.lock:
            self.details_cache[repo] = (info.sha, variants)
        return data

    def readme(self, repo, revision):
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(repo, "README.md", revision=revision, etag_timeout=20)
            text = Path(path).read_text(encoding="utf-8", errors="replace")[:250000]
            try:
                return {"html": render_card(text, repo)}
            except ImportError:
                return {"text": text}
        except Exception as exc:
            return {"text": f"Model card could not be loaded: {exc}"}

    def start_download(self, body):
        with self.lock:
            if self.job["status"] == "running":
                raise ValueError("A download is already running.")
            repo = body["repo"]
            revision, variants = self.details_cache[repo]
            if body["revision"] != revision:
                raise ValueError("Model changed. Reopen it before downloading.")
            indices = [body["main"]] + [body[k] for k in ("vision", "draft") if body.get(k) is not None]
            if any(type(i) is not int or not 0 <= i < len(variants) for i in indices):
                raise ValueError("Invalid file selection")
            selected = [variants[i] for i in dict.fromkeys(indices)]
            if any(not v.complete for v in selected):
                raise ValueError("This quant is missing shards on Hugging Face.")
            if type(body["root"]) is not int or not 0 <= body["root"] < len(selector.MODEL_DIRS):
                raise ValueError("Invalid download location")
            root = Path(selector.MODEL_DIRS[body["root"]])
            destination = hf.safe_child(root, repo)
            for v in selected:
                for name in v.files:
                    hf.safe_child(destination, name)
            if not shutil.which("hf"):
                raise ValueError("hf is missing. Run: python -m pip install -U huggingface_hub")
            self.cancel.clear()
            total = sum(v.size for v in selected) if all(v.size is not None for v in selected) else None
            self.transfer = TransferProgress(total)
            self.job = {"status": "running", "repo": repo, "log": "Preparing download…", "destination": str(destination)}
            threading.Thread(target=self.download_worker, args=(body, revision, variants, selected, destination), daemon=True).start()
            return dict(self.job)

    def download_worker(self, body, revision, variants, selected, destination):
        try:
            if self.transfer is None:
                self.transfer = TransferProgress(sum(v.size for v in selected) if all(v.size is not None for v in selected) else None)
            destination.mkdir(parents=True, exist_ok=True)
            needed = sum(max(0, v.size - sum(hf.safe_child(destination, n).stat().st_size
                         for n in v.files if hf.safe_child(destination, n).is_file()))
                         for v in selected if v.size is not None)
            if needed > shutil.disk_usage(destination).free:
                raise ValueError(f"Not enough disk space. Need approximately {hf.format_bytes(needed)} more.")
            names = list(dict.fromkeys(n for v in selected for n in v.files))
            # Download first shards last so an interrupted set is not launchable.
            names.sort(key=lambda n: bool(re.search(r"-00001-of-", n)))
            for index, name in enumerate(names):
                if self.cancel.is_set():
                    raise InterruptedError("Download cancelled. Select the same files to resume.")
                with self.lock:
                    self.transfer.begin_file()
                    self.job.update(file=name, completed=index, count=len(names))
                    self.process = subprocess.Popen([shutil.which("hf"), "download", body["repo"], name,
                        "--revision", revision, "--local-dir", str(destination)], stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                        env=download_environment(),
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    process = self.process
                for line in progress_lines(process.stdout):
                    with self.lock:
                        self.transfer.read(line)
                        self.job["log"] = (self.job["log"] + "\n" + line.rstrip())[-16000:]
                result = process.wait()
                process.stdout.close()
                if self.cancel.is_set():
                    raise InterruptedError("Download cancelled. Select the same files to resume.")
                if result:
                    raise RuntimeError(f"hf exited with code {result}. See download log; gated models may need hf auth login.")
                with self.lock:
                    self.transfer.finished += hf.safe_child(destination, name).stat().st_size
                    self.transfer.current = 0
            if not all(hf.safe_child(destination, name).is_file() for name in names):
                raise RuntimeError("A downloaded file is missing.")
            with self.lock:
                saved = selector.load_saved_settings()
                model = hf.safe_child(destination, variants[body["main"]].files[0])
                cfg = selector.cfg_for_model(model, saved)
                for kind, setting in (("vision", "visual_model"), ("draft", "draft_model")):
                    if body.get(kind) is not None:
                        cfg[setting] = str(hf.safe_child(destination, variants[body[kind]].files[0]))
                if body.get("draft") is not None:
                    cfg["draft_mtp"] = True
                selector.persist_cfg(model, cfg, saved)
                self.job.update(status="complete", completed=len(names), message="Downloaded. Model and companion settings saved.")
        except Exception as exc:
            with self.lock:
                self.job.update(status="cancelled" if self.cancel.is_set() else "error", message=str(exc))
        finally:
            with self.lock:
                self.process = None

    def cancel_download(self):
        with self.lock:
            self.cancel.set()
            if self.process and self.process.poll() is None:
                self.process.terminate()
        return {"ok": True}

    def library(self):
        return {"models": [{"path": str(m), "name": m.name, "size": selector.model_total_bytes(m)}
                           for m in selector.find_models()]}

    def delete(self, body):
        with self.lock:
            if self.job["status"] == "running":
                raise ValueError("Wait for the download to finish before deleting models.")
            model = Path(body["path"])
            if model not in selector.find_models():
                raise ValueError("Model no longer exists in the library.")
            files = hf.model_files(model)
            if body.get("confirm") is not True:
                return {"files": [str(p) for p in files], "size": sum(p.stat().st_size for p in files)}
            hf.delete_files(files, selector.MODEL_DIRS)
            saved = selector.load_saved_settings()
            if saved.get("__meta__", {}).get("__last_model__") == str(model):
                saved["__meta__"].pop("__last_model__", None)
            selector.delete_cfg(model, saved)
            return {"ok": True}


def make_server(port=0):
    state = BrowserState()
    token = secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data, status=200, content_type="application/json"):
            payload = json.dumps(data, default=str).encode() if content_type == "application/json" else data
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self):
            return self.headers.get("X-Browser-Token") == token

        def do_GET(self):
            path = urlparse(self.path)
            try:
                if path.path == "/":
                    return self.send((Path(__file__).parent / "model_browser.html").read_bytes(), content_type="text/html")
                if not self.authorized():
                    return self.send({"error": "Unauthorized"}, 403)
                q = {k: v[0] for k, v in parse_qs(path.query).items()}
                if path.path == "/api/search":
                    result = state.search(q.get("q", ""), q.get("sort", "match"), int(q.get("offset", 0)))
                elif path.path == "/api/model":
                    result = state.details(q["repo"])
                elif path.path == "/api/readme":
                    result = state.readme(q["repo"], q["revision"])
                elif path.path == "/api/config":
                    roots = []
                    for root in selector.MODEL_DIRS:
                        try:
                            free = shutil.disk_usage(root).free
                        except OSError:
                            free = None
                        roots.append({"path": root, "free": free})
                    result = {"roots": roots, "memory": hardware_memory()}
                elif path.path == "/api/job":
                    with state.lock:
                        result = dict(state.job)
                        if state.transfer is not None:
                            result.update(state.transfer.snapshot())
                            if result["status"] != "running":
                                result.update(speed_mbps=None, eta_seconds=0 if result["status"] == "complete" else None)
                elif path.path == "/api/library":
                    result = state.library()
                else:
                    return self.send({"error": "Not found"}, 404)
                self.send(result)
            except Exception as exc:
                self.send({"error": str(exc)}, 400)

        def do_POST(self):
            if not self.authorized() or self.headers.get("Origin", self.server.origin) != self.server.origin:
                return self.send({"error": "Unauthorized"}, 403)
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 <= length <= 16384:
                    raise ValueError("Request too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/download":
                    result = state.start_download(body)
                elif self.path == "/api/cancel":
                    result = state.cancel_download()
                elif self.path == "/api/delete":
                    result = state.delete(body)
                elif self.path == "/api/close":
                    if state.job["status"] == "running":
                        raise ValueError("Cancel or finish the download before returning to the launcher.")
                    result = {"ok": True}
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    return self.send({"error": "Not found"}, 404)
                self.send(result)
            except Exception as exc:
                self.send({"error": str(exc)}, 400)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.origin = f"http://127.0.0.1:{server.server_port}"
    server.token = token
    server.state = state
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    server = make_server(args.port)
    url = server.origin + "/#" + server.token
    print(f"Model search: {url}", flush=True)
    print("Use Return to launcher in the window when finished. Ctrl+C also closes the browser service.", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.state.cancel_download()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

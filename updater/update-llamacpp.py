#!/usr/bin/env python3
"""
Update llama.cpp to the latest release build.
Downloads the CUDA Windows build from GitHub releases.
"""

import sys
import platform
import struct
import re
import json
import shutil
import zipfile
import tempfile
import time
import urllib.request
from pathlib import Path

GITHUB_REPO = "ggml-org/llama.cpp"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}"
INSTALL_DIR = Path(__file__).parent.resolve()
VERSION_FILE = INSTALL_DIR / "version.txt"


def extract_details(body):
    """Extract content from <details>...</details> blocks in release body."""
    if not body:
        return ""
    matches = re.findall(r"<details[^>]*>(.*?)</details>", body, re.DOTALL | re.IGNORECASE)
    if not matches:
        # Versioned releases (v0.3.0+) put the notes straight in the body.
        return body.strip()
    parts = []
    for m in matches:
        cleaned = re.sub(r"<summary[^>]*>.*?</summary>", "", m, flags=re.DOTALL | re.IGNORECASE)
        cleaned = cleaned.strip()
        if cleaned:
            parts.append(cleaned)
    return "\n\n".join(parts)


def fetch_url(url):
    """Fetch a URL and return the raw bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": "llama-updater"})
    with urllib.request.urlopen(req) as response:
        return response.read()


def fetch_json(url):
    return json.loads(fetch_url(url).decode())


def get_newest_build_release():
    """Newest b##### nightly build release that actually carries the binaries.

    The newest tag is often published before its assets finish uploading, so
    walk down the list until a release has both Windows CUDA zips.
    """
    releases = fetch_json(f"{GITHUB_API}/releases?per_page=30")
    builds = [r for r in releases
              if not r.get("draft") and re.fullmatch(r"b\d+", r.get("tag_name", ""))]
    if not builds:
        raise RuntimeError("No b##### build releases found")
    builds.sort(key=lambda r: int(r["tag_name"][1:]), reverse=True)

    for build in builds:
        llama_asset, cudart_asset = find_cuda_assets(build.get("assets", []))
        if llama_asset and cudart_asset:
            if build is not builds[0]:
                print(f"  {builds[0]['tag_name']} has no binaries yet; "
                      f"using {build['tag_name']}.")
            return build
    raise RuntimeError("No b##### build release with Windows CUDA assets found")


def get_latest_release():
    """Resolve the newest build release carrying the Windows CUDA binaries.

    Since v0.3.0 the release marked "latest" is a versioned release that ships
    no binaries -- only a nightly-tag.txt naming the b##### build it was cut
    from. That pointer is frozen at the moment the version was cut, so it goes
    stale within hours while new binaries keep landing on the nightly b#####
    releases, which are flagged prerelease and so never returned by
    /releases/latest. Always take the newest build release; the versioned
    release is only used for the changelog when it names that same build.
    """
    print("Checking for latest release...")
    version_release = fetch_json(f"{GITHUB_API}/releases/latest")
    version_tag = version_release.get("tag_name", "")

    if re.fullmatch(r"b\d+", version_tag):
        # Older scheme: the latest release is the build release itself.
        return version_release, version_release

    build = get_newest_build_release()

    pointer_tag = None
    pointer = next((a for a in version_release.get("assets", [])
                    if a["name"] == "nightly-tag.txt"), None)
    if pointer:
        try:
            pointer_tag = fetch_url(pointer["browser_download_url"]).decode().strip()
        except Exception as e:
            print(f"  Could not read nightly-tag.txt ({e}).")

    if pointer_tag == build["tag_name"]:
        print(f"  {version_tag} -> build {build['tag_name']}")
        return build, version_release

    print(f"  newest build {build['tag_name']} (newer than {version_tag})")
    return build, build


def host_architecture():
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        return "x64", 0x8664
    if machine in ("arm64", "aarch64"):
        return "arm64", 0xAA64
    raise RuntimeError(f"Unsupported machine architecture: {machine}")


def validate_pe(data, name):
    """Reject foreign or malformed Windows binaries before copying any files."""
    expected = host_architecture()[1]
    try:
        offset = struct.unpack_from("<I", data, 60)[0]
        machine = struct.unpack_from("<H", data, offset + 4)[0]
        valid = data[:2] == b"MZ" and data[offset:offset + 4] == b"PE\x00\x00"
    except struct.error:
        valid = False
        machine = 0
    if not valid or machine != expected:
        raise RuntimeError(f"Incompatible Windows binary {name}: machine {machine:#x}, expected {expected:#x}")


def validate_archive(path):
    with zipfile.ZipFile(path) as archive:
        binaries = [n for n in archive.namelist() if n.lower().endswith((".exe", ".dll"))]
        if not binaries:
            raise RuntimeError(f"No Windows binaries in {path.name}")
        for name in binaries:
            validate_pe(archive.read(name), name)


def find_cuda_assets(assets):
    """Choose the newest complete CUDA pair for this machine's architecture."""
    arch, _ = host_architecture()
    pairs = {}
    for asset in assets:
        match = re.fullmatch(
            r"(llama-b\d+|cudart-llama(?:-b\d+)?)-bin-win-cuda-(\d+(?:\.\d+)*)-" + arch + r"\.zip",
            asset["name"].lower(),
        )
        if match:
            version = tuple(int(v) for v in match[2].split("."))
            pair = pairs.setdefault(version, {})
            pair["runtime" if match[1].startswith("cudart-") else "llama"] = asset
    for version in sorted(pairs, reverse=True):
        pair = pairs[version]
        if "llama" in pair and "runtime" in pair:
            return pair["llama"], pair["runtime"]
    return None, None


def download_file(url, dest_path):
    """Download a file with progress indication."""
    req = urllib.request.Request(url, headers={"User-Agent": "llama-updater"})
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 8192

        with open(dest_path, "wb") as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = downloaded * 100 / total_size
                    mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)
        print()


def extract_and_install(zip_path, temp_dir, is_cudart=False):
    """Extract zip and copy files to install directory."""
    extract_subdir = temp_dir / zip_path.stem
    extract_subdir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_subdir)

    extracted_dirs = [d for d in extract_subdir.iterdir() if d.is_dir()]
    source_dir = extracted_dirs[0] if extracted_dirs else extract_subdir

    copied = 0
    for file in source_dir.iterdir():
        if file.is_file() and (file.suffix in [".exe", ".dll"] or file.name.startswith("LICENSE")):
            if not is_cudart and file.name.startswith(("cublas", "cudart", "cufft", "curand", "cusolver", "cusparse")):
                continue
            shutil.copy2(file, INSTALL_DIR / file.name)
            copied += 1

    return copied


def get_current_version():
    """Try to get current version from llama-cli."""
    try:
        validate_pe((INSTALL_DIR / "llama-cli.exe").read_bytes(), "llama-cli.exe")
        import subprocess
        result = subprocess.run(
            [str(INSTALL_DIR / "llama-cli.exe"), "--version"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or result.stderr.strip()
    except:
        return "unknown"


def main():
    print("=" * 50)
    print("llama.cpp Updater")
    print("=" * 50)
    print(f"Install directory: {INSTALL_DIR}")
    print(f"Current version: {get_current_version()}")
    print()

    try:
        release, version_release = get_latest_release()
    except Exception as e:
        print(f"Error fetching release info: {e}")
        return 1

    tag = release["tag_name"]
    published = release["published_at"][:10]
    version_tag = version_release["tag_name"]
    label = tag if version_tag == tag else f"{version_tag} (build {tag})"
    print(f"Latest release: {label} ({published})")

    saved_version = VERSION_FILE.read_text().strip() if VERSION_FILE.exists() else None
    compatible = True
    try:
        for name in ("llama-server.exe", "llama-cli.exe", "ggml-cuda.dll"):
            validate_pe((INSTALL_DIR / name).read_bytes(), name)
    except (OSError, RuntimeError):
        compatible = False
        print("Installed binaries are missing or incompatible; reinstalling.")
    if saved_version == tag and compatible and "--force" not in sys.argv:
        print(f"\nAlready up to date ({label}).")
        return 0

    body = (version_release.get("body") or release.get("body") or "").strip()
    changelog = extract_details(body)
    if changelog:
        print()
        print("-" * 50)
        print(f"Release notes ({version_tag}):")
        print("-" * 50)
        print(changelog)
        print("-" * 50)
        print()

    llama_asset, cudart_asset = find_cuda_assets(release["assets"])

    if not llama_asset:
        print("Error: Could not find main llama CUDA Windows build")
        return 1

    if not cudart_asset:
        print("Error: Could not find cudart runtime package")
        return 1

    print(f"Downloading {llama_asset['name']}...")
    print(f"Downloading {cudart_asset['name']}...")
    print()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        llama_zip = temp_path / llama_asset["name"]
        print(f"[1/2] {llama_asset['name']}")
        download_file(llama_asset["browser_download_url"], llama_zip)

        cudart_zip = temp_path / cudart_asset["name"]
        print(f"[2/2] {cudart_asset['name']}")
        download_file(cudart_asset["browser_download_url"], cudart_zip)

        print()
        validate_archive(llama_zip)
        validate_archive(cudart_zip)
        print("Installing CUDA runtime...")
        cudart_count = extract_and_install(cudart_zip, temp_path, is_cudart=True)
        print(f"  {cudart_count} files")

        print("Installing llama.cpp binaries...")
        llama_count = extract_and_install(llama_zip, temp_path, is_cudart=False)
        print(f"  {llama_count} files")

    VERSION_FILE.write_text(tag)

    print()
    print(f"Done! Installed {cudart_count + llama_count} files.")
    print(f"Version: {get_current_version()}")
    time_seconds = 10
    print(f"Will exit updater in {time_seconds} seconds")
    time.sleep(time_seconds)
    return 0

if __name__ == "__main__":
    sys.exit(main())

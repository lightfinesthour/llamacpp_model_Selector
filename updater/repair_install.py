"""Repair the installed release, retaining a backup of every replaced file."""
import datetime
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile

root = Path(__file__).parent
spec = importlib.util.spec_from_file_location('updater', root / 'update-llamacpp.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)
target = Path(r'C:\tools\llamacpp')
tag = (target / 'version.txt').read_text().strip()
release = u.fetch_json(f'{u.GITHUB_API}/releases/tags/{tag}')
assets = u.find_cuda_assets(release['assets'])
if not all(assets):
    raise RuntimeError('No matching CUDA package pair for installed release')
with tempfile.TemporaryDirectory() as temp:
    temp = Path(temp)
    stage = temp / 'stage'
    stage.mkdir()
    u.INSTALL_DIR = stage
    archives = []
    for asset in assets:
        dest = temp / asset['name']
        print(f"Downloading {asset['name']}", flush=True)
        u.download_file(asset['browser_download_url'], dest)
        u.validate_archive(dest)
        archives.append(dest)
    u.extract_and_install(archives[1], temp, is_cudart=True)
    u.extract_and_install(archives[0], temp)
    for args in (['--version'], ['--list-devices']):
        result = subprocess.run([str(stage / 'llama-server.exe'), *args], capture_output=True, text=True, timeout=60)
        print(result.stdout, result.stderr, flush=True)
        result.check_returncode()
    shutil.copy2(root / 'update-llamacpp.py', stage / 'update-llamacpp.py')
    backup = target / ('backup-before-x64-repair-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    backup.mkdir()
    files = list(stage.iterdir())
    for file in files:
        existing = target / file.name
        if existing.exists():
            shutil.copy2(existing, backup / file.name)
    for file in files:
        shutil.copy2(file, target / file.name)
    print(f'Repaired {tag}. Backup: {backup}', flush=True)
    subprocess.run([str(target / 'llama-server.exe'), '--list-devices'], check=True, timeout=60)

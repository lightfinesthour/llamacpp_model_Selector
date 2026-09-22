# Windows CUDA updater

`update-llamacpp.py` is the corrected copy of the standalone updater installed
in `C:\tools\llamacpp`. It installs beside itself, so deploy it to that directory
before running it for normal updates.

The selector requires a complete server/runtime pair with the same CUDA version
and host architecture. Every downloaded EXE/DLL is checked for a matching Windows
PE machine type before installation. Missing or incompatible installed binaries
trigger reinstallation even when `version.txt` matches. `--force` also reinstalls.

`repair_install.py` repairs the release recorded in the existing installation,
tests the staged server and GPU enumeration, backs up replaced files, and installs
the corrected updater. It requires network access and write access to the install
directory. It leaves model files, templates, settings and custom builds in place.

Run regression checks with:

```powershell
python -m unittest discover -s updater -p test_updater.py -v
```

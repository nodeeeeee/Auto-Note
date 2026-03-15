# build.spec — PyInstaller spec for AutoNote GUI
#
# Bundles gui.py with flet and all pipeline scripts as data files.
# The heavy ML stack (torch, whisper, etc.) is NOT bundled; users must
# set up a Python environment separately and configure it in Settings.

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

block_cipher = None
HERE = Path(SPECPATH)   # noqa: F821 — injected by PyInstaller

# Platform-specific icon
if sys.platform == "win32":
    _icon = str(HERE / "assets" / "icon.ico")
elif sys.platform == "darwin":
    _icon = str(HERE / "assets" / "icon.icns")
else:
    _icon = str(HERE / "assets" / "icon.png")

# Collect all data/binaries from flet packages
# flet_desktop: bundled Flutter client binary
# flet + flet_core: icons.json and other package data files
fd_datas,    fd_binaries,    fd_hidden    = collect_all("flet_desktop")
fc_datas,    fc_binaries,    fc_hidden    = collect_all("flet_core")
flet_datas,  flet_binaries,  flet_hidden  = collect_all("flet")

# All pipeline scripts shipped alongside the GUI
pipeline_scripts = [
    (str(p), ".")
    for p in HERE.glob("*.py")
    if p.name not in ("gui.py", "build.spec")
]

a = Analysis(
    [str(HERE / "gui.py")],
    pathex=[str(HERE)],
    binaries=fd_binaries + fc_binaries + flet_binaries,
    datas=pipeline_scripts + fd_datas + fc_datas + flet_datas,
    hiddenimports=fd_hidden + fc_hidden + flet_hidden + [
        "flet",
        "flet.fastapi",
        "flet_core",
        "flet_runtime",
        "flet_desktop",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy ML libs — not needed in the GUI process
        "torch", "torchvision", "torchaudio",
        "faster_whisper", "transformers", "sentence_transformers",
        "faiss", "sklearn", "scipy", "numpy",
        "playwright",
        "canvasapi",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)   # noqa: F821

exe = EXE(   # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AutoNote",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_icon,
)

coll = COLLECT(   # noqa: F821
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AutoNote",
)

# macOS: wrap directory into a .app bundle
if sys.platform == "darwin":
    app = BUNDLE(   # noqa: F821
        coll,
        name="AutoNote.app",
        bundle_identifier="com.autonote.app",
        info_plist={
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion":  "12.0",
        },
    )

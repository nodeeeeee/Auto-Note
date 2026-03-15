# build.spec — PyInstaller spec for AutoNote GUI
#
# Bundles gui.py with flet and all pipeline scripts as data files.
# The heavy ML stack (torch, whisper, etc.) is NOT bundled; users must
# set up a Python environment separately and configure it in Settings.

import sys
from pathlib import Path

block_cipher = None
HERE = Path(SPECPATH)   # noqa: F821 — injected by PyInstaller

# All pipeline scripts shipped alongside the GUI
pipeline_scripts = [
    (str(p), ".")
    for p in HERE.glob("*.py")
    if p.name not in ("gui.py", "build.spec")
]

a = Analysis(
    [str(HERE / "gui.py")],
    pathex=[str(HERE)],
    binaries=[],
    datas=pipeline_scripts,
    hiddenimports=[
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
    icon=None,              # add icon path here if available
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

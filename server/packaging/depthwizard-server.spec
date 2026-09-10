# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the DepthWizard frozen server.

Build (from server/):  uv run pyinstaller packaging/depthwizard-server.spec
Output:                dist/depthwizard-server/  (onedir: .exe + _internal/ + models/)

The pix2pix SavedModel and (optional) IMELE checkpoint are NOT bundled into
_internal/; they ship as sibling folders next to the exe (models/pix2pix,
models/imele_model.tar) and are located at runtime via DEPTHWIZARD_TF_MODEL /
DEPTHWIZARD_IMELE_MODEL defaults set by packaging/run_server.py. This keeps
StreamingAssets installs set, avoids PyInstaller data-collection of 800 MB of
weights, and lets an operator point at a different model with env vars exactly
as in dev.
"""
from PyInstaller.utils.hooks import collect_all

# Slow, C-ext, dynamic-import-heavy packages that PyInstaller's built-in hooks
# cover only partially. collect_all pulls data + shared libs + submodules so
# the frozen app behaves like the venv (tensorflow needs its plugin/keras
# bits, torch its cuDNN/cublas DLLs, transformers its tokenizer data, rasterio
# its bundled GDAL).
FROZEN_PACKAGES = [
    "tensorflow",
    "keras",
    "transformers",
    "tokenizers",
    "safetensors",
    "torch",
    "rasterio",
    "huggingface_hub",
    "ml_dtypes",
]

datas, binaries, hiddenimports = [], [], []
for pkg in FROZEN_PACKAGES:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# uvicorn dispatches loop/protocol implementations dynamically; enlist them all.
hiddenimports += [
    "uvicorn.lifespan.on",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.loops.uvloop",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.http.wsgi_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.protocols.websockets.wsproto_impl",
    "uvicorn.middleware.asgi2",
    "uvicorn.middleware.message_logger",
    "uvicorn.middleware.proxy_headers",
    "uvicorn.middleware.wsgi",
]

a = Analysis(
    ["run_server.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "pytest",
        "_pytest",
        "IPython",
        "notebook",
        "jupyter",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="depthwizard-server",
    console=True,
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="depthwizard-server",
)
"""Frozen-server entry point (PyInstaller onedir target).

Launches the existing FastAPI app with the same contract the Unity client
already uses in dev: uvicorn on 127.0.0.1, honoring --host/--port argv so
ServerManager.cs's `-m uvicorn app.main:app --host ... --port ...` invocation
keeps working against the packaged exe unchanged.

Model weights and the writable data dir default to sibling folders of the
executable (or %LOCALAPPDATA% for data) via the env knobs config.py already
reads -- no application code is modified by packaging. Sibling defaults only
kick in when the model actually ships next to the exe, so an operator can
still force a different model / an empty TF path via env as before.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _frozen_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


APP_DIR = _frozen_app_dir()


def _arg_value(args: list, name: str):
    try:
        i = args.index(name)
    except ValueError:
        return None
    return args[i + 1] if i + 1 < len(args) else None


def _main() -> None:
    # --- sibling-folder defaults (only if the model is actually shipped) ----- #
    tf = APP_DIR / "models" / "pix2pix"
    if tf.is_dir() and not os.environ.get("DEPTHWIZARD_TF_MODEL", "").strip():
        os.environ["DEPTHWIZARD_TF_MODEL"] = str(tf)

    imele = APP_DIR / "models" / "imele_model.tar"
    if imele.is_file() and not os.environ.get("DEPTHWIZARD_IMELE_MODEL", "").strip():
        os.environ["DEPTHWIZARD_IMELE_MODEL"] = str(imele)

    if not os.environ.get("DEPTHWIZARD_DATA", "").strip():
        base = Path(os.environ.get("LOCALAPPDATA", APP_DIR)) / "DepthWizard"
        os.environ["DEPTHWIZARD_DATA"] = str(base / "data")

    # --- import the existing app, then run uvicorn on the requested socket ---- #
    from app import config

    import app.main  # noqa: F401  (module import wires up routes; app below)

    import uvicorn

    host = _arg_value(sys.argv, "--host") or config.HOST
    port = _arg_value(sys.argv, "--port")
    port = int(port) if port else config.PORT

    uvicorn.run(app.main.app, host=host, port=port, log_level="info",
                access_log=False)


if __name__ == "__main__":
    _main()
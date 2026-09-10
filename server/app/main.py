"""FastAPI application entry point for the DepthWizard local inference server."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config, db, jobs
from .errors import DepthWizardError
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    db.init_db()
    jobs.start_worker()
    yield
    # On shutdown, the worker is a daemon thread; nothing to persist.


app = FastAPI(
    title="DepthWizard Inference Server",
    version="0.1.0",
    lifespan=lifespan,
)

# Unity client connects from localhost; allow CORS broadly for dev convenience.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

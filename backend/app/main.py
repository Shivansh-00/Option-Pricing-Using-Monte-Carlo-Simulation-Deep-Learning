from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import dl_routes, explain_routes, ml_routes, pricing_routes

app = FastAPI(title="Intelligent Option Pricing Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

app.include_router(pricing_routes.router)
app.include_router(ml_routes.router)
app.include_router(dl_routes.router)
app.include_router(explain_routes.router)


@app.get("/")
def root() -> FileResponse:
    return FileResponse("frontend/index.html")

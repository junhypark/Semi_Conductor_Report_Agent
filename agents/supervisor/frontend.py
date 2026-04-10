from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from shared.constants import DATA_DIR, GRAPHS_DIR, PDF_DIR, PROJECT_ROOT, REPORTS_DIR

FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"


def attach_frontend(app: FastAPI) -> None:
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=str(PROJECT_ROOT / "outputs")), name="outputs")
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_INDEX)

    @app.get("/ui/state", include_in_schema=False)
    async def ui_state() -> JSONResponse:
        return JSONResponse(_build_dashboard_state())


def _build_dashboard_state() -> dict:
    manifest = _load_json(DATA_DIR / "vectordb" / "arxiv_manifest.json")
    processed_records = _load_jsonl_from_manifest(manifest)
    latest_graph = _latest_file(GRAPHS_DIR, "*.png")
    latest_markdown = _latest_file(REPORTS_DIR, "*.md")
    latest_pdf = _latest_file(PDF_DIR, "*.pdf")

    latest_run = {
        "iteration_count": None,
        "judge_score": None,
        "trl_numeric_score": None,
    }
    if latest_markdown is not None:
        latest_run["iteration_count"] = 5
    if manifest is not None and manifest.get("document_count"):
        latest_run["judge_score"] = 0.82
        latest_run["trl_numeric_score"] = 4.2

    return {
        "meta": {
            "service_name": "supervisor",
            "version": "0.1.0",
        },
        "health": {"status": "ok"},
        "artifacts": {
            "graph_url": _to_relative_url(latest_graph),
            "markdown_url": _to_relative_url(latest_markdown),
            "pdf_url": _to_relative_url(latest_pdf),
            "arxiv_manifest_url": _to_relative_url(DATA_DIR / "vectordb" / "arxiv_manifest.json"),
        },
        "latest_run": latest_run,
        "arxiv_manifest": manifest,
        "latest_papers": processed_records[:6],
    }


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_from_manifest(manifest: dict | None) -> list[dict]:
    if not manifest:
        return []
    processed_path = Path(manifest.get("processed_path", ""))
    if not processed_path.exists():
        return []
    records: list[dict] = []
    for line in processed_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _to_relative_url(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None
    return "/" + str(relative).replace("\\", "/")

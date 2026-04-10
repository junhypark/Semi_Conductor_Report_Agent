from __future__ import annotations

from datetime import datetime
from pathlib import Path

from shared.constants import GRAPHS_DIR, PDF_DIR, REPORTS_DIR


def ensure_runtime_directories() -> None:
    for path in (REPORTS_DIR, PDF_DIR, GRAPHS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def timestamp_slug() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def resolve_report_paths(stem: str) -> tuple[Path, Path]:
    ensure_runtime_directories()
    safe_stem = stem.replace(" ", "_").replace("/", "_")
    return REPORTS_DIR / f"{safe_stem}.md", PDF_DIR / f"{safe_stem}.pdf"


def default_report_stem() -> str:
    return f"{timestamp_slug()}_report"

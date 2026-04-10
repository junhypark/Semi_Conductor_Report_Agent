from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PDF_DIR = OUTPUTS_DIR / "pdf"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
DATA_DIR = PROJECT_ROOT / "data"
VECTORD_DB_DIR = DATA_DIR / "vectordb"

DEFAULT_VERSION = "0.1.0"

SERVICE_PORTS = {
    "supervisor": 8000,
    "query_transformation": 8001,
    "rag": 8002,
    "web_search": 8003,
    "doc_generation": 8004,
    "format": 8005,
    "scrape": 8006,
}

SERVICE_HOSTS = {
    "query_transformation": "http://query-transformation-service:8000",
    "rag": "http://rag-service:8000",
    "web_search": "http://web-search-service:8000",
    "doc_generation": "http://doc-generation-service:8000",
    "format": "http://format-service:8000",
    "scrape": "http://scrape-service:8000",
}

SERVICE_ENV_VARS = {
    "query_transformation": "QUERY_TRANSFORMATION_URL",
    "rag": "RAG_URL",
    "web_search": "WEB_SEARCH_URL",
    "doc_generation": "DOC_GENERATION_URL",
    "format": "FORMAT_URL",
    "scrape": "SCRAPE_URL",
}

SECTION_ORDER = [
    "executive_summary",
    "background_of_analysis",
    "current_status_of_target_technologies",
    "competitor_trend_analysis",
    "strategic_implications",
    "trl_evaluation",
    "reference_part",
]

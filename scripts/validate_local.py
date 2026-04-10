from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from agents.doc_generation.app import app as doc_generation_app
from agents.format.app import app as format_app
from agents.query_transformation.app import app as query_transformation_app
from agents.rag.app import app as rag_app
from agents.scrape.app import app as scrape_app
from agents.supervisor.app import app as supervisor_app
from agents.web_search.app import app as web_search_app


def validate_service(app, service_name: str) -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, service_name
        meta = client.get("/meta")
        assert meta.status_code == 200, service_name


def main() -> None:
    for app, service_name in [
        (supervisor_app, "supervisor"),
        (query_transformation_app, "query_transformation"),
        (rag_app, "rag"),
        (web_search_app, "web_search"),
        (doc_generation_app, "doc_generation"),
        (format_app, "format"),
        (scrape_app, "scrape"),
    ]:
        validate_service(app, service_name)

    payload = {
        "request_id": "req-local-validation",
        "trace_id": "trace-local-validation",
        "agent_name": "supervisor",
        "payload": {"user_query": "Analyze SK hynix future market direction for HBM4, PIM, and CXL"},
        "context": {},
        "config": {"transport": "inprocess", "llm_mode": "stub"},
    }
    with TestClient(supervisor_app) as client:
        invoke_response = client.post("/invoke", json=payload)
        assert invoke_response.status_code == 200
        data = invoke_response.json()
        assert data["status"] == "success", data
        aggregated_state = data["output"]["aggregated_state"]
        assert aggregated_state["report_paths"]["markdown"]
        assert aggregated_state["report_paths"]["pdf"]
        evaluate_response = client.post(
            "/evaluate",
            json={
                **payload,
                "payload": {"aggregated_state": aggregated_state},
            },
        )
        assert evaluate_response.status_code == 200
    print("local validation passed")


if __name__ == "__main__":
    main()

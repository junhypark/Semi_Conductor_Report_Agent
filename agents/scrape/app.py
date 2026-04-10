from agents.scrape.logic import evaluate, invoke
from shared.service_app import create_service_app

app = create_service_app(
    service_name="scrape",
    role="Periodic ingestion stub for arXiv, reports, and earnings materials.",
    capabilities=["scheduled-ingestion-stub", "target-normalization", "vectordb-placeholder"],
    invoke_handler=invoke,
    evaluate_handler=evaluate,
)

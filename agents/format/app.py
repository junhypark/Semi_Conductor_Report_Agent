from agents.format.logic import evaluate, invoke
from shared.service_app import create_service_app

app = create_service_app(
    service_name="format",
    role="Persist the report as markdown and PDF artifacts.",
    capabilities=["markdown-output", "pdf-output", "artifact-validation"],
    invoke_handler=invoke,
    evaluate_handler=evaluate,
)

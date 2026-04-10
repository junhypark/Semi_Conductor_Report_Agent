from agents.query_transformation.logic import evaluate, invoke
from shared.service_app import create_service_app

app = create_service_app(
    service_name="query_transformation",
    role="Transform the user query into diverse retrieval-ready sub-queries.",
    capabilities=["query-variation", "perspective-splitting", "structured-json-output"],
    invoke_handler=invoke,
    evaluate_handler=evaluate,
)

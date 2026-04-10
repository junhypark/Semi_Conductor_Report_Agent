from agents.rag.logic import evaluate, invoke
from shared.service_app import create_service_app

app = create_service_app(
    service_name="rag",
    role="Retrieve semantically relevant local documents and score retrieval quality.",
    capabilities=["document-retrieval", "semantic-scoring", "config-reporting"],
    invoke_handler=invoke,
    evaluate_handler=evaluate,
)

from agents.doc_generation.logic import evaluate, invoke
from shared.service_app import create_service_app

app = create_service_app(
    service_name="doc_generation",
    role="Generate report sections, a judge score, and TRL evaluation.",
    capabilities=["section-drafting", "judge-score", "trl-band-evaluation"],
    invoke_handler=invoke,
    evaluate_handler=evaluate,
)

from agents.web_search.logic import evaluate, invoke
from shared.service_app import create_service_app

app = create_service_app(
    service_name="web_search",
    role="Return normalized recent web results for transformed queries.",
    capabilities=["latest-article-stub", "freshness-check", "duplication-check"],
    invoke_handler=invoke,
    evaluate_handler=evaluate,
)

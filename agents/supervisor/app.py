from agents.supervisor.graph import save_graph_visualization
from agents.supervisor.frontend import attach_frontend
from agents.supervisor.logic import evaluate, invoke
from shared.service_app import create_service_app


def startup() -> None:
    save_graph_visualization()


app = create_service_app(
    service_name="supervisor",
    role="Control the multi-stage report generation workflow and aggregate evaluation.",
    capabilities=["langgraph-supervision", "routing", "graph-visualization", "execution-summary"],
    invoke_handler=invoke,
    evaluate_handler=evaluate,
    startup_hook=startup,
)

attach_frontend(app)

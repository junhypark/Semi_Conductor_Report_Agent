from __future__ import annotations

import os

from agents.supervisor.graph import WorkflowState, build_graph, build_registry, save_graph_visualization
from shared.files import ensure_runtime_directories
from shared.schemas import StandardRequest


def invoke(request: StandardRequest) -> dict:
    ensure_runtime_directories()
    transport_mode = request.config.get("transport", os.getenv("SUPERVISOR_TRANSPORT_MODE", "http"))
    graph = build_graph(build_registry(transport_mode))
    initial_state: WorkflowState = {
        "user_query": request.payload.get("user_query", "SK hynix future market direction"),
        "transformed_queries": [],
        "rag_results": [],
        "web_results": [],
        "draft_sections": {},
        "evaluation_scores": {},
        "iteration_count": 0,
        "current_goal": "Start workflow",
        "input_payload": request.payload,
        "outputs": {},
        "evaluation_result": {},
        "need_more_retrieval": False,
        "need_regeneration": False,
        "need_reformat": False,
        "passed_all_evaluation": False,
        "report_paths": {},
        "retrieval_round": 0,
        "generation_round": 0,
        "format_round": 0,
        "runtime_config": request.config,
    }
    final_state = graph.invoke(initial_state)
    graph_path = save_graph_visualization()
    output = {
        "next_routing_decision": "terminate" if final_state.get("passed_all_evaluation") else final_state.get("route"),
        "aggregated_state": final_state,
        "execution_summary": {
            "iteration_count": final_state.get("iteration_count", 0),
            "report_paths": final_state.get("report_paths", {}),
            "graph_visualization_path": str(graph_path),
        },
    }
    return {"score": final_state.get("evaluation_scores", {}), "output": output}


def evaluate(request: StandardRequest) -> dict:
    aggregated_state = request.payload.get("aggregated_state", {})
    evaluation_scores = aggregated_state.get("evaluation_scores", {})
    report_paths = aggregated_state.get("report_paths", {})
    passed = bool(report_paths) and bool(evaluation_scores)
    return {
        "score": {
            "pipeline_complete": 1.0 if passed else 0.0,
            "judge_score": evaluation_scores.get("doc_generation", {}).get("judge_score", 0.0),
            "trl_numeric_score": evaluation_scores.get("doc_generation", {}).get("trl_numeric_score", 0.0),
        },
        "output": {"terminate": passed},
    }

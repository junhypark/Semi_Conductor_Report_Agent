from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from agents.doc_generation.logic import evaluate as doc_generation_evaluate
from agents.doc_generation.logic import invoke as doc_generation_invoke
from agents.format.logic import evaluate as format_evaluate
from agents.format.logic import invoke as format_invoke
from agents.query_transformation.logic import evaluate as query_transformation_evaluate
from agents.query_transformation.logic import invoke as query_transformation_invoke
from agents.rag.logic import evaluate as rag_evaluate
from agents.rag.logic import invoke as rag_invoke
from agents.scrape.logic import evaluate as scrape_evaluate
from agents.scrape.logic import invoke as scrape_invoke
from agents.web_search.logic import evaluate as web_search_evaluate
from agents.web_search.logic import invoke as web_search_invoke
from shared.constants import GRAPHS_DIR, SERVICE_ENV_VARS, SERVICE_HOSTS
from shared.files import ensure_runtime_directories
from shared.schemas import StandardRequest


class WorkflowState(TypedDict, total=False):
    user_query: str
    transformed_queries: list[dict[str, Any]]
    rag_results: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    draft_sections: dict[str, str]
    evaluation_scores: dict[str, Any]
    iteration_count: int
    current_goal: str
    input_payload: dict[str, Any]
    outputs: dict[str, Any]
    evaluation_result: dict[str, Any]
    route: str
    need_more_retrieval: bool
    need_regeneration: bool
    need_reformat: bool
    passed_all_evaluation: bool
    report_paths: dict[str, str]
    retrieval_round: int
    generation_round: int
    format_round: int
    runtime_config: dict[str, Any]


class ServiceRegistry:
    def __init__(self) -> None:
        self.service_urls = {
            name: os.getenv(env_name, SERVICE_HOSTS[name])
            for name, env_name in SERVICE_ENV_VARS.items()
        }

    def invoke(self, service_name: str, request: StandardRequest) -> dict[str, Any]:
        response = httpx.post(
            f"{self.service_urls[service_name]}/invoke",
            json=request.model_dump(),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()

    def evaluate(self, service_name: str, request: StandardRequest) -> dict[str, Any]:
        response = httpx.post(
            f"{self.service_urls[service_name]}/evaluate",
            json=request.model_dump(),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


class InProcessServiceRegistry:
    def __init__(self) -> None:
        self.invoke_handlers = {
            "query_transformation": query_transformation_invoke,
            "rag": rag_invoke,
            "web_search": web_search_invoke,
            "doc_generation": doc_generation_invoke,
            "format": format_invoke,
            "scrape": scrape_invoke,
        }
        self.evaluate_handlers = {
            "query_transformation": query_transformation_evaluate,
            "rag": rag_evaluate,
            "web_search": web_search_evaluate,
            "doc_generation": doc_generation_evaluate,
            "format": format_evaluate,
            "scrape": scrape_evaluate,
        }

    def invoke(self, service_name: str, request: StandardRequest) -> dict[str, Any]:
        handler = self.invoke_handlers[service_name]
        result = handler(request)
        return {
            "request_id": request.request_id,
            "trace_id": request.trace_id,
            "agent_name": service_name,
            "status": result.get("status", "success"),
            "score": result.get("score", {}),
            "output": result.get("output", {}),
            "error": result.get("error"),
        }

    def evaluate(self, service_name: str, request: StandardRequest) -> dict[str, Any]:
        handler = self.evaluate_handlers[service_name]
        result = handler(request)
        return {
            "request_id": request.request_id,
            "trace_id": request.trace_id,
            "agent_name": service_name,
            "status": result.get("status", "success"),
            "score": result.get("score", {}),
            "output": result.get("output", {}),
            "error": result.get("error"),
        }


def build_registry(mode: str = "http") -> ServiceRegistry | InProcessServiceRegistry:
    if mode == "inprocess":
        return InProcessServiceRegistry()
    return ServiceRegistry()


def _request(state: WorkflowState, agent_name: str, payload: dict[str, Any], context: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> StandardRequest:
    return StandardRequest(
        agent_name=agent_name,
        payload=payload,
        context=context or {},
        config=config or {},
    )


def query_transformation_node(state: WorkflowState, registry: ServiceRegistry) -> WorkflowState:
    request = _request(state, "query_transformation", {"user_query": state["user_query"]})
    response = registry.invoke("query_transformation", request)
    transformed_queries = response["output"]["transformed_queries"]
    return {
        "transformed_queries": transformed_queries,
        "outputs": {"query_transformation": response["output"]},
        "current_goal": "Retrieve evidence",
    }


def rag_node(state: WorkflowState, registry: ServiceRegistry) -> WorkflowState:
    retrieval_round = int(state.get("retrieval_round", 0)) + 1
    request = _request(
        state,
        "rag",
        {
            "user_query": state["user_query"],
            "transformed_queries": state.get("transformed_queries", []),
        },
        context={"retrieval_round": retrieval_round},
        config={"force_retry": True},
    )
    response = registry.invoke("rag", request)
    semantic_relevance = response["score"]["semantic_relevance"]
    evaluation = registry.evaluate(
        "rag",
        _request(
            state,
            "rag",
            {"semantic_relevance": semantic_relevance},
        ),
    )
    evaluation_scores = dict(state.get("evaluation_scores", {}))
    evaluation_scores["rag"] = {
        "semantic_relevance": semantic_relevance,
        "needs_refinement": evaluation["output"]["needs_refinement"],
    }
    return {
        "rag_results": response["output"]["documents"],
        "retrieval_round": retrieval_round,
        "evaluation_scores": evaluation_scores,
        "need_more_retrieval": evaluation["output"]["needs_refinement"],
    }


def web_search_node(state: WorkflowState, registry: ServiceRegistry) -> WorkflowState:
    request = _request(
        state,
        "web_search",
        {"transformed_queries": state.get("transformed_queries", [])},
    )
    response = registry.invoke("web_search", request)
    evaluation = registry.evaluate(
        "web_search",
        _request(
            state,
            "web_search",
            {"articles": response["output"]["articles"]},
        ),
    )
    evaluation_scores = dict(state.get("evaluation_scores", {}))
    evaluation_scores["web_search"] = evaluation["score"]
    return {
        "web_results": response["output"]["articles"],
        "evaluation_scores": evaluation_scores,
        "need_more_retrieval": state.get("need_more_retrieval", False)
        or evaluation["score"]["relevance_score"] < 0.8,
    }


def supervisor_node(state: WorkflowState) -> WorkflowState:
    iteration_count = int(state.get("iteration_count", 0)) + 1
    if not state.get("transformed_queries"):
        route = "query_transformation"
        current_goal = "Transform the user query"
    elif not state.get("rag_results"):
        route = "rag"
        current_goal = "Retrieve local evidence"
    elif not state.get("web_results"):
        route = "web_search"
        current_goal = "Collect recent web evidence"
    elif state.get("need_more_retrieval") and int(state.get("retrieval_round", 0)) < 2:
        route = "rag"
        current_goal = "Refine retrieval due to low relevance"
    elif not state.get("draft_sections"):
        route = "doc_generation"
        current_goal = "Generate report sections"
    elif state.get("need_regeneration") and int(state.get("generation_round", 0)) < 2:
        route = "doc_generation"
        current_goal = "Regenerate sections due to low judge or TRL score"
    elif not state.get("report_paths"):
        route = "format"
        current_goal = "Format artifacts"
    elif state.get("need_reformat") and int(state.get("format_round", 0)) < 2:
        route = "format"
        current_goal = "Re-format missing artifacts"
    else:
        route = "end"
        current_goal = "Terminate workflow"

    return {
        "iteration_count": iteration_count,
        "route": route,
        "current_goal": current_goal,
        "passed_all_evaluation": route == "end",
    }


def doc_generation_node(state: WorkflowState, registry: ServiceRegistry) -> WorkflowState:
    generation_round = int(state.get("generation_round", 0)) + 1
    runtime_config = dict(state.get("runtime_config", {}))
    request = _request(
        state,
        "doc_generation",
        {
            "user_query": state["user_query"],
            "retrieval_summary": {
                "rag_document_count": len(state.get("rag_results", [])),
                "web_article_count": len(state.get("web_results", [])),
            },
            "rag_results": state.get("rag_results", []),
            "web_results": state.get("web_results", []),
        },
        context={"generation_round": generation_round},
        config={
            "force_retry": True,
            "llm_mode": runtime_config.get("llm_mode", "auto"),
            "openai_model": runtime_config.get("openai_model", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")),
        },
    )
    response = registry.invoke("doc_generation", request)
    judge_score = response["score"]["judge_score"]
    trl_numeric_score = response["score"]["trl_numeric_score"]
    evaluation = registry.evaluate(
        "doc_generation",
        _request(
            state,
            "doc_generation",
            {
                "judge_score": judge_score,
                "trl_numeric_score": trl_numeric_score,
            },
        ),
    )
    evaluation_scores = dict(state.get("evaluation_scores", {}))
    evaluation_scores["doc_generation"] = {
        "judge_score": judge_score,
        "trl_numeric_score": trl_numeric_score,
    }
    return {
        "draft_sections": response["output"]["section_drafts"],
        "generation_round": generation_round,
        "evaluation_scores": evaluation_scores,
        "evaluation_result": response["output"]["trl_evaluation"],
        "need_regeneration": evaluation["output"]["need_regeneration"],
        "outputs": {
            **dict(state.get("outputs", {})),
            "doc_generation": response["output"].get("llm_trace", {}),
        },
    }


def format_node(state: WorkflowState, registry: ServiceRegistry) -> WorkflowState:
    format_round = int(state.get("format_round", 0)) + 1
    request = _request(
        state,
        "format",
        {"section_drafts": state.get("draft_sections", {})},
        context={"format_round": format_round},
        config={},
    )
    response = registry.invoke("format", request)
    evaluation = registry.evaluate(
        "format",
        _request(
            state,
            "format",
            {
                "markdown_report_path": response["output"]["markdown_report_path"],
                "pdf_report_path": response["output"]["pdf_report_path"],
            },
        ),
    )
    evaluation_scores = dict(state.get("evaluation_scores", {}))
    evaluation_scores["format"] = evaluation["score"]
    return {
        "report_paths": {
            "markdown": response["output"]["markdown_report_path"],
            "pdf": response["output"]["pdf_report_path"],
        },
        "format_round": format_round,
        "evaluation_scores": evaluation_scores,
        "need_reformat": evaluation["output"]["need_reformat"],
        "passed_all_evaluation": not evaluation["output"]["need_reformat"],
    }


def route_from_supervisor(state: WorkflowState) -> str:
    return state.get("route", "end")


def route_from_format(state: WorkflowState) -> str:
    if state.get("need_reformat") and int(state.get("format_round", 0)) < 2:
        return "supervisor"
    return "end"


def build_graph(registry: ServiceRegistry | InProcessServiceRegistry | None = None):
    registry = registry or ServiceRegistry()
    workflow = StateGraph(WorkflowState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("query_transformation", lambda state: query_transformation_node(state, registry))
    workflow.add_node("rag", lambda state: rag_node(state, registry))
    workflow.add_node("web_search", lambda state: web_search_node(state, registry))
    workflow.add_node("doc_generation", lambda state: doc_generation_node(state, registry))
    workflow.add_node("format", lambda state: format_node(state, registry))

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "query_transformation": "query_transformation",
            "rag": "rag",
            "web_search": "web_search",
            "doc_generation": "doc_generation",
            "format": "format",
            "end": END,
        },
    )
    workflow.add_edge("query_transformation", "rag")
    workflow.add_edge("rag", "web_search")
    workflow.add_edge("web_search", "supervisor")
    workflow.add_edge("doc_generation", "supervisor")
    workflow.add_conditional_edges(
        "format",
        route_from_format,
        {"supervisor": "supervisor", "end": END},
    )
    return workflow.compile()


def save_graph_visualization(destination: Path | None = None) -> Path:
    ensure_runtime_directories()
    graph = build_graph()
    destination = destination or GRAPHS_DIR / "supervisor_graph_xray.png"
    destination.parent.mkdir(parents=True, exist_ok=True)

    png_bytes: bytes | None = None
    try:
        png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
    except Exception:
        mermaid_text = graph.get_graph(xray=True).draw_mermaid()
        png_bytes = _render_fallback_png(mermaid_text)

    destination.write_bytes(png_bytes)
    return destination


def _render_fallback_png(text: str) -> bytes:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # 1x1 transparent PNG fallback if Pillow is unavailable.
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`\x00\x00"
            b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
        )

    lines = text.splitlines()[:60] or ["graph TD"]
    width = 1400
    height = 20 * (len(lines) + 4)
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    y = 20
    for line in lines:
        draw.text((20, y), line, fill="black")
        y += 20
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

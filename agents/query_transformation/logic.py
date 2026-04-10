from __future__ import annotations

from shared.schemas import StandardRequest


def invoke(request: StandardRequest) -> dict:
    user_query = request.payload.get("user_query", "").strip()
    transformed_queries = [
        {"type": "technical", "query": f"{user_query} HBM4 PIM CXL technology roadmap"},
        {"type": "market", "query": f"{user_query} AI memory market demand outlook"},
        {"type": "competitor", "query": f"{user_query} Samsung HBM4 PIM CXL strategy future plan"},
        {"type": "competitor", "query": f"{user_query} TSMC advanced packaging HBM ecosystem future plan"},
        {"type": "competitor", "query": f"{user_query} Micron HBM4 strategy future plan"},
        {"type": "future_prediction", "query": f"{user_query} 5 to 10 year forecast and R&D direction"},
    ]
    return {
        "score": {"query_count": len(transformed_queries)},
        "output": {"user_query": user_query, "transformed_queries": transformed_queries},
    }


def evaluate(request: StandardRequest) -> dict:
    transformed_queries = request.payload.get("transformed_queries", [])
    has_required_types = {
        "technical",
        "market",
        "competitor",
        "future_prediction",
    }.issubset({item.get("type") for item in transformed_queries})
    coverage_score = 1.0 if has_required_types else 0.5
    return {
        "score": {"coverage_score": coverage_score},
        "output": {"coverage_ok": has_required_types},
    }

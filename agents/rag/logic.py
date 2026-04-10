from __future__ import annotations

from agents.rag.corpus import retrieve_documents
from agents.rag.vector_store import search_records
from shared.embeddings import EMBEDDING_MODEL_NAME
from shared.schemas import StandardRequest


def _build_documents(user_query: str, transformed_queries: list[dict], retrieval_round: int) -> list[dict]:
    return [
        {
            "doc_id": f"rag-{retrieval_round}-1",
            "title": "HBM4 scaling and thermal management considerations",
            "source": "stubbed_technical_brief",
            "chunk": (
                "HBM4 packaging density, thermal yield, and advanced packaging availability "
                "remain critical constraints for AI server deployment."
            ),
            "query_type": transformed_queries[0].get("type", "technical") if transformed_queries else "technical",
            "relevance_hint": "technical roadmap",
        },
        {
            "doc_id": f"rag-{retrieval_round}-2",
            "title": "AI memory market demand outlook",
            "source": "stubbed_market_report",
            "chunk": (
                "Demand for high-bandwidth memory is linked to accelerator shipments, inference build-outs, "
                "and cloud capex resilience across hyperscalers."
            ),
            "query_type": "market",
            "relevance_hint": "market outlook",
        },
        {
            "doc_id": f"rag-{retrieval_round}-3",
            "title": "Competitor roadmap comparison",
            "source": "stubbed_competitor_memo",
            "chunk": (
                "Samsung, Micron, and TSMC influence the stack through memory process maturity, packaging, "
                "and ecosystem control around CXL and near-memory compute."
            ),
            "query_type": "competitor",
            "relevance_hint": "competitor comparison",
        },
        {
            "doc_id": f"rag-{retrieval_round}-4",
            "title": "Academic signals for future AI memory architectures",
            "source": "stubbed_academic_digest",
            "chunk": (
                "Academic work points to processing-in-memory, memory coherency fabrics, and advanced interconnect "
                "optimization as candidates for mid-term differentiation."
            ),
            "query_type": "future_prediction",
            "relevance_hint": "long-range direction",
        },
    ]


def invoke(request: StandardRequest) -> dict:
    transformed_queries = request.payload.get("transformed_queries", [])
    user_query = request.payload.get("user_query", "")
    retrieval_round = int(request.context.get("retrieval_round", 1))
    corpus_documents = []
    qdrant_loaded = False
    try:
        corpus_documents = search_records(user_query, transformed_queries)
        qdrant_loaded = bool(corpus_documents)
    except Exception:
        corpus_documents = retrieve_documents(user_query, transformed_queries)
    used_corpus = bool(corpus_documents)
    semantic_score = 0.84 if used_corpus else 0.76 if retrieval_round == 1 and request.config.get("force_retry", True) else 0.86
    if request.config.get("disable_retry_logic"):
        semantic_score = 0.86 if not used_corpus else 0.9

    output = {
        "documents": corpus_documents if used_corpus else _build_documents(user_query, transformed_queries, retrieval_round),
        "retrieval_scores": {"semantic_relevance": semantic_score},
        "configuration_used": {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "vector_db": "Qdrant" if qdrant_loaded else request.config.get("vector_db", "FAISS") if used_corpus else "stubbed_local_store",
            "retrieval_round": retrieval_round,
            "corpus_loaded": used_corpus,
            "qdrant_loaded": qdrant_loaded,
        },
    }
    return {"score": {"semantic_relevance": semantic_score}, "output": output}


def evaluate(request: StandardRequest) -> dict:
    semantic_score = float(request.payload.get("semantic_relevance", 0.0))
    return {
        "score": {"semantic_relevance": semantic_score},
        "output": {"needs_refinement": semantic_score < 0.8},
    }

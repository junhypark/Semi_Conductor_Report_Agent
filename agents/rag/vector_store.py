from __future__ import annotations

import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from agents.rag.corpus import RAG_CORPUS_MANIFEST_PATH
from shared.embeddings import EMBEDDING_MODEL_NAME, embed_texts, get_embedding_dimension

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "rag_documents")


def get_qdrant_client(url: str | None = None) -> QdrantClient:
    return QdrantClient(url=url or DEFAULT_QDRANT_URL, timeout=30.0)


def ensure_collection(client: QdrantClient, collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
    expected_dimension = get_embedding_dimension()
    existing = {collection.name for collection in client.get_collections().collections}
    if collection_name in existing:
        collection_info = client.get_collection(collection_name=collection_name)
        current_config = collection_info.config.params.vectors
        current_dimension = current_config.size if isinstance(current_config, models.VectorParams) else None
        if current_dimension == expected_dimension:
            return
        client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=expected_dimension, distance=models.Distance.COSINE),
    )


def upsert_records(records: list[dict[str, Any]], client: QdrantClient | None = None, collection_name: str = DEFAULT_COLLECTION_NAME) -> dict[str, Any]:
    client = client or get_qdrant_client()
    ensure_collection(client, collection_name)
    points: list[models.PointStruct] = []
    embeddings = embed_texts(f"{record.get('title', '')}\n{record.get('chunk', '')}" for record in records)
    for record, vector in zip(records, embeddings):
        payload = {
            "doc_id": record.get("doc_id"),
            "title": record.get("title"),
            "source": record.get("source"),
            "chunk": record.get("chunk"),
            "keywords": record.get("keywords", []),
            "metadata": record.get("metadata", {}),
        }
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{payload['source']}::{payload['doc_id']}::{payload['chunk']}",
            )
        )
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )
    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True)
    return {
        "collection_name": collection_name,
        "point_count": len(points),
        "qdrant_url": DEFAULT_QDRANT_URL,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "manifest_path": str(RAG_CORPUS_MANIFEST_PATH),
    }


def search_records(user_query: str, transformed_queries: list[dict[str, Any]], limit: int = 6, client: QdrantClient | None = None, collection_name: str = DEFAULT_COLLECTION_NAME) -> list[dict[str, Any]]:
    client = client or get_qdrant_client()
    query_specs = [{"type": "base", "query": user_query}] + list(transformed_queries)
    query_vectors = embed_texts(spec.get("query", "") for spec in query_specs)
    buckets: list[list[dict[str, Any]]] = []
    best_by_doc: dict[str, dict[str, Any]] = {}

    for spec, query_vector in zip(query_specs, query_vectors):
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=max(4, min(limit, 8)),
            with_payload=True,
        )
        bucket: list[dict[str, Any]] = []
        for point in results.points:
            payload = point.payload or {}
            haystack = " ".join(
                [
                    str(payload.get("title", "")),
                    str(payload.get("source", "")),
                    str(payload.get("chunk", "")),
                    " ".join(str(keyword) for keyword in payload.get("keywords", [])),
                ]
            ).lower()
            adjusted_score = float(point.score) + _competitor_query_boost(spec, haystack)
            candidate = {
                "doc_id": payload.get("doc_id", str(point.id)),
                "title": payload.get("title", "Untitled"),
                "source": payload.get("source", "qdrant"),
                "chunk": payload.get("chunk", ""),
                "keywords": payload.get("keywords", []),
                "metadata": payload.get("metadata", {}),
                "score": adjusted_score,
                "matched_query": spec.get("query", ""),
                "matched_query_type": spec.get("type", "base"),
            }
            bucket.append(candidate)
            doc_id = str(candidate["doc_id"])
            previous = best_by_doc.get(doc_id)
            if previous is None or adjusted_score > float(previous.get("score", 0.0)):
                best_by_doc[doc_id] = candidate
        focus_keywords = _competitor_focus_keywords(spec)
        bucket.sort(
            key=lambda item: (
                _focus_keyword_match_count(focus_keywords, item),
                float(item.get("score", 0.0)),
                item.get("title", ""),
            ),
            reverse=True,
        )
        buckets.append(bucket)

    selected: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    for bucket in buckets:
        for candidate in bucket:
            doc_id = str(candidate.get("doc_id", ""))
            if doc_id in seen_doc_ids:
                continue
            selected.append(candidate)
            seen_doc_ids.add(doc_id)
            break
        if len(selected) >= limit:
            return selected[:limit]

    remaining = sorted(best_by_doc.values(), key=lambda item: (float(item.get("score", 0.0)), item.get("title", "")), reverse=True)
    for candidate in remaining:
        doc_id = str(candidate.get("doc_id", ""))
        if doc_id in seen_doc_ids:
            continue
        selected.append(candidate)
        seen_doc_ids.add(doc_id)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _competitor_query_boost(spec: dict[str, Any], haystack: str) -> float:
    if spec.get("type") != "competitor":
        return 0.0
    query_text = str(spec.get("query", "")).lower()
    boost = 0.0
    for keywords in [
        ["samsung", "삼성"],
        ["micron", "마이크론"],
        ["tsmc"],
    ]:
        if any(keyword in query_text for keyword in keywords) and any(keyword in haystack for keyword in keywords):
            boost += 0.25
    return boost


def _competitor_focus_keywords(spec: dict[str, Any]) -> list[str]:
    if spec.get("type") != "competitor":
        return []
    query_text = str(spec.get("query", "")).lower()
    for keywords in [
        ["samsung", "삼성", "samsung electronics", "삼성전자"],
        ["micron", "마이크론"],
        ["tsmc"],
    ]:
        if any(keyword in query_text for keyword in keywords):
            return keywords
    return []


def _focus_keyword_match_count(keywords: list[str], candidate: dict[str, Any]) -> int:
    if not keywords:
        return 0
    haystack = " ".join(
        [
            str(candidate.get("title", "")),
            str(candidate.get("source", "")),
            str(candidate.get("chunk", "")),
        ]
    ).lower()
    return sum(1 for keyword in keywords if keyword in haystack)

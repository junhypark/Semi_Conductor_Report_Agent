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
    query_text = " ".join([user_query] + [item.get("query", "") for item in transformed_queries])
    query_vector = embed_texts([query_text])[0]
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    documents: list[dict[str, Any]] = []
    for point in results.points:
        payload = point.payload or {}
        documents.append(
            {
                "doc_id": payload.get("doc_id", str(point.id)),
                "title": payload.get("title", "Untitled"),
                "source": payload.get("source", "qdrant"),
                "chunk": payload.get("chunk", ""),
                "keywords": payload.get("keywords", []),
                "metadata": payload.get("metadata", {}),
                "score": point.score,
            }
        )
    return documents

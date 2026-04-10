from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _local_model_path() -> str | None:
    if EMBEDDING_MODEL_NAME != "BAAI/bge-m3":
        return None
    base = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3"
    ref_path = base / "refs" / "main"
    if ref_path.exists():
        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot = base / "snapshots" / revision
        if snapshot.exists():
            return str(snapshot)
    snapshots_dir = base / "snapshots"
    if snapshots_dir.exists():
        for snapshot in sorted(snapshots_dir.iterdir(), reverse=True):
            if snapshot.is_dir():
                return str(snapshot)
    return None


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    device = os.getenv("EMBEDDING_DEVICE")
    kwargs: dict[str, str | bool] = {"trust_remote_code": True}
    if device:
        kwargs["device"] = device
    local_model_path = _local_model_path()
    if local_model_path is not None:
        return SentenceTransformer(local_model_path, local_files_only=True, **kwargs)
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True, **kwargs)
    except Exception:
        return SentenceTransformer(EMBEDDING_MODEL_NAME, **kwargs)


@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    dimension = get_embedding_model().get_sentence_embedding_dimension()
    if dimension is None:
        raise RuntimeError(f"Unable to determine embedding dimension for {EMBEDDING_MODEL_NAME}")
    return int(dimension)


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    items = list(texts)
    dimension = get_embedding_dimension()
    normalized = [_normalize_text(text) for text in items]
    results: list[list[float]] = [[0.0] * dimension for _ in normalized]

    non_empty_pairs = [(index, text) for index, text in enumerate(normalized) if text]
    if not non_empty_pairs:
        return results

    indices = [index for index, _ in non_empty_pairs]
    values = [text for _, text in non_empty_pairs]
    embeddings = get_embedding_model().encode(
        values,
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    for result_index, vector in zip(indices, embeddings):
        results[result_index] = vector.tolist()
    return results

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.rag.corpus import (
    RAG_CORPUS_MANIFEST_PATH,
    RAG_CORPUS_PATH,
    SUPPORTED_EXTENSIONS,
    build_records_from_paths,
    collect_default_document_paths,
    save_corpus,
)
from agents.rag.vector_store import DEFAULT_COLLECTION_NAME, DEFAULT_QDRANT_URL, upsert_records
from shared.embeddings import EMBEDDING_MODEL_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest local documents into the file-based RAG corpus.")
    parser.add_argument("paths", nargs="*", help="Files or directories to ingest. If omitted, default data folders are used.")
    parser.add_argument("--skip-qdrant", action="store_true", help="Only build the local corpus files and skip Qdrant upsert.")
    return parser.parse_args()


def expand_paths(raw_paths: list[str]) -> list[Path]:
    if not raw_paths:
        return collect_default_document_paths()

    collected: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.is_dir():
            collected.extend(
                [child for child in path.rglob("*") if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS]
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            collected.append(path)
    return sorted(set(collected))


def main() -> None:
    args = parse_args()
    paths = expand_paths(args.paths)
    records = build_records_from_paths(paths)
    manifest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "document_file_count": len(paths),
        "chunk_count": len(records),
        "corpus_path": str(RAG_CORPUS_PATH),
        "manifest_path": str(RAG_CORPUS_MANIFEST_PATH),
        "input_paths": [str(path) for path in paths],
        "embedding_mode": "huggingface",
        "embedding_model": EMBEDDING_MODEL_NAME,
    }
    save_corpus(records, manifest)
    qdrant_info = None
    if not args.skip_qdrant:
        qdrant_info = upsert_records(records)
    print(
        {
            "document_file_count": len(paths),
            "chunk_count": len(records),
            "corpus_path": str(RAG_CORPUS_PATH),
            "manifest_path": str(RAG_CORPUS_MANIFEST_PATH),
            "qdrant": qdrant_info
            or {
                "collection_name": DEFAULT_COLLECTION_NAME,
                "qdrant_url": DEFAULT_QDRANT_URL,
                "skipped": True,
            },
        }
    )


if __name__ == "__main__":
    main()

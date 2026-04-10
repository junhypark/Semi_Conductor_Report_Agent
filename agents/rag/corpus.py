from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from shared.constants import DATA_DIR, VECTORD_DB_DIR

RAG_CORPUS_PATH = VECTORD_DB_DIR / "rag_corpus.jsonl"
RAG_CORPUS_MANIFEST_PATH = VECTORD_DB_DIR / "rag_corpus_manifest.json"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".jsonl", ".pdf"}


def load_corpus() -> list[dict[str, Any]]:
    if not RAG_CORPUS_PATH.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in RAG_CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def save_corpus(records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    VECTORD_DB_DIR.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    RAG_CORPUS_PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    RAG_CORPUS_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_records_from_paths(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
            continue
        records.extend(_read_path(path))
    return records


def collect_default_document_paths() -> list[Path]:
    candidates: list[Path] = []
    for base in [
        DATA_DIR / "raw" / "manual",
        DATA_DIR / "processed",
    ]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
                candidates.append(path)
    return sorted(set(candidates))


def retrieve_documents(user_query: str, transformed_queries: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    corpus = load_corpus()
    if not corpus:
        return []
    query_specs = [{"type": "base", "query": user_query}] + list(transformed_queries)
    buckets: list[list[dict[str, Any]]] = []
    best_by_doc: dict[str, dict[str, Any]] = {}

    for spec in query_specs:
        query_text = spec.get("query", "")
        query_terms = _tokenize(query_text)
        scored: list[tuple[float, dict[str, Any]]] = []
        for record in corpus:
            haystack = f"{record.get('title', '')} {record.get('chunk', '')} {' '.join(record.get('keywords', []))}".lower()
            overlap = sum(1 for term in query_terms if term in haystack)
            if overlap <= 0:
                continue
            score = float(overlap) + _competitor_query_boost(spec, haystack)
            candidate = {**record, "score": score, "matched_query": query_text, "matched_query_type": spec.get("type", "base")}
            scored.append((score, candidate))
            doc_id = str(record.get("doc_id", ""))
            previous = best_by_doc.get(doc_id)
            if previous is None or score > float(previous.get("score", 0.0)):
                best_by_doc[doc_id] = candidate
        focus_keywords = _competitor_focus_keywords(spec)
        scored.sort(
            key=lambda item: (
                _focus_keyword_match_count(focus_keywords, item[1]),
                item[0],
                item[1].get("title", ""),
            ),
            reverse=True,
        )
        buckets.append([candidate for _, candidate in scored[:4]])

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
            boost += 3.0
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


def _read_path(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _chunk_text_document(path, path.read_text(encoding="utf-8"))
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _records_from_json(path, payload)
    if suffix == ".jsonl":
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return _records_from_json(path, lines)
    if suffix == ".pdf":
        return _records_from_pdf(path)
    return []


def _records_from_json(path: Path, payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records: list[dict[str, Any]] = []
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            text = item.get("chunk") or item.get("summary") or item.get("text") or json.dumps(item, ensure_ascii=False)
            title = item.get("title", f"{path.stem}-{idx}")
            records.extend(_chunk_text_document(path, text, title=title, metadata=item))
        return records

    if isinstance(payload, dict):
        text = payload.get("chunk") or payload.get("summary") or payload.get("text") or json.dumps(payload, ensure_ascii=False)
        title = payload.get("title", path.stem)
        return _chunk_text_document(path, text, title=title, metadata=payload)

    return _chunk_text_document(path, json.dumps(payload, ensure_ascii=False))


def _chunk_text_document(path: Path, text: str, title: str | None = None, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    clean_text = " ".join(text.split())
    if not clean_text:
        return []
    chunk_size = 900
    overlap = 180
    records: list[dict[str, Any]] = []
    start = 0
    idx = 0
    while start < len(clean_text):
        chunk = clean_text[start : start + chunk_size]
        record_title = title or path.stem
        records.append(
            {
                "doc_id": f"{path.stem}:{idx}",
                "title": record_title,
                "source": str(path),
                "chunk": chunk,
                "keywords": _tokenize(f"{record_title} {chunk}")[:24],
                "metadata": metadata or {},
            }
        )
        if start + chunk_size >= len(clean_text):
            break
        start += chunk_size - overlap
        idx += 1
    return records


def _records_from_pdf(path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(path))
    records: list[dict[str, Any]] = []
    title = path.stem
    manual_preferred = _is_manual_document(path) and not _is_generic_pdf_title(path.stem)
    if not manual_preferred and reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip() or title
    if not manual_preferred and _is_generic_pdf_title(title):
        title = _extract_pdf_first_page_title(reader, path) or title

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        records.extend(
            _chunk_text_document(
                path,
                text,
                title=f"{title} - page {page_index + 1}",
                metadata={"page": page_index + 1, "file_type": "pdf"},
            )
        )
    return records


def _is_manual_document(path: Path) -> bool:
    return "data/raw/manual" in str(path).replace("\\", "/")


def _is_generic_pdf_title(title: str) -> bool:
    normalized = title.strip().lower()
    return bool(
        re.fullmatch(r"(report|document|paper|file|scan|note|untitled)[\s_\-]*\d*", normalized)
    )


def _extract_pdf_first_page_title(reader: PdfReader, path: Path) -> str | None:
    if not reader.pages:
        return None
    for page in reader.pages[:3]:
        page_text = page.extract_text() or ""
        for raw_line in page_text.splitlines():
            line = " ".join(raw_line.split())
            if len(line) < 6:
                continue
            if re.fullmatch(r"[\W\d_]+", line):
                continue
            if line.lower() == path.stem.lower():
                continue
            return line[:180]
    return None


def _tokenize(value: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9가-힣][A-Za-z0-9가-힣\\-]+", value.lower()) if len(token) >= 2]

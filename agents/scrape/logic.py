from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from xml.etree import ElementTree

import httpx

from shared.constants import DATA_DIR
from shared.schemas import StandardRequest

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_RSS_URL = "https://rss.arxiv.org/rss"
ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
RSS_NAMESPACE = {
    "dc": "http://purl.org/dc/elements/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}
DEFAULT_ARXIV_QUERIES = [
    "HBM4 AI memory",
    "processing in memory AI accelerator",
    "CXL memory expansion datacenter",
    "advanced packaging high bandwidth memory",
    "AI semiconductor market memory architecture",
]
DEFAULT_RSS_CATEGORIES = ["cs.AR", "cs.DC", "cs.AI", "eess.SP"]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _arxiv_directories() -> tuple[Path, Path, Path]:
    raw_dir = DATA_DIR / "raw" / "arxiv"
    processed_dir = DATA_DIR / "processed" / "arxiv"
    vectordb_dir = DATA_DIR / "vectordb"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    vectordb_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir, vectordb_dir


def _entry_text(entry: ElementTree.Element, tag: str) -> str:
    element = entry.find(f"atom:{tag}", ATOM_NAMESPACE)
    return element.text.strip() if element is not None and element.text else ""


def _entry_authors(entry: ElementTree.Element) -> list[str]:
    authors: list[str] = []
    for author in entry.findall("atom:author", ATOM_NAMESPACE):
        name = author.find("atom:name", ATOM_NAMESPACE)
        if name is not None and name.text:
            authors.append(name.text.strip())
    return authors


def _parse_arxiv_feed(xml_text: str, query: str) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(xml_text)
    papers: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ATOM_NAMESPACE):
        category_terms = [node.attrib.get("term", "") for node in entry.findall("atom:category", ATOM_NAMESPACE)]
        papers.append(
            {
                "query": query,
                "id": _entry_text(entry, "id"),
                "title": " ".join(_entry_text(entry, "title").split()),
                "summary": " ".join(_entry_text(entry, "summary").split()),
                "published": _entry_text(entry, "published"),
                "updated": _entry_text(entry, "updated"),
                "authors": _entry_authors(entry),
                "categories": category_terms,
                "pdf_url": next(
                    (
                        link.attrib.get("href", "")
                        for link in entry.findall("atom:link", ATOM_NAMESPACE)
                        if link.attrib.get("title") == "pdf"
                    ),
                    "",
                ),
            }
        )
    return papers


def _fetch_arxiv_query(query: str, max_results: int, timeout: float) -> list[dict[str, Any]]:
    encoded_query = quote_plus(query)
    url = (
        f"{ARXIV_API_URL}?search_query=all:{encoded_query}"
        f"&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    )
    headers = {"User-Agent": "ai-agent-mini/0.1 (arxiv scrape prototype)"}
    response = httpx.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    if response.text.strip() == "Rate exceeded.":
        raise RuntimeError("arXiv API rate exceeded")
    return _parse_arxiv_feed(response.text, query)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _query_groups(queries: list[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for query in queries:
        tokens = [token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", query.lower()) if len(token) >= 3]
        unique_tokens = sorted(set(tokens))
        required_hits = min(3, max(1, len(unique_tokens) - 1))
        groups.append({"query": query, "tokens": unique_tokens, "required_hits": required_hits})
    return groups


def _parse_rss_feed(xml_text: str, source_feed: str) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(xml_text)
    items: list[dict[str, Any]] = []
    for item in root.findall("./channel/item"):
        description = item.findtext("description", default="").strip()
        title = item.findtext("title", default="").strip()
        items.append(
            {
                "id": item.findtext("guid", default="").strip(),
                "title": title,
                "summary": " ".join(description.split()),
                "published": item.findtext("pubDate", default="").strip(),
                "updated": item.findtext("pubDate", default="").strip(),
                "authors": [name.strip() for name in item.findtext("dc:creator", default="", namespaces=RSS_NAMESPACE).split(",") if name.strip()],
                "categories": [category.text.strip() for category in item.findall("category") if category.text],
                "pdf_url": "",
                "link": item.findtext("link", default="").strip(),
                "source_feed": source_feed,
            }
        )
    return items


def _rss_matches(record: dict[str, Any], query_groups: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    haystack = " ".join(
        [
            record.get("title", ""),
            record.get("summary", ""),
            " ".join(record.get("categories", [])),
        ]
    ).lower()
    matched_queries: list[str] = []
    for group in query_groups:
        token_hits = sum(1 for token in group["tokens"] if token in haystack)
        phrase_hit = group["query"].lower() in haystack
        if phrase_hit or token_hits >= group["required_hits"]:
            matched_queries.append(group["query"])
    return bool(matched_queries), matched_queries


def _fetch_rss_candidates(categories: list[str], timeout: float) -> list[dict[str, Any]]:
    headers = {"User-Agent": "ai-agent-mini/0.1 (arxiv scrape prototype)"}
    candidates: list[dict[str, Any]] = []
    for category in categories:
        response = httpx.get(f"{ARXIV_RSS_URL}/{category}", headers=headers, timeout=timeout)
        response.raise_for_status()
        candidates.extend(_parse_rss_feed(response.text, category))
    return candidates


def invoke(request: StandardRequest) -> dict:
    target_config = request.payload.get("target_config", {})
    targets = target_config.get("targets", ["arxiv", "company_reports", "earnings_reports"])
    raw_dir, processed_dir, vectordb_dir = _arxiv_directories()

    if "arxiv" not in targets:
        return {
            "score": {"ingestion_jobs": len(targets), "paper_count": 0},
            "output": {
                "schedule": "every 4 months",
                "targets": targets,
                "last_run": date.today().isoformat(),
                "stored_in": "data/vectordb",
                "note": "arxiv target not requested",
            },
        }

    queries = target_config.get("queries", DEFAULT_ARXIV_QUERIES)
    max_results_per_query = int(target_config.get("max_results_per_query", 3))
    timeout = float(target_config.get("timeout_seconds", 30.0))
    rss_categories = target_config.get("rss_categories", DEFAULT_RSS_CATEGORIES)
    run_id = _utc_timestamp()

    raw_results: list[dict[str, Any]] = []
    unique_by_id: dict[str, dict[str, Any]] = {}
    source_mode = "arxiv_api"
    fetch_errors: list[str] = []
    try:
        for query in queries:
            papers = _fetch_arxiv_query(query, max_results_per_query, timeout)
            raw_results.append({"query": query, "papers": papers, "source_mode": "api"})
            for paper in papers:
                unique_by_id[paper["id"]] = paper
    except (httpx.HTTPError, RuntimeError, ElementTree.ParseError) as exc:
        source_mode = "arxiv_rss_fallback"
        fetch_errors.append(str(exc))
        query_groups = _query_groups(queries)
        rss_candidates = _fetch_rss_candidates(rss_categories, timeout)
        matched: list[dict[str, Any]] = []
        for record in rss_candidates:
            is_match, matched_queries = _rss_matches(record, query_groups)
            if is_match:
                enriched = dict(record)
                enriched["matched_queries"] = matched_queries
                matched.append(enriched)
        raw_results.append(
            {
                "rss_categories": rss_categories,
                "query_groups": query_groups,
                "candidate_count": len(rss_candidates),
                "match_count": len(matched),
                "papers": matched,
                "source_mode": "rss",
            }
        )
        for record in matched:
            record["query"] = "rss_keyword_match"
            unique_by_id[record["id"] or record.get("link", record["title"])] = record

    normalized_records = sorted(unique_by_id.values(), key=lambda item: item.get("published", ""), reverse=True)
    raw_path = raw_dir / f"arxiv_run_{run_id}.json"
    processed_path = processed_dir / f"arxiv_run_{run_id}.jsonl"
    manifest_path = vectordb_dir / "arxiv_manifest.json"

    _write_json(
        raw_path,
        {
            "run_id": run_id,
            "source": "arxiv",
            "query_count": len(queries),
            "source_mode": source_mode,
            "fetch_errors": fetch_errors,
            "results": raw_results,
        },
    )
    _write_jsonl(processed_path, normalized_records)

    manifest_payload = {
        "last_run_id": run_id,
        "source": "arxiv",
        "document_count": len(normalized_records),
        "raw_path": str(raw_path),
        "processed_path": str(processed_path),
        "ingestion_mode": "file-manifest-placeholder",
        "source_mode": source_mode,
    }
    _write_json(manifest_path, manifest_payload)

    return {
        "score": {
            "ingestion_jobs": len(targets),
            "paper_count": len(normalized_records),
            "query_count": len(queries),
        },
        "output": {
            "schedule": "every 4 months",
            "targets": targets,
            "queries": queries,
            "source_mode": source_mode,
            "fetch_errors": fetch_errors,
            "last_run": date.today().isoformat(),
            "stored_in": "data/vectordb",
            "raw_output_path": str(raw_path),
            "processed_output_path": str(processed_path),
            "vectordb_manifest_path": str(manifest_path),
            "papers": normalized_records,
        },
    }


def evaluate(request: StandardRequest) -> dict:
    targets = request.payload.get("targets", [])
    paper_count = int(request.payload.get("paper_count", 0))
    quality_score = 0.8 if targets and paper_count > 0 else 0.0
    return {
        "score": {"quality_score": quality_score},
        "output": {"parse_quality_ok": bool(targets) and paper_count > 0},
    }

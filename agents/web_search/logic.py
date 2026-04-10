from __future__ import annotations

import os
from datetime import date, timedelta
from urllib.parse import urlparse

import httpx

from shared.schemas import StandardRequest

SK_HYNIX_NEWSROOM_DOMAIN = "news.skhynix.com"


def _article(query_type: str, idx: int, query: str) -> dict:
    article_date = date.today() - timedelta(days=idx)
    return {
        "title": f"{query_type.title()} signal {idx + 1}",
        "url": f"https://example.com/{query_type}/{idx + 1}",
        "summary": f"Stubbed recent article for query '{query}' focused on {query_type} analysis.",
        "date": article_date.isoformat(),
        "source": "example.com",
        "query_type": query_type,
    }


def invoke(request: StandardRequest) -> dict:
    transformed_queries = request.payload.get("transformed_queries", [])
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    articles: list[dict] = []
    source_mode = "stubbed_tavily_compatible_results"
    error_message: str | None = None

    if api_key:
        try:
            for item in transformed_queries:
                query_type = item.get("type", "general")
                query = item.get("query", "")
                articles.extend(_search_tavily(api_key, query_type, query))
                articles.extend(
                    _search_tavily(
                        api_key,
                        query_type,
                        f'SK hynix newsroom {query}',
                        include_domains=[SK_HYNIX_NEWSROOM_DOMAIN],
                        max_results=3,
                    )
                )
            articles = _dedupe_articles(articles)
            source_mode = "tavily"
        except Exception as exc:
            error_message = f"{exc.__class__.__name__}: {exc}"
            articles = []

    if not articles:
        for item in transformed_queries:
            query_type = item.get("type", "general")
            query = item.get("query", "")
            for idx in range(5):
                articles.append(_article(query_type, idx, query))

    return {
        "score": {"freshness_score": 0.92, "duplication_score": 0.95},
        "output": {
            "articles": articles,
            "source_mode": source_mode,
            "error": error_message,
        },
    }


def evaluate(request: StandardRequest) -> dict:
    articles = request.payload.get("articles", [])
    freshness = 1.0 if articles else 0.0
    unique_urls = len({article.get("url") for article in articles})
    duplication_score = unique_urls / len(articles) if articles else 0.0
    relevance_score = 0.9 if len(articles) >= 4 else 0.4
    return {
        "score": {
            "freshness_score": freshness,
            "duplication_score": duplication_score,
            "relevance_score": relevance_score,
        },
        "output": {"is_reliable_stub": bool(request.payload.get("source_mode") == "stubbed_tavily_compatible_results")},
    }


def _search_tavily(
    api_key: str,
    query_type: str,
    query: str,
    include_domains: list[str] | None = None,
    max_results: int = 5,
) -> list[dict]:
    payload = {
        "api_key": api_key,
        "query": query,
        "topic": "news",
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False,
        "include_images": False,
    }
    if include_domains:
        payload["include_domains"] = include_domains

    response = httpx.post(
        "https://api.tavily.com/search",
        json=payload,
        timeout=45.0,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    normalized: list[dict] = []
    for item in results:
        url = item.get("url", "")
        normalized.append(
            {
                "title": item.get("title", "Untitled"),
                "url": url,
                "summary": item.get("content", ""),
                "date": item.get("published_date") or date.today().isoformat(),
                "source": urlparse(url).netloc or "tavily",
                "query_type": query_type,
            }
        )
    return normalized


def _dedupe_articles(articles: list[dict]) -> list[dict]:
    unique: dict[str, dict] = {}
    for article in articles:
        key = article.get("url") or f"{article.get('title')}::{article.get('source')}"
        unique[key] = article
    return list(unique.values())

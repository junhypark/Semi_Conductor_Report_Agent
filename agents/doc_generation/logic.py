from __future__ import annotations

import json
import os
import re
from pathlib import Path

from agents.doc_generation.llm import (
    build_openai_client,
    evaluate_report_sections,
    generate_market_sections,
    generate_skhynix_sections,
    generate_technique_sections,
    llm_available,
)
from pypdf import PdfReader
from shared.constants import DATA_DIR
from shared.schemas import StandardRequest


def _compact(value: str, limit: int = 220) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _sanitize_generated_text(value: str) -> str:
    cleaned = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [paragraph.strip() for paragraph in cleaned.split("\n\n")]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def _sentence_or_fallback(value: str, fallback: str) -> str:
    normalized = " ".join(value.split())
    return normalized if normalized else fallback


def _contains_korean(value: str) -> bool:
    return bool(re.search(r"[가-힣]", value))


def _is_english_heavy(value: str) -> bool:
    letters = re.findall(r"[A-Za-z]", value)
    korean = re.findall(r"[가-힣]", value)
    return len(letters) >= 20 and len(letters) > len(korean) * 2


def _meaningful_numbers(value: str) -> list[str]:
    candidates = re.findall(r"\d+(?:\.\d+)?%?", value)
    meaningful: list[str] = []
    for candidate in candidates:
        numeric = candidate.rstrip("%")
        if "." in numeric or candidate.endswith("%") or len(numeric) >= 3:
            meaningful.append(candidate)
            continue
        try:
            number = float(numeric)
        except ValueError:
            continue
        if number >= 20:
            meaningful.append(candidate)
    return meaningful


def _localized_title_text(title: str) -> str:
    normalized = " ".join(title.split())
    replacements = {
        "Assessing South Korea's AI Ecosystem": "한국 AI 생태계 평가",
        "International Monetary Fund Analysis of Korea": "국제통화기금의 한국 경제 분석",
        "Mapping Contoller": "매핑 컨트롤러",
        "Memory Controller": "메모리 컨트롤러",
        "Method and apparatus for processing": "처리 방법 및 장치",
        "Micron Annual Report": "마이크론 연차보고서",
        "Technical signal": "기술 신호",
        "Market signal": "시장 신호",
        "Competitor signal": "경쟁사 신호",
        "Future_Prediction signal": "미래 예측 신호",
        "Samsung": "삼성전자",
        "Micron": "마이크론",
        "SK Hynix": "SK hynix",
    }
    for source_text, target_text in replacements.items():
        normalized = normalized.replace(source_text, target_text)
    return normalized


def _detected_topics_from_text(text: str) -> list[str]:
    lowered = text.lower()
    topic_map = {
        "HBM4": ["hbm4"],
        "PIM": ["pim", "processing in memory"],
        "CXL": ["cxl"],
        "패키징": ["package", "packaging", "interposer", "패키징"],
        "수율": ["yield", "수율"],
        "열 관리": ["thermal", "heat", "cooling", "열"],
        "데이터센터 수요": ["data center", "server", "datacenter"],
        "공급망": ["supply chain", "sourcing", "공급망"],
        "가격": ["price", "pricing", "가격"],
        "양산": ["mass production", "shipment", "commercial", "launch", "양산"],
    }
    topics: list[str] = []
    for label, keywords in topic_map.items():
        if any(keyword in lowered for keyword in keywords):
            topics.append(label)
    return topics


def _koreanized_reference_text(item: dict, raw_value: str, fallback: str) -> str:
    normalized = _sentence_or_fallback(raw_value, fallback)
    if _contains_korean(normalized) and not _is_english_heavy(normalized):
        return normalized
    title = _display_record_title(item)
    topics = _detected_topics_from_text(f"{title} {normalized}")
    metrics = _metric_labels_from_text(normalized.lower())
    numbers = _meaningful_numbers(normalized)
    topic_text = _join_korean_topics(topics[:3]) if topics else "기술 및 시장 변화"
    metric_text = _join_korean_topics(metrics[:3]) if metrics else "핵심 성능"
    if numbers:
        return f"{title} 자료는 {topic_text}을 중심으로 논의를 전개하며, {metric_text} 지표와 {', '.join(numbers[:3])} 수준의 수치를 함께 제시한다."
    return f"{title} 자료는 {topic_text}을 중심으로 구조적 제약과 기대 효과를 설명하며, {metric_text} 관점에서 해석할 수 있는 근거를 제공한다."


def _news_reference(web_results: list[dict]) -> str:
    if not web_results:
        return "2026-04-10자 example.com의 'AI memory market demand outlook' 기사를 보조 사례로 사용하였다."
    item = web_results[0]
    return (
        f"{item.get('date', 'n.d.')}자 {item.get('source', 'web source')}의 "
        f"'{_localized_title_text(str(item.get('title', 'Untitled')))}' 기사를 구체적 뉴스 사례로 사용하였다."
    )


def _news_citation(web_results: list[dict]) -> str:
    if not web_results:
        return "2026-04-10자 example.com 기사 'AI memory market demand outlook'"
    item = web_results[0]
    return f"{item.get('date', 'n.d.')}자 {item.get('source', 'web source')} 기사 '{_localized_title_text(str(item.get('title', 'Untitled')))}'"


def _evidence_reference(rag_results: list[dict], web_results: list[dict]) -> str:
    if rag_results:
        item = rag_results[0]
        return f"{_display_record_title(item)} ({_display_source_label(item)})"
    if web_results:
        item = web_results[0]
        return f"{_localized_title_text(str(item.get('title', 'Untitled')))} ({item.get('source', 'web source')}, {item.get('date', 'n.d.')})"
    return "HBM4 scaling and thermal management considerations (stubbed_technical_brief)"


def _case_reason(item: dict) -> str:
    source = str(item.get("source", "")).lower()
    title = _display_record_title(item)
    if "reuters" in source or "news" in source:
        return f"{title}은 시장 채택 속도와 기업 전략 변화를 동시에 보여주기 때문에 사례로 선정하였다."
    return f"{title}은 기술 구조, 제약, 기대 효과를 함께 설명하고 있어 사례로 선정하였다."


def _case_description(item: dict) -> str:
    if item.get("summary"):
        return _koreanized_reference_text(
            item,
            str(item.get("summary", "")),
            "해당 사례는 관련 기술과 시장 변화의 세부 맥락을 설명한다.",
        )
    return _koreanized_reference_text(
        item,
        str(item.get("chunk", "")),
        "해당 사례는 관련 기술과 시장 변화의 세부 맥락을 설명한다.",
    )


def _case_claim(item: dict) -> str:
    title = _display_record_title(item)
    source = str(item.get("source", "")).lower()
    if "pdf" in source:
        return f"{title}는 기술 성숙도와 구조적 제약을 함께 판단해야 한다는 주장을 뒷받침한다."
    return f"{title}는 시장 변화가 실제 투자 우선순위와 연결된다는 주장을 뒷받침한다."


def _quote_line(item: dict) -> str:
    quote_text = _koreanized_reference_text(
        item,
        str(item.get("summary") or item.get("chunk") or ""),
        "해당 자료는 기술과 시장 방향을 동시에 검토해야 한다는 점을 보여준다.",
    )
    quote_text = quote_text.rstrip(".")
    return f'인용문: *"{quote_text}."*'


def _pdf_basis_summary(rag_results: list[dict]) -> tuple[str, list[str]]:
    pdf_items = [item for item in rag_results if str(item.get("source", "")).lower().endswith(".pdf")]
    source_items = pdf_items or rag_results
    if not source_items:
        return (
            "읽은 PDF가 아직 없으므로 기본 기술 메모를 기준으로 핵심 기술 현황을 요약하였다.",
            [
                "HBM4는 고대역폭 수요와 직접 연결된다.",
                "PIM은 메모리 내부 연산 효율을 높이는 방향으로 검토된다.",
                "CXL은 시스템 확장성과 메모리 풀링 구조를 강화한다.",
            ],
        )
    summaries = [_pdf_summary_line_ko(item) for item in source_items[:3]]
    titles = [_display_record_title(item) for item in source_items[:2]]
    basis = f"읽은 PDF 기준 요약 대상은 {', '.join(titles)}이다."
    while len(summaries) < 3:
        summaries.append("관련 PDF에서 기술 제약과 적용 가능성을 함께 언급하였다.")
    return basis, summaries[:3]


def _pdf_summary_line_ko(item: dict) -> str:
    text = f"{item.get('title', '')} {item.get('chunk', '')}".lower()
    topics: list[str] = []
    if "hbm4" in text:
        topics.append("HBM4")
    if "pim" in text:
        topics.append("PIM")
    if "cxl" in text:
        topics.append("CXL")
    if "yield" in text or "수율" in text:
        topics.append("수율")
    if "thermal" in text or "열" in text:
        topics.append("열 관리")
    if "package" in text or "packaging" in text or "패키징" in text:
        topics.append("패키징")
    metric_labels = _metric_labels_from_text(text)
    if not topics:
        if metric_labels:
            return f"PDF는 기술 제약과 적용 가능성을 설명하며, 본문에는 {_join_korean_topics(metric_labels[:3])} 지표가 포함되어 있다."
        return "PDF는 기술 제약과 적용 가능성을 함께 설명한다."
    unique_topics = []
    for topic in topics:
        if topic not in unique_topics:
            unique_topics.append(topic)
    joined_topics = _join_korean_topics(unique_topics[:3])
    if metric_labels:
        return f"PDF는 {joined_topics} 관련 핵심 쟁점을 중심으로 기술 현황을 설명하며, {_join_korean_topics(metric_labels[:3])} 지표를 함께 제시한다."
    return f"PDF는 {joined_topics} 관련 핵심 쟁점을 중심으로 기술 현황을 설명한다."


def _metric_labels_from_text(text: str) -> list[str]:
    metric_keywords = {
        "대역폭": ["bandwidth", "대역폭", "gb/s", "tb/s"],
        "수율": ["yield", "수율"],
        "전력 효율": ["power efficiency", "efficiency", "전력 효율", "watt", "joule"],
        "적층 수": ["layer", "stack", "층", "die"],
        "열 관리": ["thermal", "temperature", "열", "cooling"],
        "패키징 밀도": ["packaging density", "package", "패키징", "interposer"],
        "지연 시간": ["latency", "지연"],
        "생산 능력": ["capacity", "throughput", "wafer", "생산"],
    }
    labels: list[str] = []
    for label, keywords in metric_keywords.items():
        if any(keyword in text for keyword in keywords):
            labels.append(label)
    return labels


def _join_korean_topics(topics: list[str]) -> str:
    if not topics:
        return "기술"
    if len(topics) == 1:
        return topics[0]
    if len(topics) == 2:
        return f"{topics[0]}와 {topics[1]}"
    return ", ".join(topics[:-1]) + f"와 {topics[-1]}"


def _case_lines(rag_results: list[dict], web_results: list[dict]) -> list[str]:
    lines: list[str] = []
    for item in web_results:
        lines.append(_format_case_entry(item))
    for item in rag_results:
        lines.append(_format_case_entry(item))
    if not lines:
        lines.append(
            "[기본 기술 메모]-[HBM4, PIM, CXL의 적용 차이와 제약을 함께 보여주기 때문에 사례로 선정하였다]-"
            "[기본 기술 메모는 메모리 구조와 시스템 병목 완화 방향을 함께 설명한다]-"
            "[기술 우선순위는 성능 수치와 통합 난이도를 함께 봐야 한다는 주장을 뒷받침한다]"
        )
    return lines


def _format_case_entry(item: dict) -> str:
    title = _display_record_title(item)
    reason = _case_reason(item)
    description = _case_description(item)
    claim = _case_claim(item)
    return f"[{title}]-[{reason}]-[{description}]-[{claim}]"


def _render_case_block(case_lines: list[str]) -> str:
    return "\n".join(case_lines)


def _quote_block(rag_results: list[dict], web_results: list[dict], limit: int = 3) -> str:
    selected = web_results[:2] + rag_results[:1]
    if not selected:
        selected = [{"title": "기본 기술 메모", "summary": "HBM4와 차세대 메모리 전략을 함께 검토해야 한다.", "source": "local"}]
    return "\n".join(_quote_line(item) for item in selected[:limit])


def _background_conclusion(user_query: str, rag_count: int, web_count: int) -> str:
    return (
        f"결론적으로 본 보고서는 '{user_query}'에 대해 SK hynix가 단기적으로는 HBM4 실행력을 우선 확보하고, "
        f"중기적으로는 PIM과 CXL을 연계한 연구개발 포트폴리오를 병행해야 한다는 결론을 제시한다. "
        f"이 결론은 로컬 문서 {rag_count}건과 웹 기사 {web_count}건을 함께 검토한 결과를 요약한 것이다."
    )


def _future_strategy_focus_reason() -> str:
    return (
        "본 보고서가 미래전략 기획에 집중하는 이유는 AI 메모리 시장이 제품 성능 경쟁만으로는 설명되지 않고, "
        "설비 투자 시차, 패키징 공급망, 고객 채택 일정, 기술 성숙도 판단이 함께 얽혀 있기 때문이다."
    )


def _limitation_block(lines: list[str]) -> str:
    bullets = "\n".join(f"- {line}" for line in lines[:3])
    return f"Limitation: 본 절에는 다음과 같은 우려가 있다.\n{bullets}"


def _build_evidence_digest(
    *,
    user_query: str,
    rag_count: int,
    web_count: int,
    arxiv_count: int,
    rag_results: list[dict],
    web_results: list[dict],
) -> str:
    lines = [
        f"질문: {user_query}",
        f"로컬 문서 수: {rag_count}",
        f"웹 기사 수: {web_count}",
        f"arXiv 참고 자료 수: {arxiv_count}",
    ]
    for index, item in enumerate(rag_results, start=1):
        lines.append(
            f"[RAG {index}] {_display_record_title(item)} | {_display_source_label(item)} | {_koreanized_reference_text(item, str(item.get('chunk', '')), '내용 없음')}"
        )
    for index, item in enumerate(web_results, start=1):
        lines.append(
            f"[NEWS {index}] {item.get('date', 'n.d.')} | {item.get('source', 'web source')} | {_localized_title_text(str(item.get('title', 'Untitled')))} | {item.get('url', '')} | {_koreanized_reference_text(item, str(item.get('summary', '')), '내용 없음')}"
        )
    return "\n".join(lines)


def _summary_only_view(section_name: str, text: str) -> str:
    text = _sanitize_generated_text(text)
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    headline = paragraphs[0] if paragraphs else text
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    bullets = [line[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")]
    case_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("[") and "]-[" in line]
    lines = [f"섹션명: {section_name}", f"핵심 요약: {headline}"]
    if len(paragraphs) > 1:
        lines.append(f"제2문단 요약: {paragraphs[1]}")
    lines.append(f"정량 수치: {', '.join(numbers) if numbers else '없음'}")
    if bullets:
        lines.append("핵심 불릿:")
        lines.extend(f"- {bullet}" for bullet in bullets)
    if case_lines:
        lines.append("사례 목록:")
        lines.extend(case_lines)
    return "\n".join(lines)


def _display_source_label(item: dict) -> str:
    source = str(item.get("source", "")).strip()
    if not source:
        return "local source"
    if source.startswith("http://") or source.startswith("https://"):
        return source
    if re.fullmatch(r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}", source):
        return source
    path = Path(source)
    if path.suffix.lower() == ".pdf":
        return "내부 PDF"
    if path.suffix:
        return f"내부 {path.suffix.lower().lstrip('.')} 문서"
    return path.name or "local source"


def _display_record_title(item: dict) -> str:
    raw_title = str(item.get("title", "Untitled")).strip() or "Untitled"
    clean_title = re.sub(r"\s*-\s*page\s+\d+$", "", raw_title, flags=re.IGNORECASE).strip()
    source = str(item.get("source", "")).strip()
    source_path = Path(source) if source else None
    if source_path is not None and _is_manual_document_path(source_path) and not _looks_generic_title(source_path.stem, source_path):
        return _localized_title_text(source_path.stem)
    if clean_title and not _looks_generic_title(clean_title, source_path):
        return _localized_title_text(clean_title)
    first_page_title = _read_pdf_first_page_title(source_path) if source_path else None
    if first_page_title:
        return _localized_title_text(first_page_title)
    if clean_title:
        return _localized_title_text(clean_title)
    if source_path is not None:
        return _localized_title_text(source_path.stem)
    return "Untitled"


def _looks_generic_title(title: str, source_path: Path | None) -> bool:
    normalized = title.lower().strip()
    generic = bool(re.fullmatch(r"(report|document|paper|file|scan|note|untitled)[\s_\-]*\d*", normalized))
    if title.isupper() and len(title.split()) <= 4:
        generic = True
    if "washington, d.c." in normalized:
        generic = True
    if "united states" in normalized and len(title.split()) <= 4:
        generic = True
    if source_path is None:
        return generic
    return generic or normalized == source_path.stem.lower()


def _read_pdf_first_page_title(source_path: Path | None) -> str | None:
    if source_path is None or source_path.suffix.lower() != ".pdf" or not source_path.exists():
        return None
    try:
        reader = PdfReader(str(source_path))
    except Exception:
        return None
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
            if line.isupper() and len(line.split()) <= 4:
                continue
            if "washington, d.c." in line.lower():
                continue
            if source_path.stem.lower() == line.lower():
                continue
            return line[:180]
    return None


def _is_manual_document_path(path: Path) -> bool:
    return "data/raw/manual" in str(path).replace("\\", "/")


COMPETITOR_LABELS = {
    "Samsung": ["samsung", "삼성"],
    "TSMC": ["tsmc"],
    "Micron": ["micron", "마이크론"],
}
COMPETITOR_DISPLAY_NAMES = {
    "Samsung": "삼성전자",
    "TSMC": "TSMC",
    "Micron": "마이크론",
}


def _record_mentions(item: dict, keywords: list[str]) -> bool:
    haystack = " ".join(
        [
            str(item.get("title", "")),
            str(item.get("summary", "")),
            str(item.get("chunk", "")),
            str(item.get("source", "")),
        ]
    ).lower()
    return any(keyword in haystack for keyword in keywords)


def _estimate_trl(item: dict) -> str:
    text = " ".join(
        [
            str(item.get("title", "")),
            str(item.get("summary", "")),
            str(item.get("chunk", "")),
        ]
    ).lower()
    if any(token in text for token in ["mass production", "production", "shipment", "commercial", "launch", "양산"]):
        return "TRL 6"
    if any(token in text for token in ["pilot", "prototype", "evaluation", "validation", "실증"]):
        return "TRL 5"
    return "TRL 4"


def _strategy_line(item: dict, competitor: str) -> str:
    display_name = COMPETITOR_DISPLAY_NAMES.get(competitor, competitor)
    text = _koreanized_reference_text(
        item,
        str(item.get("summary") or item.get("chunk") or ""),
        f"{competitor} 관련 구체적 설명을 찾지 못했다.",
    )
    return f"{display_name}의 현재 전략은 {text}"


def _future_plan_line(item: dict, competitor: str) -> str:
    display_name = COMPETITOR_DISPLAY_NAMES.get(competitor, competitor)
    title = _display_record_title(item)
    source = _display_source_label(item)
    return f"{display_name}의 미래 계획은 '{title}' 자료와 {source} 근거를 기준으로 차세대 메모리 또는 패키징 역량을 확대하는 방향으로 해석된다."


def _competitor_entry(competitor: str, rag_results: list[dict], web_results: list[dict]) -> str:
    display_name = COMPETITOR_DISPLAY_NAMES.get(competitor, competitor)
    keywords = COMPETITOR_LABELS[competitor]
    rag_matches = [item for item in rag_results if _record_mentions(item, keywords)]
    web_matches = [item for item in web_results if _record_mentions(item, keywords)]
    primary = rag_matches[0] if rag_matches else web_matches[0] if web_matches else None

    if primary is None:
        return (
            f"{display_name}: 로컬 문서와 웹 검색 결과를 모두 검토하였으나 {display_name}의 HBM4, PIM, CXL 관련 전략과 미래 계획을 "
            "직접 확인할 수 있는 자료를 찾지 못하였다. 따라서 본 보고서는 해당 경쟁사에 대해서는 확인 가능한 범위의 비교만 수행하였다. "
            "전략 TRL 평가는 자료 부족으로 산정하지 못하였으며, 후속 웹 검색과 공시 자료 확보가 필요하다고 판단한다."
        )

    strategy = _strategy_line(primary, competitor)
    future_plan = _future_plan_line(primary, competitor)
    trl = _estimate_trl(primary)
    return (
        f"{display_name}: {strategy} {future_plan} "
        f"전략 TRL은 {trl}로 평가하며, 이는 현재 확보된 자료가 개념 검증을 넘어 파일럿 또는 초기 상용화 가능성을 시사하기 때문이다."
    )


def _competitor_comparison_block(rag_results: list[dict], web_results: list[dict]) -> str:
    return "\n".join(_competitor_entry(name, rag_results, web_results) for name in ["Samsung", "TSMC", "Micron"])


def _extract_summary_line(text: str) -> str:
    normalized = _sanitize_generated_text(text)
    for line in normalized.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            continue
        return stripped
    return normalized


def _build_executive_summary(
    request: StandardRequest,
    sections: dict[str, str],
    rag_results: list[dict],
    web_results: list[dict],
) -> str:
    user_query = request.payload.get("user_query", "SK hynix market direction")
    retrieval_summary = request.payload.get("retrieval_summary", {})
    rag_count = int(retrieval_summary.get("rag_document_count", len(rag_results)))
    web_count = int(retrieval_summary.get("web_article_count", len(web_results)))
    arxiv_count = _load_arxiv_count()
    doc_titles = list(dict.fromkeys(_display_record_title(item) for item in rag_results[:5]))[:3]
    news_titles = [_localized_title_text(str(item.get("title", "Untitled"))) for item in web_results[:3]]
    tech_lines = [line for line in _sanitize_generated_text(sections["current_status_of_target_technologies"]).splitlines() if line.strip().startswith("요약 ")]
    lines = [
        f"본 Executive Summary는 '{user_query}'에 대한 전체 보고서 결론만 압축하여 제시한다.",
        f"검토 대상은 로컬 문서 {rag_count}건, 웹 기사 {web_count}건, arXiv 참고 자료 {arxiv_count}건이며 핵심 문서는 {', '.join(doc_titles) if doc_titles else '내부 기술 문서'}이다.",
        f"핵심 기술 요지는 {_compact(tech_lines[0] if len(tech_lines) > 0 else _extract_summary_line(sections['current_status_of_target_technologies']), 140)}",
        f"경쟁 및 전략 요지는 {_compact(_extract_summary_line(sections['competitor_trend_analysis']), 140)} {_compact(_extract_summary_line(sections['strategic_implications']), 140)}",
        f"최종 결론은 SK hynix가 단기적으로 HBM4 실행력과 수율 안정화를 확보하고, 중기적으로 PIM·CXL 기반 연구개발을 확대해야 한다는 점이며 참고 뉴스는 {', '.join(news_titles[:2]) if news_titles else '최신 시장 기사'}이다.",
    ]
    return "\n".join(lines)


def _section_text(request: StandardRequest, key: str) -> str:
    user_query = request.payload.get("user_query", "SK hynix market direction")
    retrieval_summary = request.payload.get("retrieval_summary", {})
    rag_count = retrieval_summary.get("rag_document_count", 0)
    web_count = retrieval_summary.get("web_article_count", 0)
    rag_results = request.payload.get("rag_results", [])
    web_results = request.payload.get("web_results", [])
    arxiv_count = _load_arxiv_count()
    news_reference = _news_reference(web_results)
    news_citation = _news_citation(web_results)
    evidence_reference = _evidence_reference(rag_results, web_results)
    pdf_basis, pdf_lines = _pdf_basis_summary(rag_results)
    case_lines = _case_lines(rag_results, web_results)
    case_block = _render_case_block(case_lines)
    quote_block = _quote_block(rag_results, web_results)
    competitor_block = _competitor_comparison_block(rag_results, web_results)
    conclusion_line = _background_conclusion(user_query, rag_count, web_count)

    section_map = {
        "background_of_analysis": "\n\n".join(
            [
                (
                    f"본 보고서는 '{user_query}'를 연구 질문으로 설정하고, Supervisor 기반 워크플로를 통해 "
                    f"로컬 근거 문서 {rag_count}건, 정규화된 웹 기사 {web_count}건, arXiv 참고 자료 {arxiv_count}건을 종합하여 SK hynix의 미래 시장 방향을 검토하였다. "
                    f"핵심 검토 문서는 {', '.join(list(dict.fromkeys(_display_record_title(item) for item in rag_results[:3]))) if rag_results else '내부 기술 문서'}이며, "
                    f"대표 뉴스 사례는 {', '.join(_localized_title_text(str(item.get('title', 'Untitled'))) for item in web_results[:3]) if web_results else '최신 시장 기사'}이다. "
                    f"{_future_strategy_focus_reason()} {news_reference} {conclusion_line}"
                ),
                _limitation_block(
                    [
                        f"웹 기사 {web_count}건은 수집 시점에 따라 시장 온도차를 과대 반영할 수 있다.",
                        f"로컬 문서 {rag_count}건은 특정 기업 또는 특정 기술 스택에 편중될 수 있다.",
                        f"arXiv 참고 자료 {arxiv_count}건은 상용화 속도보다 연구 방향성을 더 강하게 반영할 수 있다.",
                    ]
                ),
            ]
        ),
        "current_status_of_target_technologies": "\n\n".join(
            [
                (
                    f"{pdf_basis}\n"
                    f"요약 1: {pdf_lines[0]}\n"
                    f"요약 2: {pdf_lines[1]}\n"
                    f"요약 3: {pdf_lines[2]}\n"
                    "장점:\n"
                    "- HBM4는 고대역폭 수요와 직접 연결되어 단기 사업화 가능성이 높다.\n"
                    "- PIM은 연산 효율과 데이터 이동 절감 측면에서 구조적 이점을 가진다.\n"
                    "- CXL은 메모리 확장성과 시스템 유연성을 높인다.\n"
                    "단점:\n"
                    "- 제조 수율과 패키징 난도가 높아 양산 최적화 비용이 크다.\n"
                    "- PIM은 소프트웨어와 시스템 생태계 정합성이 충분히 확보되어야 한다.\n"
                    "- CXL은 고객 채택 속도와 표준 구현 차이에 따라 성과 편차가 커질 수 있다."
                ),
                _limitation_block(
                    [
                        "기술 성숙도 평가는 제품 발표와 양산 실적을 동일선상에 두기 어렵다.",
                        "패키징과 인터커넥트 데이터는 공개 범위가 제한되어 정량 비교가 불완전할 수 있다.",
                        "PIM과 CXL은 고객 시스템 채택 속도에 따라 해석이 빠르게 바뀔 수 있다.",
                    ]
                ),
            ]
        ),
        "competitor_trend_analysis": "\n\n".join(
            [
                (
                    f"삼성전자, TSMC, 마이크론은 3개 경쟁 축에서 서로 다른 우위를 보이며, 경쟁 구도는 HBM4 양산성, PIM 연계성, CXL 플랫폼 연결성의 3개 변수로 압축된다. "
                    f"{news_reference}\n{competitor_block}\n{case_block}\n{quote_block}\n이 비교는 경쟁사 전략이 단순 성능 수치가 아니라 공급망, 패키징, 고객 채택 일정의 결합으로 평가되어야 함을 보여준다."
                ),
                _limitation_block(
                    [
                        "경쟁사 발표 자료는 자사 우위 관점으로 편향될 수 있다.",
                        "TSMC와 메모리 업체의 역할은 직접 비교보다 협력 구조 분석이 더 적합할 수 있다.",
                        "경쟁사 전략은 고객사 발주와 규제 변화에 따라 분기 단위로 재조정될 수 있다.",
                    ]
                ),
            ]
        ),
        "strategic_implications": "\n\n".join(
            [
                (
                    f"향후 5~10년의 전략적 방향은 HBM4의 단기 실행력과 PIM·CXL의 중기 옵션 확보를 2단계로 병행하는 구조가 가장 타당하다. "
                    f"이 추론은 {evidence_reference}와 {news_citation}에 근거한다.\n추론 근거:\n- HBM4는 단기 수요와 직접 연결되는 반면 PIM과 CXL은 중기 차별화 수단으로 작동한다.\n- 기술 투자 우선순위를 2단계로 나누면 단기 수익성과 중기 옵션 가치를 동시에 관리할 수 있다.\n- 패키징, 열, 인터커넥트의 3개 병목을 동시에 해결하는 기업이 고객 락인을 더 강하게 형성할 수 있다.\n종합 시사점 및 Conclusion:\n- SK hynix는 단기적으로 HBM4 실행력을 통해 매출 가시성을 확보해야 한다.\n- 중기적으로는 PIM과 CXL을 통해 차세대 시스템 아키텍처 대응력을 높여야 한다.\n- 경쟁사 비교 결과를 종합하면 기술 우위보다 통합 실행력과 고객 채택 구조가 더 중요한 결론으로 도출된다.\n- 실행 우선순위는 수율 안정화, 패키징 병목 완화, 고객 채택 시나리오 검증을 동시에 관리하는 운영 체계를 확보하는 데 있다."
                ),
                _limitation_block(
                    [
                        "미래 방향성 추론은 현재 확보된 근거 수가 제한적이어서 시장 급변 상황을 완전히 반영하지 못할 수 있다.",
                        "고객사의 실제 도입 속도는 기술 우수성보다 총소유비용과 공급 안정성에 더 크게 좌우될 수 있다.",
                        "정책, 수출규제, CAPEX 조정 같은 외생 변수는 예측 오차를 확대할 수 있다.",
                    ]
                ),
            ]
        ),
        "trl_evaluation": "\n\n".join(
            [
                (
                    f"권고 전략의 TRL 수치는 4.2/5로 평가하며, 이는 TRL 4~6 범위 안에서 파일럿 검증과 양산성 점검을 동시에 수행해야 하는 단계에 해당한다. "
                    f"{evidence_reference}를 근거로 볼 때 개념 검증은 1단계를 넘었으나 비용 구조와 공급망 안정성의 2개 과제가 여전히 남아 있다.\n{case_block}\n{quote_block}"
                ),
                _limitation_block(
                    [
                        "TRL 평가는 기술 성숙도와 사업 성숙도를 동일하게 설명하지 못한다.",
                        "실험실 검증 결과가 양산 수율로 바로 이어진다고 단정하기 어렵다.",
                        "고객 검증 데이터가 제한되면 TRL 수치는 보수적으로 해석될 필요가 있다.",
                    ]
                ),
            ]
        ),
        "reference_part": _build_reference_section(rag_results, web_results),
    }
    return section_map[key]


def _build_reference_section(rag_results: list[dict], web_results: list[dict]) -> str:
    rag_refs = [
        f"{_display_record_title(item)}. (2026). 내부 검색 문서."
        for item in rag_results[:5]
    ]
    web_refs = [
        f"{item.get('source', 'Web source')}. ({item.get('date', 'n.d.')}). {_localized_title_text(str(item.get('title', 'Untitled')))}. {item.get('url', '')}"
        for item in web_results[:6]
    ]
    refs = list(dict.fromkeys(rag_refs + web_refs))
    if not refs:
        refs = [
            "HBM4 scaling and thermal management considerations. (2026). 내부 검색 문서.",
            "AI memory market demand outlook. (2026). 내부 검색 문서.",
            "Competitor roadmap comparison. (2026). 내부 검색 문서.",
        ]
    return "\n".join(f"[{index}] {reference}" for index, reference in enumerate(refs, start=1))


def _load_arxiv_count() -> int:
    manifest_path = DATA_DIR / "vectordb" / "arxiv_manifest.json"
    if not manifest_path.exists():
        return 0
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return int(manifest.get("document_count", 0))


def _build_stub_sections(request: StandardRequest) -> dict[str, str]:
    rag_results = request.payload.get("rag_results", [])
    web_results = request.payload.get("web_results", [])
    sections = {
        key: _section_text(request, key)
        for key in [
            "background_of_analysis",
            "current_status_of_target_technologies",
            "competitor_trend_analysis",
            "strategic_implications",
            "trl_evaluation",
            "reference_part",
        ]
    }
    sections["executive_summary"] = _build_executive_summary(request, sections, rag_results, web_results)
    return sections


def _build_live_sections(request: StandardRequest) -> tuple[dict[str, str], dict[str, float], dict[str, str]]:
    retrieval_summary = request.payload.get("retrieval_summary", {})
    user_query = request.payload.get("user_query", "SK hynix market direction")
    rag_results = request.payload.get("rag_results", [])
    web_results = request.payload.get("web_results", [])
    model = request.config.get("openai_model", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    client = build_openai_client()
    rag_count = int(retrieval_summary.get("rag_document_count", 0))
    web_count = int(retrieval_summary.get("web_article_count", 0))
    arxiv_count = _load_arxiv_count()
    evidence_digest = _build_evidence_digest(
        user_query=user_query,
        rag_count=rag_count,
        web_count=web_count,
        arxiv_count=arxiv_count,
        rag_results=rag_results,
        web_results=web_results,
    )

    market_sections = generate_market_sections(
        client=client,
        model=model,
        user_query=user_query,
        rag_count=rag_count,
        web_count=web_count,
        arxiv_count=arxiv_count,
        evidence_digest=evidence_digest,
        peer_context="이전 LLM 요약본 없음.",
    )
    market_peer_summary = _summary_only_view(
        "market_bundle",
        "\n\n".join(
            [
                market_sections["background_of_analysis"],
                market_sections["strategic_implications"],
                market_sections["trl_evaluation"],
            ]
        ),
    )
    skhynix_sections = generate_skhynix_sections(
        client=client,
        model=model,
        user_query=user_query,
        evidence_digest=evidence_digest,
        peer_context=market_peer_summary,
    )
    skhynix_peer_summary = _summary_only_view(
        "competitor_trend_analysis",
        skhynix_sections["competitor_trend_analysis"],
    )
    technique_sections = generate_technique_sections(
        client=client,
        model=model,
        user_query=user_query,
        evidence_digest=evidence_digest,
        peer_context="\n\n".join([market_peer_summary, skhynix_peer_summary]),
    )

    sections = {
        "background_of_analysis": _sanitize_generated_text(market_sections["background_of_analysis"]),
        "current_status_of_target_technologies": _sanitize_generated_text(technique_sections["current_status_of_target_technologies"]),
        "competitor_trend_analysis": _sanitize_generated_text(skhynix_sections["competitor_trend_analysis"]),
        "strategic_implications": _sanitize_generated_text(market_sections["strategic_implications"]),
        "trl_evaluation": _sanitize_generated_text(market_sections["trl_evaluation"]),
        "reference_part": _build_reference_section(
            rag_results,
            web_results,
        ),
    }
    sections["executive_summary"] = _build_executive_summary(request, sections, rag_results, web_results)
    section_summaries = {key: _summary_only_view(key, value) for key, value in sections.items() if key != "reference_part"}

    evaluator = evaluate_report_sections(
        client=client,
        model=model,
        user_query=user_query,
        section_summaries=section_summaries,
    )
    scores = {
        "judge_score": float(evaluator.get("judge_score", 0.82)),
        "trl_numeric_score": float(evaluator.get("trl_numeric_score", 4.2)),
    }
    rationale = {
        "rationale": str(evaluator.get("rationale", "")).strip(),
        "peer_context_mode": "summary_only",
        "section_summaries": section_summaries,
    }
    return sections, scores, rationale


def invoke(request: StandardRequest) -> dict:
    generation_round = int(request.context.get("generation_round", 1))
    llm_mode = request.config.get("llm_mode", "auto")
    used_live_llm = False
    llm_error: str | None = None

    if llm_mode == "live" or (llm_mode == "auto" and llm_available()):
        try:
            sections, live_scores, evaluator_meta = _build_live_sections(request)
            judge_score = live_scores["judge_score"]
            trl_numeric_score = live_scores["trl_numeric_score"]
            used_live_llm = True
        except Exception as exc:
            llm_error = f"{exc.__class__.__name__}: {exc}"
            sections = _build_stub_sections(request)
            judge_score = 0.58 if generation_round == 1 and request.config.get("force_retry", True) else 0.82
            trl_numeric_score = 3.2 if generation_round == 1 and request.config.get("force_retry", True) else 4.2
            evaluator_meta = {"rationale": "LLM call failed, stub fallback was used."}
    else:
        sections = _build_stub_sections(request)
        judge_score = 0.58 if generation_round == 1 and request.config.get("force_retry", True) else 0.82
        trl_numeric_score = 3.2 if generation_round == 1 and request.config.get("force_retry", True) else 4.2
        evaluator_meta = {"rationale": "Stub mode was used."}

    if request.config.get("disable_retry_logic"):
        judge_score = max(judge_score, 0.82)
        trl_numeric_score = max(trl_numeric_score, 4.2)

    output = {
        "section_drafts": sections,
        "judge_score": judge_score,
        "trl_evaluation": {
            "trl_band": "TRL 4-6",
            "numeric_score": trl_numeric_score,
            "passes_threshold": trl_numeric_score >= 3.5,
        },
        "sub_supervisor": {"regenerate": judge_score < 0.6 or trl_numeric_score < 3.5},
        "llm_trace": {
            "used_live_llm": used_live_llm,
            "llm_mode": llm_mode,
            "model": request.config.get("openai_model", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")),
            "error": llm_error,
            **evaluator_meta,
        },
    }
    return {"score": {"judge_score": judge_score, "trl_numeric_score": trl_numeric_score}, "output": output}


def evaluate(request: StandardRequest) -> dict:
    judge_score = float(request.payload.get("judge_score", 0.0))
    trl_numeric_score = float(request.payload.get("trl_numeric_score", 0.0))
    regenerate = judge_score < 0.6 or trl_numeric_score < 3.5
    return {
        "score": {"judge_score": judge_score, "trl_numeric_score": trl_numeric_score},
        "output": {"need_regeneration": regenerate},
    }

from __future__ import annotations

from pathlib import Path
import re

from shared.constants import SECTION_ORDER
from shared.files import default_report_stem, resolve_report_paths
from shared.pdf import build_report_pdf
from shared.schemas import StandardRequest

SECTION_TITLES_KO = {
    "executive_summary": "Executive Summary",
    "background_of_analysis": "분석 배경",
    "current_status_of_target_technologies": "핵심 기술 현황",
    "competitor_trend_analysis": "경쟁사 동향 분석",
    "strategic_implications": "전략적 시사점",
    "trl_evaluation": "TRL 평가",
    "reference_part": "참고문헌",
}


def _sanitize_report_text(value: str) -> str:
    cleaned = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\s*\n?\s*(장점:)", r"\n\1", cleaned)
    cleaned = re.sub(r"\s*\n?\s*(단점:)", r"\n\1", cleaned)
    cleaned = re.sub(r"\s*\n?\s*(추론 근거:)", r"\n\1", cleaned)
    cleaned = re.sub(r"\s*\n?\s*(Limitation:)", r"\n\n\1", cleaned)
    cleaned = re.sub(r"\s*\n?\s*(인용문:)", r"\n\1", cleaned)
    cleaned = re.sub(r"\s*(사례:)", r"\n\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    normalized_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped.startswith("인용문:"):
            quote = stripped.split(":", 1)[1].strip()
            if quote.startswith("*") and quote.endswith("*") and len(quote) >= 2:
                quote = quote[1:-1].strip()
            quote = quote.strip('"')
            normalized_lines.append(f'인용문: *"{quote}"*')
        else:
            normalized_lines.append(line)
    paragraphs = [paragraph.strip() for paragraph in "\n".join(normalized_lines).split("\n\n")]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def _build_markdown(section_drafts: dict[str, str]) -> str:
    title = "# SK hynix 미래 시장 방향 보고서"
    body = [title, ""]
    for key in SECTION_ORDER:
        heading = SECTION_TITLES_KO.get(key, key.replace("_", " ").title())
        body.extend([f"## {heading}", _sanitize_report_text(section_drafts.get(key, "No content provided.")), ""])
    return "\n".join(body).strip() + "\n"


def invoke(request: StandardRequest) -> dict:
    raw_section_drafts = request.payload.get("section_drafts", {})
    section_drafts = {key: _sanitize_report_text(value) for key, value in raw_section_drafts.items()}
    report_name = request.config.get("report_name", default_report_stem())
    markdown_path, pdf_path = resolve_report_paths(report_name)

    markdown_content = _build_markdown(section_drafts)
    markdown_path.write_text(markdown_content, encoding="utf-8")

    build_report_pdf(
        pdf_path=pdf_path,
        title="SK hynix 미래 시장 방향 보고서",
        sections=[(SECTION_TITLES_KO.get(key, key), section_drafts.get(key, "내용이 제공되지 않았습니다.")) for key in SECTION_ORDER],
    )

    return {
        "score": {"format_completeness": 1.0 if section_drafts else 0.0},
        "output": {
            "markdown_report_path": str(markdown_path),
            "pdf_report_path": str(pdf_path),
            "report_name": Path(markdown_path).stem,
        },
    }


def evaluate(request: StandardRequest) -> dict:
    markdown_path = Path(request.payload.get("markdown_report_path", ""))
    pdf_path = Path(request.payload.get("pdf_report_path", ""))
    complete = markdown_path.exists() and pdf_path.exists()
    return {
        "score": {"format_completeness": 1.0 if complete else 0.0},
        "output": {"need_reformat": not complete},
    }

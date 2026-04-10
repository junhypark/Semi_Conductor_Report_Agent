from __future__ import annotations

from pathlib import Path
import re

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

LOCAL_FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_CANDIDATES = [
    ("KoPubWorldDotum", LOCAL_FONT_DIR / "KoPubWorld Dotum Medium.ttf"),
    ("KoPubWorldDotumBold", LOCAL_FONT_DIR / "KoPubWorld Dotum Bold.ttf"),
    ("KoPubWorldBatang", LOCAL_FONT_DIR / "KoPubWorld Batang Medium.ttf"),
    ("KoPubWorldBatangBold", LOCAL_FONT_DIR / "KoPubWorld Batang Bold.ttf"),
    ("NotoSansCJKkr", Path("/System/Library/Fonts/Supplemental/NotoSansCJKkr-Regular.otf")),
    ("NotoSansKR", Path("/Library/Fonts/NotoSansKR-Regular.otf")),
    ("AppleSDGothicNeo", Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")),
    ("KorPubWorld", Path("/Library/Fonts/KoPubWorld Batang_Pro Medium.otf")),
]
CID_FALLBACK_FONT_NAME = "HYGothic-Medium"


def _register_korean_fonts() -> tuple[str, str]:
    registered: dict[str, str] = {}
    for font_name, font_path in FONT_CANDIDATES:
        if not font_path.exists():
            continue
        try:
            registerFont(TTFont(font_name, str(font_path)))
            registered[font_name] = font_name
        except Exception:
            continue
    if "KoPubWorldDotum" in registered:
        registerFontFamily(
            "KoPubWorldDotum",
            normal=registered["KoPubWorldDotum"],
            bold=registered.get("KoPubWorldDotumBold", registered["KoPubWorldDotum"]),
            italic=registered["KoPubWorldDotum"],
            boldItalic=registered.get("KoPubWorldDotumBold", registered["KoPubWorldDotum"]),
        )
        return registered["KoPubWorldDotum"], registered.get("KoPubWorldDotumBold", registered["KoPubWorldDotum"])
    if "KoPubWorldBatang" in registered:
        registerFontFamily(
            "KoPubWorldBatang",
            normal=registered["KoPubWorldBatang"],
            bold=registered.get("KoPubWorldBatangBold", registered["KoPubWorldBatang"]),
            italic=registered["KoPubWorldBatang"],
            boldItalic=registered.get("KoPubWorldBatangBold", registered["KoPubWorldBatang"]),
        )
        return registered["KoPubWorldBatang"], registered.get("KoPubWorldBatangBold", registered["KoPubWorldBatang"])
    if registered:
        first_font = next(iter(registered.values()))
        return first_font, first_font
    try:
        registerFont(UnicodeCIDFont(CID_FALLBACK_FONT_NAME))
        return CID_FALLBACK_FONT_NAME, CID_FALLBACK_FONT_NAME
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def build_report_pdf(pdf_path: Path | str, title: str, sections: list[tuple[str, str]]) -> None:
    pdf_path = Path(pdf_path)
    font_name, heading_font_name = _register_korean_fonts()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
        author="SK hynix Multi-Agent Prototype",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "KoreanTitle",
        parent=styles["Title"],
        fontName=heading_font_name,
        fontSize=24,
        leading=32,
        alignment=TA_CENTER,
        textColor=HexColor("#17322d"),
        spaceAfter=4 * mm,
    )
    subtitle_style = ParagraphStyle(
        "KoreanSubtitle",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=16,
        alignment=TA_CENTER,
        textColor=HexColor("#56706a"),
        spaceAfter=8 * mm,
    )
    heading_style = ParagraphStyle(
        "KoreanHeading",
        parent=styles["Heading2"],
        fontName=heading_font_name,
        fontSize=14,
        leading=20,
        textColor=HexColor("#0b5f4e"),
        backColor=HexColor("#e8f3ef"),
        borderPadding=(6, 8, 6),
        spaceBefore=3 * mm,
        spaceAfter=3 * mm,
        wordWrap="CJK",
    )
    body_style = ParagraphStyle(
        "KoreanBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=17,
        textColor=HexColor("#1f2a24"),
        wordWrap="CJK",
        allowWidows=1,
        allowOrphans=1,
        spaceAfter=2.4 * mm,
        leftIndent=2,
    )
    bullet_style = ParagraphStyle(
        "KoreanBullet",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        bulletIndent=4,
        spaceAfter=2 * mm,
    )
    quote_style = ParagraphStyle(
        "KoreanQuote",
        parent=body_style,
        leftIndent=14,
        rightIndent=6,
        textColor=HexColor("#41544f"),
        backColor=HexColor("#f2f6f5"),
        borderPadding=(5, 7, 5),
        spaceAfter=2.5 * mm,
    )

    story = [
        Spacer(1, 4 * mm),
        Paragraph(_escape_for_paragraph(title), title_style),
        Paragraph(_escape_for_paragraph("SK hynix 미래 시장 방향 통합 보고서"), subtitle_style),
        HRFlowable(width="100%", thickness=1.2, color=HexColor("#0d6b57"), spaceAfter=6 * mm),
    ]
    for index, (heading, content) in enumerate(sections):
        story.append(Paragraph(_escape_for_paragraph(heading), heading_style))
        paragraphs = _split_paragraphs(content)
        for paragraph in paragraphs:
            for line in _normalize_text(paragraph).splitlines():
                stripped = line.strip()
                if stripped.startswith("인용문:"):
                    style = quote_style
                    markup = _render_quote_markup(stripped)
                else:
                    style = bullet_style if stripped.startswith("- ") else body_style
                    markup = _escape_for_paragraph(stripped)
                story.append(Paragraph(markup, style))
        if index != len(sections) - 1:
            story.append(Spacer(1, 3 * mm))

    doc.build(story)


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return "\n".join(lines) or "내용이 제공되지 않았습니다."


def _split_paragraphs(value: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in value.split("\n\n") if paragraph.strip()]
    return paragraphs or ["내용이 제공되지 않았습니다."]


def _escape_for_paragraph(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_quote_markup(value: str) -> str:
    prefix = "인용문:"
    quote_text = value[len(prefix) :].strip()
    if quote_text.startswith("*") and quote_text.endswith("*") and len(quote_text) >= 2:
        quote_text = quote_text[1:-1].strip()
    if not quote_text.startswith('"'):
        quote_text = '"' + quote_text.strip('"') + '"'
    escaped_prefix = _escape_for_paragraph(prefix)
    escaped_quote = _escape_for_paragraph(quote_text)
    return f"{escaped_prefix} <i>{escaped_quote}</i>"

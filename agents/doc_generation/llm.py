from __future__ import annotations

import json
import os
from typing import Any

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI


def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def build_openai_client() -> OpenAI:
    return wrap_openai(OpenAI())


def _common_writing_rules(*, require_news: bool = False, require_future_reasoning: bool = False) -> str:
    rules = [
        "모든 문장은 한국어 평서문으로 작성한다.",
        "반드시 JSON만 출력한다.",
        "각 값은 정확히 두 개의 문단으로 작성하고 문단 사이는 \\n\\n 으로 구분한다.",
        "각 값의 첫 번째 문단은 최소 6문장 이상으로 작성한다.",
        "각 값의 첫 번째 문단에는 정량적 수치를 최소 2개 포함한다.",
        "정량 정보를 쓸 때는 숫자만 나열하지 말고 대역폭, 수율, 전력 효율, 적층 수, 열저항, 생산 능력처럼 지표 이름을 함께 명시한다.",
        "요약이나 전달 과정에서 정보를 축약하여 누락하면 안 된다.",
        "이해한 정보를 구체적으로 다시 서술하되 근거, 기술 구조, 수치, 효과를 보존한다.",
        "각 값은 Executive Summary보다 명확히 더 자세해야 하며, 핵심 근거와 세부 설명을 생략하면 안 된다.",
        "각 값은 최소 900자 이상을 목표로 작성한다.",
        "참조한 원문이 영어여도 출력은 반드시 자연스러운 한국어로 번역하여 작성한다.",
        "모든 사례는 [기사 제목]-[사례로 선정한 이유]-[사례 설명]-[주장] 형식으로 작성한다.",
        "사례가 여러 개이면 각 사례를 줄바꿈으로 분리한다.",
        "직접 인용을 쓸 때는 반드시 '인용문: *\"...\"*' 형식을 사용한다.",
        "각 값의 마지막 문단은 반드시 'Limitation:'으로 시작한다.",
        "Limitation 단락에는 우려 이유를 설명하는 불릿 포인트를 최소 3개 포함한다.",
        "불릿 포인트는 반드시 '- ' 형식을 사용한다.",
        "HTML 태그나 <br> 같은 줄바꿈 태그를 사용하지 않는다.",
    ]
    if require_news:
        rules.append("관련성이 있는 최신 뉴스 1건 이상을 제목, 출처, 날짜와 함께 본문에 포함한다.")
    if require_future_reasoning:
        rules.append("미래 방향성 추론에는 최소 1개의 근거 출처를 포함한다.")
        rules.append("첫 번째 문단 안에 '추론 근거:' 문구를 넣고, 그 아래에 최소 3개의 불릿 포인트를 포함한다.")
    return " ".join(rules)


@traceable(name="market_llm", run_type="llm")
def generate_market_sections(
    *,
    client: OpenAI,
    model: str,
    user_query: str,
    rag_count: int,
    web_count: int,
    arxiv_count: int,
    evidence_digest: str,
    peer_context: str,
) -> dict[str, str]:
    prompt = (
        "당신은 SK hynix 전략 보고서를 작성하는 Market LLM이다. "
        "키는 background_of_analysis, strategic_implications, trl_evaluation 이어야 한다. "
        f"{_common_writing_rules(require_news=True, require_future_reasoning=True)} "
        "background_of_analysis는 전체 업무 개요, 검토한 문서 수, 보고서의 최종 결론을 함께 포함한다. "
        "background_of_analysis는 왜 미래전략 기획에 집중하는지도 설명한다. "
        "strategic_implications는 미래 방향성과 R&D 방향을 제시하고, 반드시 근거 출처와 '추론 근거:' 불릿 목록을 포함한다. "
        "strategic_implications는 마지막에 '종합 시사점 및 Conclusion:'을 넣고 결론을 정리한다. "
        "Executive Summary는 후속 단계에서 최대 5줄 수준으로 짧게 작성되므로, 현재 작성하는 각 절은 그보다 최소 2배 이상 자세해야 한다. "
        "trl_evaluation는 TRL 4~6 범위와 1~5 수치 평가를 함께 서술한다. "
        "다른 LLM이 남긴 요약본도 정보 손실 없이 이어받아 반영한다. "
        f"질문: {user_query}. 로컬 문서 수: {rag_count}. 웹 기사 수: {web_count}. arXiv 참고 자료 수: {arxiv_count}. "
        f"근거 요약: {evidence_digest} "
        f"다른 LLM 요약본: {peer_context}"
    )
    return _json_completion(client=client, model=model, prompt=prompt)


@traceable(name="sk_hynix_llm", run_type="llm")
def generate_skhynix_sections(
    *,
    client: OpenAI,
    model: str,
    user_query: str,
    evidence_digest: str,
    peer_context: str,
) -> dict[str, str]:
    prompt = (
        "당신은 SK hynix LLM이다. 키는 competitor_trend_analysis 이어야 한다. "
        f"{_common_writing_rules(require_news=True)} "
        "경쟁사는 삼성전자, TSMC, 마이크론을 포함하고, HBM4, PIM, CXL, 패키징, 공급망, 데이터센터 채택 관점을 반영한다. "
        "경쟁사 동향에는 경쟁사별로 현재 전략, 미래 계획, 전략 TRL 평가를 각각 설명한다. "
        "RAG에 정보가 없으면 웹 검색 결과를 사용하고, 웹 검색에도 없으면 확인 가능한 자료를 찾지 못해 사용하지 못했다는 내용을 보고서 문체로 쓴다. "
        "경쟁사 동향에는 최소 1개의 구체적인 뉴스 사례를 반드시 넣는다. "
        "Executive Summary는 후속 단계에서 최대 5줄 수준으로 짧게 작성되므로, 현재 작성하는 절은 사례와 근거를 생략하지 않고 그보다 최소 2배 이상 자세해야 한다. "
        "다른 LLM이 남긴 요약본도 정보 손실 없이 이어받아 반영한다. "
        f"질문: {user_query}. 근거 요약: {evidence_digest} 다른 LLM 요약본: {peer_context}"
    )
    return _json_completion(client=client, model=model, prompt=prompt)


@traceable(name="technique_llm", run_type="llm")
def generate_technique_sections(
    *,
    client: OpenAI,
    model: str,
    user_query: str,
    evidence_digest: str,
    peer_context: str,
) -> dict[str, str]:
    prompt = (
        "당신은 Technique LLM이다. 키는 current_status_of_target_technologies 이어야 한다. "
        f"{_common_writing_rules()} "
        "HBM4, PIM, CXL의 현재 기술 상태, 제조 제약, 생태계 성숙도, 시스템 통합성을 분석한다. "
        "읽은 PDF를 기준으로 약 3줄 요약을 먼저 제시하고, 이어서 장점과 단점을 불릿 포인트로 정리한다. "
        "PDF 제목이 일반적인 파일명처럼 보이면 문서 첫 페이지의 실제 제목을 우선 사용한다. "
        "Executive Summary는 후속 단계에서 최대 5줄 수준으로 짧게 작성되므로, 현재 작성하는 절은 기술 구조와 효과를 압축하지 않고 그보다 최소 2배 이상 자세해야 한다. "
        "다른 LLM이 남긴 요약본도 정보 손실 없이 이어받아 반영한다. "
        f"질문: {user_query}. 근거 요약: {evidence_digest} 다른 LLM 요약본: {peer_context}"
    )
    return _json_completion(client=client, model=model, prompt=prompt)


@traceable(name="judge_llm", run_type="llm")
def evaluate_report_sections(*, client: OpenAI, model: str, user_query: str, section_summaries: dict[str, str]) -> dict[str, Any]:
    prompt = (
        "당신은 보고서 품질 평가자이다. 반드시 JSON만 출력하고 키는 judge_score, trl_numeric_score, rationale 이어야 한다. "
        "judge_score는 0에서 1 사이 소수, trl_numeric_score는 1에서 5 사이 소수로 출력한다. "
        "rationale은 한국어 한 단락으로 작성한다. "
        "평가 기준은 평서문 사용, 정량 수치 포함, 정량 수치에 지표 이름 동반 여부, 관련 뉴스 포함 여부, 정보 손실 없는 구체적 설명, 사례의 4요소 형식 유지, 미래 방향성의 근거 제시, 추론 근거 불릿 3개 이상, Limitation 불릿 3개 이상, Executive Summary가 최대 5줄 수준으로 짧게 유지되면서 전체 섹션을 반영하는지 여부, 각 본문 섹션이 Executive Summary보다 최소 2배 이상 자세한지 여부, 분석 배경의 미래전략 집중 이유와 결론 정리, 경쟁사별 전략/미래 계획/TRL 비교, 핵심 기술 현황의 3줄 요약과 장단점 불릿, HTML 태그 미사용, 참고문헌이 번호형 각주만 포함하는지 여부, TRL 4~6 적합성이다. "
        "평가에는 원문이 아닌 섹션 요약본만 사용한다. "
        f"질문: {user_query}. 섹션 요약본: {json.dumps(section_summaries, ensure_ascii=False)}"
    )
    return _json_completion(client=client, model=model, prompt=prompt)


def _json_completion(*, client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a precise JSON-only assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)

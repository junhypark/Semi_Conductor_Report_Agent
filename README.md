# Subject
- RAG에 넣어볼 자료는 phjpurpleoob@gmail.com 으로 문의주시면 감사하겠습니다.

## Overview
- Objective : SK hynix의 HBM4, PIM, CXL 중심 미래 시장 방향을 분석하고 멀티 에이전트 기반 보고서를 자동 생성합니다.
- Method : Supervisor가 Query Transformation, RAG, Web Search, Doc Generation, Format, Scrape를 LangGraph로 오케스트레이션하고 PDF 문서와 웹 기사, 벡터 검색 결과를 종합합니다.
- Tools : FastAPI, LangGraph, Qdrant, Tavily, OpenAI API

## Features
- PDF 자료 기반 정보 추출 및 RAG 인덱싱
- Tavily 기반 최신 기사 검색과 SK hynix Newsroom 검색
- Executive Summary, 경쟁사 비교, TRL 평가, 참고문헌을 포함한 Markdown/PDF 보고서 생성
- HTML 대시보드에서 최신 결과, 그래프, PDF를 확인 가능
- Docker Compose 기반 멀티 서비스 실행
- 확증 편향 방지 전략 : 로컬 문서, 최신 웹 기사, 경쟁사 비교, TRL 평가, Limitation 절을 함께 사용하여 단일 근거에 치우치지 않도록 구성

## Tech Stack

| Category | Details |
|----------|---------|
| Framework | FastAPI, LangGraph, Python, Docker Compose |
| LLM | OpenAI API (`gpt-4.1-mini` 기본값) |
| Retrieval | Qdrant + local corpus fallback |
| Embedding | BAAI/bge-m3 |
| Frontend | HTML dashboard |
| Observability | LangSmith (optional) |

## Agents

- Supervisor: 전체 파이프라인 상태를 관리하고 재시도, 종료, 평가 흐름을 제어합니다.
- Query Transformation: 사용자 질문을 기술, 시장, 경쟁사, 미래 예측 쿼리로 확장합니다.
- RAG: 로컬 PDF와 문서 코퍼스를 벡터 검색하여 관련 근거를 반환합니다.
- Web Search: Tavily를 사용해 최신 기사와 공식 Newsroom 결과를 수집합니다.
- Doc Generation: 수집된 근거를 바탕으로 한국어 보고서 초안을 생성합니다.
- Format: Markdown과 PDF 결과물을 생성합니다.
- Scrape: 정기 수집용 ingestion API를 제공합니다.

## Architecture

![Architecture]([./outputs/graphs/supervisor_graph_xray.png](https://github.com/user-attachments/assets/e6eb8b8a-3567-4bc2-bb99-c4f8c0cb77b7))

## Directory Structure

```text
├── data/                  # PDF 문서, 처리 결과, 벡터 저장소
├── agents/                # Agent 모듈
├── prompts/               # 프롬프트 템플릿
├── outputs/               # 보고서, PDF, 그래프 저장
├── docker/                # Dockerfile, docker-compose 설정
├── frontend/              # HTML 대시보드
├── scripts/               # 그래프 생성, 검증, 문서 인덱싱 스크립트
├── shared/                # 공통 스키마, PDF, 임베딩, 상수
├── app.py                 # 실행 진입점
└── README.md
```

## Setup With uv

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

기존 Docker 실행은 그대로 `requirements.txt`를 사용합니다.

## Run

```bash
./.venv/bin/python -m uvicorn app:app --reload
```

브라우저에서 `http://127.0.0.1:8000/`를 열면 됩니다.

## Docker Compose

```bash
docker compose -f docker/docker-compose.yml up --build
```

## RAG Ingestion

```bash
docker compose -f docker/docker-compose.yml up -d qdrant-service
./.venv/bin/python scripts/ingest_documents.py data/raw/manual
```

## Validation

```bash
./.venv/bin/python scripts/generate_graph.py
./.venv/bin/python scripts/validate_local.py
docker compose -f docker/docker-compose.yml config
```

## Contributors
- 박준형

# AGENTS.md

## Overview
- This project makes a report about SK-Hynix future market direction.
- It is compromised with supervisor structure using Langgraph
- The overall architecture is same with the image `./mermaid-diagram (1).png`
- Supervisor is connected with
    - Query Transformation -> RAG Agent, Web Search Agent
    - Doc Writer Agent
    - Format Agent
- Each edge has conditions or not
- Agent analyze these parts
    - Market Trends regarding with AI makret
    - Analyze AI consumer target and predict next AI consumer
    - Analyze Market Trends and predict next 5 to 10 years trend
    - Analyze competetors: Samsung, TSMC, Micron
        - Regarding with HBM4, PIM, CXL
        - Their future directions
        - Techniques' critical points
    - Analyze SK-Hynix market demanding points and using this demanding points, analyze SK-Hynix's HBM4, PIM, CXL manufacturing techniques' critical points and future market direction
    - Analyze academic paper regarding with predicted market direction
    - Suggest next future market direction and R&D direction

---

## Evaluation

| Evaluation Method | Target | Criteria |
|------------------|--------|----------|
| LLM Semantic Evaluation | Relevance between retrieved data and user query | ≥ 0.8 |
| TRL Score | Whether suggested strategy is between TRL 4~6 | ≥ 3.5 / 5 |
| Judge Score | Whether each step output is acceptable | ≥ 0.6 / 1 |

---

## Agents

### Supervisor
#### Role
- Input the user query
- Control overall pipeline execution

#### Evaluation Criteria
- Whether each stage passes defined thresholds
- Whether final report is complete and consistent

#### Routing Rule
- If retrieval score < 0.8 → send to RAG / Web Search
- If Judge score < 0.6 → regenerate stage
- If all satisfied → terminate

#### API Responsibility
- Receive the initial user query
- Manage the overall state
- Route to Query Transformation, RAG, Web Search, Doc Generation, and Format Node
- Aggregate evaluation results from each agent
- Decide retry / continue / terminate

---

### Query Transformation
#### Role
- Transform user query into multiple diverse queries

#### Process
- Generate multiple query variations (semantic / perspective-based)
- Include:
  - technical query
  - market query
  - competitor query
  - future prediction query

#### Output
- List of transformed queries

#### API Responsibility
- Accept user query or supervisor payload
- Return transformed queries in structured JSON

---

### RAG
#### Role
- Input transformed queries
- Retrieve relevant documents

#### Process
- Use embedding model: BAAI/bge-m3
- Select Vector DB: FAISS or Qdrant
- Use retrieval strategies

#### Evaluation
- LLM Semantic Evaluation
    - If score < 0.8 → refine retrieval

#### Output
- Retrieved documents
- Retrieval scores
- Configuration used

#### API Responsibility
- Accept transformed queries
- Retrieve documents from vector DB
- Return retrieved chunks, metadata, and evaluation score

---

### Web Search
#### Role
- Use Tavily to retrieve latest information

#### Process
- Use transformed queries
- Collect most recent 5 news per category

#### Evaluation
- Must include recent, non-duplicate, reliable sources

#### API Responsibility
- Accept transformed queries
- Search latest web results
- Return normalized articles with title, url, summary, date, source

---

### Doc Generation
#### Role
- Generate report using 3 LLMs + 1 evaluator

#### Structure
- Market LLM
- SK-Hynix LLM
- Technique LLM

#### TRL Evaluation (IMPORTANT CHANGE)
- Market LLM MUST evaluate TRL internally
- TRL score must be between TRL 4~6 level
- Also output numeric score (1~5 scale)
- If TRL score < 3.5 → regenerate strategy

#### Evaluation (Sub-Supervisor)
- Judge Score
    - If < 0.6 → regenerate or debate again

#### Output Structure
- Background of the Analysis
- Current Status of Target Technologies
- Competitor Trend Analysis
- Strategic Implications
- TRL Evaluation (included in Market section)

#### API Responsibility
- Accept RAG results + Web Search results + user query
- Generate section drafts
- Return section-wise drafts, judge score, and TRL evaluation

---

### Scrape
- Scrape agent collects data periodically (every 4 months)
- Not part of real-time execution

#### API Responsibility
- Run scheduled ingestion
- Scrape arXiv, company reports, and earnings reports
- Store parsed contents into vector DB

---

## FastAPI Specification

### Common API Rules
- Every container must expose similar API path names
- Every agent container must use the same style of endpoint naming for consistency
- All services must communicate through FastAPI
- All request and response bodies must use JSON
- Every service must include health check and execution endpoint
- Every service must return standardized response format

### Common API Paths for All Containers
- `GET /health`
    - Check whether the service is alive
- `GET /meta`
    - Return service name, version, role, and supported capabilities
- `POST /invoke`
    - Main execution endpoint
- `POST /evaluate`
    - Optional endpoint for score / validation / judge
- `GET /docs`
    - FastAPI auto-generated documents
- `GET /openapi.json`
    - OpenAPI schema

### Standard Request Format
```json
{
  "request_id": "string",
  "trace_id": "string",
  "agent_name": "string",
  "payload": {},
  "context": {},
  "config": {}
}
````

### Standard Response Format

```json
{
  "request_id": "string",
  "trace_id": "string",
  "agent_name": "string",
  "status": "success",
  "score": {},
  "output": {},
  "error": null
}
```

### Service-specific API Expectations

#### Supervisor Service

* `POST /invoke`

  * Input: user query, optional runtime config
  * Output: next routing decision, aggregated state, execution summary

#### Query Transformation Service

* `POST /invoke`

  * Input: user query
  * Output: transformed query list

#### RAG Service

* `POST /invoke`

  * Input: transformed queries
  * Output: retrieved documents, retrieval metadata, semantic evaluation score
* `POST /evaluate`

  * Input: retrieved results + user query
  * Output: semantic relevance score

#### Web Search Service

* `POST /invoke`

  * Input: transformed queries
  * Output: normalized news/article results
* `POST /evaluate`

  * Input: articles + user query
  * Output: relevance / freshness / duplication check result

#### Doc Generation Service

* `POST /invoke`

  * Input: retrieval results + web search results + user query
  * Output: section drafts + judge score + TRL score
* `POST /evaluate`

  * Input: generated document sections
  * Output: judge result and regeneration decision

#### Format Service

* `POST /invoke`

  * Input: section drafts
  * Output: markdown report path, pdf report path
* `POST /evaluate`

  * Input: formatted report
  * Output: format completeness result

#### Scrape Service

* `POST /invoke`

  * Input: scrape target config
  * Output: ingestion result
* `POST /evaluate`

  * Optional for document quality / parse quality check

---

## Docker Compose

### Requirement

* All modules must be runnable with `docker-compose`
* Each agent should be an independent service container
* Services should communicate over internal docker network
* Supervisor should call other services by service name
* Shared volumes can be used for `data/` and `outputs/`

### Minimum Services

* supervisor
* query-transformation
* rag
* web-search
* doc-generation
* format
* scrape

### Compose Expectation

* Each service should define:

  * `build`
  * `container_name`
  * `ports`
  * `volumes`
  * `environment`
  * `depends_on`
  * `networks`

### Example Service Naming Rule

* `supervisor-service`
* `query-transformation-service`
* `rag-service`
* `web-search-service`
* `doc-generation-service`
* `format-service`
* `scrape-service`

### Internal Communication Rule

* Use docker service names for API calls
* Example:

  * `http://query-transformation-service:8000/invoke`
  * `http://rag-service:8000/invoke`
  * `http://web-search-service:8000/invoke`

---

## Lang Graph

### Global State

* user_query
* transformed_queries
* rag_results
* web_results
* draft_sections
* evaluation_scores
* iteration_count

### Local State

* current_goal
* input_payload
* outputs
* evaluation_result

### Nodes

* Supervisor Node
* Query Transformation Node
* RAG Node
* Web Search Node
* Doc Generation Node
* Format Node

### Visualization Requirement

* The graph MUST be visualized after graph construction
* Visualization MUST use `xray=True`
* Save graph visualization output as image file
* The generated graph image must be stored in project output directory

### Visualization Example Requirement

* Use LangGraph built graph visualization function
* Example expectation:

`graph.get_graph(xray=True).draw_mermaid_png()`

* Save or display the visualization result
* The project should include code to generate the graph visualization automatically

### Flow

* Query Transformation → Retrieval → Supervisor
* Doc Generation → Supervisor
* Format → End

### Edges

#### Conditional Flags

* need_more_retrieval
* need_regeneration
* need_reformat
* passed_all_evaluation

---

## Output Artifacts

* Markdown report
* PDF report
* Graph visualization image with `xray=True`

---

## Rules

* Clean Code
* Clean Architecture
* Explainable Code
* Use Docker Container

---

## Folder Format

├── data/
├── agents/
├── prompts/
├── outputs/
├── app.py
├── .env
└── README.md

### Recommended Detailed Folder Format

├── data/
│   ├── raw/
│   ├── processed/
│   └── vectordb/
├── agents/
│   ├── supervisor/
│   ├── query_transformation/
│   ├── rag/
│   ├── web_search/
│   ├── doc_generation/
│   ├── format/
│   └── scrape/
├── prompts/
│   ├── supervisor/
│   ├── query_transformation/
│   ├── rag/
│   ├── web_search/
│   ├── doc_generation/
│   └── format/
├── outputs/
│   ├── reports/
│   ├── pdf/
│   └── graphs/
├── docker/
│   └── docker-compose.yml
├── app.py
└── README.md

---

## Docker

* Make Each Modules into Docker
* Each Docker will communicate using Fast API

### API Standard

* /health
* /meta
* /invoke
* /evaluate
* /docs
* /openapi.json

### Docker Compose Rule

* The whole system must be executable with docker-compose
* All containers must share consistent API endpoint naming
* Environment variables must be configurable from compose
* Output and data folders should be mounted as volumes
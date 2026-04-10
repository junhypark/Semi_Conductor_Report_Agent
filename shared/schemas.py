from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class StandardRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: f"req-{uuid4().hex}")
    trace_id: str = Field(default_factory=lambda: f"trace-{uuid4().hex}")
    agent_name: str = "unknown"
    payload: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class StandardResponse(BaseModel):
    request_id: str
    trace_id: str
    agent_name: str
    status: str = "success"
    score: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class MetaResponse(BaseModel):
    service_name: str
    version: str
    role: str
    supported_capabilities: list[str]

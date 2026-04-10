from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable

from fastapi import FastAPI
from dotenv import load_dotenv

from shared.constants import DEFAULT_VERSION, PROJECT_ROOT
from shared.schemas import HealthResponse, MetaResponse, StandardRequest, StandardResponse

Handler = Callable[[StandardRequest], dict[str, Any] | Awaitable[dict[str, Any]]]
StartupHook = Callable[[], None | Awaitable[None]]

load_dotenv(PROJECT_ROOT / ".env", override=False)


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _success_response(service_name: str, request: StandardRequest, result: dict[str, Any]) -> StandardResponse:
    return StandardResponse(
        request_id=request.request_id,
        trace_id=request.trace_id,
        agent_name=service_name,
        status=result.get("status", "success"),
        score=result.get("score", {}),
        output=result.get("output", {}),
        error=result.get("error"),
    )


def _error_response(service_name: str, request: StandardRequest, exc: Exception) -> StandardResponse:
    return StandardResponse(
        request_id=request.request_id,
        trace_id=request.trace_id,
        agent_name=service_name,
        status="error",
        score={},
        output={},
        error={"type": exc.__class__.__name__, "message": str(exc)},
    )


def create_service_app(
    *,
    service_name: str,
    role: str,
    capabilities: list[str],
    invoke_handler: Handler,
    evaluate_handler: Handler,
    version: str = DEFAULT_VERSION,
    startup_hook: StartupHook | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if startup_hook is not None:
            await _maybe_await(startup_hook())
        yield

    app = FastAPI(title=service_name, version=version, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", service=service_name, version=version)

    @app.get("/meta", response_model=MetaResponse)
    async def meta() -> MetaResponse:
        return MetaResponse(
            service_name=service_name,
            version=version,
            role=role,
            supported_capabilities=capabilities,
        )

    @app.post("/invoke", response_model=StandardResponse)
    async def invoke(request: StandardRequest) -> StandardResponse:
        try:
            result = await _maybe_await(invoke_handler(request))
            return _success_response(service_name, request, result)
        except Exception as exc:  # pragma: no cover - defensive API envelope
            return _error_response(service_name, request, exc)

    @app.post("/evaluate", response_model=StandardResponse)
    async def evaluate(request: StandardRequest) -> StandardResponse:
        try:
            result = await _maybe_await(evaluate_handler(request))
            return _success_response(service_name, request, result)
        except Exception as exc:  # pragma: no cover - defensive API envelope
            return _error_response(service_name, request, exc)

    return app

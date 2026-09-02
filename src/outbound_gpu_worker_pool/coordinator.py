"""The standalone coordinator application (install the `coordinator` extra).

The app is deliberately small: a health probe and the authenticated worker
router. Job submission is not an HTTP surface here — a host calls
`WorkerPoolService.submit` from behind its own authentication and mounts
`create_worker_router` next to its own routes when it wants one process.
"""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI

from outbound_gpu_worker_pool.routes import create_worker_router
from outbound_gpu_worker_pool.service import WorkerPoolService

DEFAULT_COORDINATOR_TITLE = "Outbound GPU Worker Pool Coordinator"


class Startable(Protocol):
    """A store whose connections open with the app and close with it."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


def create_coordinator_app(
    service: WorkerPoolService,
    *,
    title: str = DEFAULT_COORDINATOR_TITLE,
    lifecycle: Sequence[Startable] = (),
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        started: list[Startable] = []
        try:
            for component in lifecycle:
                await component.start()
                started.append(component)
            yield
        finally:
            for component in reversed(started):
                await component.stop()

    application = FastAPI(title=title, lifespan=lifespan)

    @application.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "healthy",
            "worker_pool": {
                "auth": service.auth_method,
                "capabilities": sorted(service.capabilities_schema()["capabilities"]),
            },
        }

    application.include_router(create_worker_router(service))
    return application

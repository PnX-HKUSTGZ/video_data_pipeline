"""HTTP endpoints serving typed metadata and images registered in the active index."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .models import IndexStatus, LabelState, OpenRequest, SampleDetail, SamplePage, Stats
from .service import SessionError, ViewerService


def create_app() -> FastAPI:
    """Create an isolated single-user service; worker cleanup follows ASGI lifespan."""
    service = ViewerService()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            service.close()

    app = FastAPI(title="YOLO Dataset Viewer", lifespan=lifespan)

    @app.exception_handler(SessionError)
    async def session_error(request: Request, error: SessionError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(LookupError)
    async def lookup_error(request: Request, error: LookupError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.post("/api/datasets", response_model=IndexStatus, status_code=202)
    def open_dataset(body: OpenRequest, request: Request) -> IndexStatus:
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "请从本服务网页打开数据集")
        return service.open(body)

    @app.get("/api/datasets/{session_id}", response_model=IndexStatus)
    def status(session_id: str) -> IndexStatus:
        return service.status(session_id)

    @app.get("/api/datasets/{session_id}/samples", response_model=SamplePage)
    def samples(
        session_id: str,
        split: str | None = None,
        class_id: int | None = None,
        state: LabelState | None = None,
        search: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(48, ge=1, le=200),
    ) -> SamplePage:
        dataset = service.dataset(session_id)
        selected = dataset.select(split, class_id, state, search)
        return SamplePage(
            items=[s.info() for s in selected[offset : offset + limit]],
            total=len(selected),
            offset=offset,
            limit=limit,
        )

    @app.get("/api/datasets/{session_id}/stats", response_model=Stats)
    def stats(
        session_id: str,
        split: str | None = None,
        class_id: int | None = None,
        state: LabelState | None = None,
        search: str = "",
    ) -> Stats:
        dataset = service.dataset(session_id)
        return dataset.stats(dataset.select(split, class_id, state, search))

    @app.get("/api/datasets/{session_id}/samples/{sample_id}", response_model=SampleDetail)
    def detail(session_id: str, sample_id: int) -> SampleDetail:
        return service.detail(session_id, sample_id)

    @app.get("/api/datasets/{session_id}/samples/{sample_id}/image")
    def image(session_id: str, sample_id: int, thumbnail: bool = False) -> Response:
        try:
            data = service.image(session_id, sample_id, thumbnail)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            if isinstance(error, SessionError):
                raise
            raise HTTPException(422, f"图片无法解码：{error}") from error
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="web")
    return app

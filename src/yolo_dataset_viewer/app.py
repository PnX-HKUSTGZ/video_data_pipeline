"""HTTP endpoints serving typed metadata and images registered in the active index."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.middleware.base import RequestResponseEndpoint

from .dataset import DatasetError
from .edit_models import DraftStatus, EditRequest, RevisionRequest
from .editing import DraftError
from .exports import ExportPreview, ExportRequest, ExportStatus, RecoveryInfo
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

    @app.exception_handler(DraftError)
    async def draft_error(request: Request, error: DraftError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(DatasetError)
    async def export_error(request: Request, error: DatasetError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(OSError)
    async def storage_error(request: Request, error: OSError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": f"文件操作失败：{error}"})

    @app.middleware("http")
    async def local_writes(request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Apply the same same-origin policy to every write, including export."""
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse(status_code=403, content={"detail": "请从本服务网页操作"})
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse(status_code=403, content={"detail": "不允许跨站写入"})
        return await call_next(request)

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
        edit_state: Literal["modified", "deleted"] | None = None,
    ) -> SamplePage:
        return service.page(session_id, split, class_id, state, search, offset, limit, edit_state)

    @app.get("/api/datasets/{session_id}/stats", response_model=Stats)
    def stats(
        session_id: str,
        split: str | None = None,
        class_id: int | None = None,
        state: LabelState | None = None,
        search: str = "",
        edit_state: Literal["modified", "deleted"] | None = None,
    ) -> Stats:
        return service.stats(session_id, split, class_id, state, search, edit_state)

    @app.get("/api/datasets/{session_id}/draft", response_model=DraftStatus)
    def draft(session_id: str) -> DraftStatus:
        return service.draft(session_id)

    @app.post("/api/datasets/{session_id}/draft/operations", response_model=DraftStatus)
    def edit(session_id: str, body: EditRequest) -> DraftStatus:
        return service.edit(session_id, body)

    @app.post("/api/datasets/{session_id}/draft/{action}", response_model=DraftStatus)
    def history(
        session_id: str, action: Literal["undo", "redo", "discard"], body: RevisionRequest
    ) -> DraftStatus:
        return service.history(session_id, action, body.expected_revision)

    @app.post("/api/datasets/{session_id}/exports/preview", response_model=ExportPreview)
    def preview_export(session_id: str, body: ExportRequest) -> ExportPreview:
        return service.preview_export(session_id, body)

    @app.post("/api/datasets/{session_id}/exports", response_model=ExportStatus, status_code=202)
    def start_export(session_id: str, body: ExportRequest) -> ExportStatus:
        return service.start_export(session_id, body)

    @app.get("/api/datasets/{session_id}/exports/recoveries", response_model=list[RecoveryInfo])
    def recoveries(session_id: str) -> list[RecoveryInfo]:
        return service.recoveries(session_id)

    @app.post(
        "/api/datasets/{session_id}/exports/recover", response_model=IndexStatus, status_code=202
    )
    def recover(session_id: str, body: RecoveryRequest) -> IndexStatus:
        return service.recover(session_id, body.expected_revision, body.backup_path)

    @app.get("/api/datasets/{session_id}/exports/{export_id}", response_model=ExportStatus)
    def export_status(session_id: str, export_id: str) -> ExportStatus:
        return service.export_status(session_id, export_id)

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


class RecoveryRequest(RevisionRequest):
    """Explicitly choose a verified, server-generated backup for recovery."""

    backup_path: str

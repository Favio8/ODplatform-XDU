from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..services.dataset_upload import upload_dataset_archive
from ..services.workspace import (
    list_dataset_profiles,
    create_dataset_profile,
    update_dataset_profile,
    delete_dataset_profile,
)

router = APIRouter(prefix="/api", tags=["datasets"])


class DatasetCreate(BaseModel):
    name: str
    class_names: list[str]
    train: int = 0
    val: int = 0
    test: int = 0


class DatasetUpdate(BaseModel):
    class_names: list[str] | None = None
    train: int | None = None
    val: int | None = None
    test: int | None = None


class DatasetUploadResponse(BaseModel):
    dataset_name: str
    raw_path: str
    file_count: int
    dir_count: int
    total_bytes: int
    detected_format: dict
    next_step: str


def _profile_to_dict(profile) -> dict:
    return {
        "name": profile.name,
        "task": profile.task,
        "yaml_path": str(profile.yaml_path),
        "data_root": str(profile.data_root),
        "splits": profile.splits,
        "class_names": profile.class_names,
        "coverage": profile.coverage,
        "status": profile.status,
    }


@router.get("/datasets")
def get_datasets() -> dict:
    return {
        "items": [_profile_to_dict(p) for p in list_dataset_profiles()]
    }


@router.post("/datasets")
def create_dataset(body: DatasetCreate) -> dict:
    try:
        profile = create_dataset_profile(
            name=body.name,
            class_names=body.class_names,
            train=body.train,
            val=body.val,
            test=body.test,
        )
        return _profile_to_dict(profile)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    dataset_name: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "只支持上传 ZIP 数据集包。")
    try:
        content = await file.read()
        return upload_dataset_archive(dataset_name=dataset_name, archive_bytes=content)
    except FileExistsError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"数据集上传失败: {exc}")


@router.put("/datasets/{name}")
def update_dataset(name: str, body: DatasetUpdate) -> dict:
    try:
        profile = update_dataset_profile(
            name=name,
            class_names=body.class_names,
            train=body.train,
            val=body.val,
            test=body.test,
        )
        return _profile_to_dict(profile)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@router.delete("/datasets/{name}")
def delete_dataset(name: str) -> dict:
    try:
        delete_dataset_profile(name)
        return {"success": True, "name": name}
    except ValueError as exc:
        raise HTTPException(404, str(exc))

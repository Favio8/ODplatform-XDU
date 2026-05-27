from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np
import pytest

from odp_platform.frame_source import (
    AsyncSource,
    CameraConfig,
    CameraSource,
    FPSCounter,
    ImageFolderSource,
    ImageSource,
    Metrics,
    RateCounter,
    ThreadedSource,
    VideoSource,
    create_async_source,
    create_frame_source,
    create_threaded_source,
)


def _write_image(path: Path, value: int = 64) -> None:
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (32, 24),
    )
    assert writer.isOpened()
    for idx in range(frame_count):
        frame = np.full((24, 32, 3), idx * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_public_api_smoke() -> None:
    assert CameraConfig().camera_id == 0
    assert FPSCounter().fps == 0.0
    assert RateCounter().fps == 0.0
    assert Metrics().snapshot()["capture_fps"] == 0.0


def test_create_frame_source_for_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    _write_image(image_path)

    source = create_frame_source(str(image_path))

    assert isinstance(source, ImageSource)
    assert source.open() is True
    frame = source.read()
    assert frame is not None
    assert frame.width == 32
    assert frame.height == 24
    assert source.read() is None
    source.close()


def test_create_frame_source_for_folder(tmp_path: Path) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    _write_image(folder / "b.JPG", 10)
    _write_image(folder / "a.png", 20)

    source = create_frame_source(str(folder))

    assert isinstance(source, ImageFolderSource)
    assert source.open() is True
    frame = source.read()
    assert frame is not None
    assert frame.info.filename == "a.png"
    source.close()


def test_create_frame_source_for_video(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_video(video_path)

    source = create_frame_source(str(video_path))

    assert isinstance(source, VideoSource)
    assert source.open() is True
    frame = source.read()
    assert frame is not None
    assert frame.width == 32
    assert frame.height == 24
    source.close()


def test_create_frame_source_for_camera_id() -> None:
    source = create_frame_source("0")
    assert isinstance(source, CameraSource)


def test_create_frame_source_rejects_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jpg"
    with pytest.raises(ValueError, match="路径不存在"):
        create_frame_source(str(missing))


def test_create_frame_source_rejects_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("nope", encoding="utf-8")

    with pytest.raises(ValueError, match="不支持的文件格式"):
        create_frame_source(str(file_path))


def test_create_threaded_source_wraps_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    _write_image(image_path)

    source = create_threaded_source(str(image_path), read_timeout=0.5)

    assert isinstance(source, ThreadedSource)
    assert source.open() is True
    frame = source.read()
    assert frame is not None
    source.close()


def test_create_async_source_wraps_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    _write_image(image_path)

    async def _run() -> None:
        source = create_async_source(str(image_path), threaded=False)
        assert isinstance(source, AsyncSource)
        await source.open()
        frame = await source.read()
        assert frame is not None
        await source.close()

    asyncio.run(_run())


def test_camera_config_model_copy_updates_camera_id() -> None:
    config = CameraConfig(camera_id=1, width=640, height=480, fps=60)
    source = create_frame_source("2", camera_config=config)

    assert isinstance(source, CameraSource)
    assert source.config.camera_id == 2
    assert config.camera_id == 1


def test_video_source_seek_by_frame(tmp_path: Path) -> None:
    video_path = tmp_path / "seek.mp4"
    _write_video(video_path, frame_count=4)
    source = VideoSource(str(video_path))

    assert source.open() is True
    assert source.seek(frame=2) is True
    frame = source.read()
    assert frame is not None
    assert frame.info.frame_index == 2
    source.close()

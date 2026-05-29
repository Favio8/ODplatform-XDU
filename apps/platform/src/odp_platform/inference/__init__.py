"""Public API for the inference subsystem."""
from __future__ import annotations

from .cancel import CancelToken, InferenceCancelled
from .components import FramePrediction, InferenceArtifact, InferenceSummary
from .hooks import FrameEvent, InferHooks, ProgressEvent
from .pipeline import InferencePipeline, ThreadedPipeline
from .pipeline_config import PipelineConfig, load_pipeline_config
from .room_segmentation import (
    RoomSegmentationOptions,
    RoomSegmentationResult,
    run_room_segmentation,
    write_room_segmentation_outputs,
)
from .service import (
    InferResult,
    InferService,
    InferStats,
    InferenceResult,
    InferenceService,
    infer_yolo,
    run_inference,
)
from .sinks import LocalFileSink, NullSink, OutputSink

__all__ = [
    "CancelToken",
    "FrameEvent",
    "FramePrediction",
    "InferHooks",
    "InferResult",
    "InferService",
    "InferStats",
    "InferenceArtifact",
    "InferenceCancelled",
    "InferencePipeline",
    "InferenceResult",
    "InferenceService",
    "InferenceSummary",
    "LocalFileSink",
    "NullSink",
    "OutputSink",
    "PipelineConfig",
    "ProgressEvent",
    "RoomSegmentationOptions",
    "RoomSegmentationResult",
    "ThreadedPipeline",
    "infer_yolo",
    "load_pipeline_config",
    "run_inference",
    "run_room_segmentation",
    "write_room_segmentation_outputs",
]

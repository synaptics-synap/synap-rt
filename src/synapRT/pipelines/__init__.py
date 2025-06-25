import os
from typing import Any, Callable

from . import base
from . import image_classification
from . import object_detection
from . import runners

from .base import BasePipeline
from .audio_classification import MelSpecClassificationPipeline
from .image_classification import SynapImageClassificationPipeline
from .object_detection import SynapObjectDetectionPipeline
from .utils import PipelineState

__all__ = [
    "base",
    "image_classification",
    "object_detection",
    "runners"
    "pipeline",
    "BasePipeline",
    "PipelineState",
]


def pipeline(
    task: str,
    model: os.PathLike,
    handler: Callable[[Any, float | None], Any] | None = None,
    name: str | None = None,
    **infer_params: Any,
) -> BasePipeline:
    """
    Create a pipeline for the specified task.

    Currently supported tasks:
    - `image-classification`
    - `object-detection`

    `model` can be any valid SyNAP model file.

    If a handler is provided, it must accept two arguments:
    - `result`: Inference result from the model
    - `infer_time`: Inference time in ms (if pipeline is profiled, `None` otherwise)

    :param task: Task to perform
    :type task: str
    :param model: Path to the SyNAP model file
    :type model: os.PathLike
    :param handler: Optional handler to process inference results
    :type handler: Callable[[Any, float | None], Any]
    :param name: Optional name for the pipeline
    :type name: str, optional
    :param infer_params: Additional parameters for inference
    :type infer_params: Any
    :return: Pipeline for the specified task
    :rtype: BasePipeline
    """

    pipeline: BasePipeline | None = None
    if task.lower() == "image-classification":
        pipeline = SynapImageClassificationPipeline(model, handler, **infer_params)
    elif task.lower() == "object-detection":
        pipeline = SynapObjectDetectionPipeline(model, handler, **infer_params)
    elif task.lower() == "audio-classification":
        pipeline = MelSpecClassificationPipeline(model, handler, **infer_params)
    else:
        raise NotImplementedError(f"No pipelines available for task '{task}'")
    if name:
        pipeline.name = name
    return pipeline

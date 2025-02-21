import os
from typing import Any, Callable

from synap.postprocessor import Detector

from .base import SynapBasePipeline
from ..constants import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MAX_RESULTS,
    DEFAULT_USE_NMS,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_IOU_WITH_MIN
)

__all__ = [
    "SynapObjectDetectionPipeline",
]


class SynapObjectDetectionPipeline(SynapBasePipeline):

    """
    Object detection pipeline using SyNAP preprocessor and postprocessor.

    :param model: Path to SyNAP model file
    :type model: os.PathLike
    :param handler: An optional callback function to handle post-processed inference results
    :type handler: Callable[[dict[str, Any], float | None], Any], optional
    :param infer_params: Additional parameters for the inference pipeline
    :type infer_params: Any
    """

    def __init__(
        self,
        model: os.PathLike,
        handler: Callable[[dict[str, Any], float | None], Any] | None = None,
        **infer_params: Any
    ):
        postprocessor: Detector = Detector(
            score_threshold=float(infer_params.get("confidence", DEFAULT_CONFIDENCE)),
            n_max=int(infer_params.get("max_results", DEFAULT_MAX_RESULTS)),
            nms=str(infer_params.get("use_nms", DEFAULT_USE_NMS)).lower() == "true",
            iou_threshold=float(infer_params.get("iou_threshold", DEFAULT_IOU_THRESHOLD)),
            iou_with_min=str(infer_params.get("iou_with_min", DEFAULT_IOU_WITH_MIN)).lower() == "true"
        )

        super().__init__(
            model,
            postprocessor=postprocessor,
            handler=handler,
            **infer_params
        )

import os
from typing import Any, Callable

from synap.postprocessor import Classifier

from .base import SynapBasePipeline
from ..constants import (
    DEFAULT_TOP_N
)

__all__ = [
    "SynapImageClassificationPipeline",
]


class SynapImageClassificationPipeline(SynapBasePipeline):
    """
    Classification pipeline using SyNAP preprocessor and postprocessor.

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
        postprocessor: Classifier = Classifier(
            top_count=int(infer_params.get("top_n", DEFAULT_TOP_N)),
        )

        super().__init__(
            model,
            postprocessor=postprocessor,
            handler=handler,
            **infer_params
        )

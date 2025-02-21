import os
from dataclasses import dataclass, replace
from enum import Enum, auto

__all__ = [
    "DataType",
    "PipelineParameters",
]


class DataType(Enum):
    AUDIO = auto()
    IMAGE = auto()
    NP_ARRAY = auto()
    VIDEO = auto()
    VID_CAM = auto()
    VID_FILE = auto()
    VID_RTSP = auto()
    INVALID = auto()


@dataclass
class PipelineParameters:
    task: str | None = None
    input_src: os.PathLike | None = None
    model: os.PathLike | None = None
    confidence: float | None = None
    skip_frames: int | None = None

    def __or__(self, other: 'PipelineParameters') -> 'PipelineParameters':
        if not isinstance(other, PipelineParameters):
            return NotImplemented

        return replace(self,
            input_src=self.input_src or other.input_src,
            model=self.model or other.model,
            confidence=self.confidence if self.confidence is not None else other.confidence,
            skip_frames=self.skip_frames if self.skip_frames is not None else other.skip_frames
        )
    
    @classmethod
    def from_json(cls, json_params: dict) -> 'PipelineParameters':
        return cls(
            json_params.get("input_src", None),
            json_params.get("model", None),
            json_params.get("confidence", None),
            json_params.get("skip_frames", None),
        )

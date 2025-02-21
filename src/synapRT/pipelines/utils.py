from enum import Enum


class PipelineState(Enum):
    INIT = 0
    RUNNING = 1
    FINISHED = 2
    ERROR = 3
    ABORTED = 4
    PAUSED = 5
    PENDING_RESULT = 6

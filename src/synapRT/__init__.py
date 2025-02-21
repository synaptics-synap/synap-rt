__version__ = "0.0.1"

from . import constants
from . import pipelines
from . import utils

__all__ = [
    "__version__",
    "__doc__",
    "constants",
    "pipelines",
    "utils",
]

import importlib.resources as _resources
import json as _json
import logging as _logging
import logging.config as _logging_config

_logger = _logging.getLogger(__name__)
_logger.addHandler(_logging.NullHandler())
_log_config = _resources.files("synapRT") / "log_config.json"
with _log_config.open("r") as __f:
    _logging_config.dictConfig(_json.load(__f))

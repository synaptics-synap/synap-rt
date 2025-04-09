import os
import time
import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Generator

import numpy as np
from synap import Network, Tensors
from synap.preprocessor import Preprocessor
from synap.postprocessor import Classifier, ClassifierResult, Detector, DetectorResult
from synap.types import Layout, Rect, Shape

from .runners import *
from .utils import PipelineState
from ..constants import (
    DEFAULT_EN_PROFILING,
    DEFAULT_PROFILE_WINDOW,
    DEFAULT_SKIP_FRAMES,
    DEFAULT_VALID_INPUT_TYPES
)
from ..utils.datatypes import DataType
from ..utils.input import (
    check_model_file,
    get_input_type,
    get_model_input_dims
)

__all__ = [
    "BasePipeline",
    "SynapBasePipeline",
]


class BasePipeline(ABC):
    """
    Abstract base class for an inference pipeline.

    Must be subclassed for specific inference tasks.
    Subclasses must implement `preprocesses()` and `postprocesses()`

    :param model: Path to SyNAP model file
    :type model: os.PathLike
    :param handler: An optional callback function to handle post-processed inference results
    :type handler: Callable[[dict[str, Any], float | None], Any], None
    :param valid_input_types: Tuple of valid input types for the pipeline runner
    :type valid_input_types: tuple[type]
    :param profile: Enable profiling of inference time
    :type profile: bool
    :param profile_window: Number of inference times to keep for profiling
    :type profile_window: int
    :param runner_params: Additional parameters for the pipeline runner
    :type runner_params: Any
    """

    def __init__(
        self,
        model: os.PathLike,
        handler: Callable[[dict[str, Any], float | None], Any] | None,
        valid_input_types: tuple[type],
        profile: bool,
        profile_window: int,
        **runner_params: Any
    ):
        self._model = model
        self._handler = handler
        self._valid_input_types = valid_input_types
        self._profile = profile
        self._profile_window = profile_window
        self._runner_params = runner_params

        self._name: str = f"{self.__class__.__name__}@{id(self)}"
        self._model_inp_width: int | None = None
        self._model_inp_height: int | None = None
        self._infer_times: deque = deque(maxlen=self._profile_window)
        self._mean_inference_time: int = 0
        self._network: Network | None = None
        self._runner: BaseRunner | None = None

        self._inputs_info: list[tuple[str | os.PathLike, DataType]] = []
        self._inputs_data_type: DataType = DataType.INVALID
        self._error: Exception | None = None
        self._results: Any | None = None
        self._state: PipelineState = PipelineState.INIT
        self._lock: threading.RLock = threading.RLock()

    @property
    def error(self) -> Exception | None:
        return self._error
    
    @property
    def finished(self) -> bool:
        return self._state == PipelineState.FINISHED

    @property
    def inputs(self) -> Tensors:
        return self._network.inputs

    @property
    def outputs(self) -> Tensors:
        return self._network.outputs
    
    @property
    def inference_time(self) -> float:
        return self._mean_inference_time
    
    @property
    def state(self) -> PipelineState:
        return self._state
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def runner(self) -> BaseRunner | None:
        return self._runner
    
    @runner.setter
    def runner(self, runner: BaseRunner) -> None:
        self._runner = runner

    def __call__(self, *inputs: Any) -> None:
        """
        Run the pipeline with the given inputs.

        :param inputs: Variable length argument list of inputs to the pipeline.
                       Each input must be of a valid data type.
        :type inputs: Any
        :raises RuntimeError: If the pipeline is already running
        :raises Exception: Raises any exception that occurred while running the pipeline if called from the main thread
        """
        with self._lock:
            if self._state in (PipelineState.RUNNING, PipelineState.PAUSED):
                raise RuntimeError(f"Pipeline '{self.name}' is already running")

            self._inputs_info.clear()
            self._results = None
            self._error = None

        try:
            with self._lock:
                if not self._network:
                    self._load_network()
            self._validate_inputs(*inputs)
            self._init_runner()
            self._execute_pipeline()
            with self._lock:
                self._state = PipelineState.PENDING_RESULT if self._results else PipelineState.FINISHED
        except Exception as e:
            with self._lock:
                self._error = e
            if threading.current_thread() is threading.main_thread():
                raise e
    
    def __iter__(self) -> Generator[tuple[PipelineState, Any | None], None, None]:
        while self.state != PipelineState.FINISHED:
            yield self.poll()

    def _calc_inference_time(self):
        """
        Calculate the mean inference time over the profiling window.
        """
        window_len = len(self._infer_times)
        self._mean_inference_time = (sum(self._infer_times) / window_len) * 1000
            
    def _execute_pipeline(self) -> None:
        """
        Execute the pipeline runner.

        Runner errors are captured and re-raised as RuntimeError.

        :raises RuntimeError: If the pipeline runner is not initialized
        :raises RuntimeError: If an error occurs while running the pipeline
        """
        if not self._runner:
            self._raise_with_lock(RuntimeError("Pipeline runner not initialized"), PipelineState.ABORTED)
        try:
            with self._lock:
                self._state = PipelineState.RUNNING
            self._runner.run()
        except (TypeError, ValueError, RuntimeError) as e:
            self._raise_with_lock(RuntimeError(f"Error while running pipeline '{self.name}': {e}"), PipelineState.ERROR, e)


    def _infer(self, data: list) -> None:
        """
        Run inference on the given data.

        The preprocessor method must return a list of numpy arrays (where length of list must match number of network inputs), 
        or directly assign input data to the network.

        The postprocessor method must accept a `synap.Tensors` object as input.
        
        :param data: Input data for inference, list length must match the number of model inputs
        :type data: list
        :raises RuntimeError: If the model is not loaded
        :raises RuntimeError: If the preprocessed inputs is not a list of numpy arrays
        """
        if not self._network:
            self._raise_with_lock(RuntimeError("Fatal: Model not loaded"), PipelineState.ERROR)

        ts = time.time()
        inputs = self.preprocess(data)
        if not inputs:
            outputs = self._network.predict()
        else:
            if isinstance(inputs, list) and all(isinstance(inp, np.ndarray) for inp in inputs):
                outputs = self._network.predict(inputs)
            else:
                self._raise_with_lock(RuntimeError("Preprocessed inputs must be a list of numpy arrays"), PipelineState.ERROR)
        self.postprocess(outputs)

        with self._lock:
            self._infer_times.append(time.time() - ts)
            if self._profile and len(self._infer_times) > 0:
                self._calc_inference_time()

            if self._handler:
                self._handler(self._results, self._mean_inference_time if self._profile else None)

    def _init_runner(self) -> None:
        """
        Initialize the pipeline runner.

        :raises NotImplementedError: If the input data type is valid but not yet supported
        :raises TypeError: If the input data type is invalid
        """
        if isinstance(self._runner, BaseRunner):
            return
        if self._inputs_data_type == DataType.AUDIO:
            self._runner = GstAudioRunner(
                self._inputs_info,
                self._infer,
                sample_rate=self._runner_params.get("sample_rate"),
                chunk_duration=self._runner_params.get("chunk_duration")
            )
        elif self._inputs_data_type == DataType.NP_ARRAY:
            self._raise_with_lock(NotImplementedError("Numpy array input not supported yet"), PipelineState.ABORTED)
        elif self._inputs_data_type == DataType.IMAGE:
            self._runner = ImageRunner(
                self._inputs_info, self._infer
            )
        elif self._inputs_data_type == DataType.VIDEO:
            self._runner = GstVideoRunner(
                self._inputs_info,
                self._infer,
                self._model_inp_width,
                self._model_inp_height,
                skip_frames = self._runner_params.get("skip_frames")
            )
        else:
            self._raise_with_lock(TypeError(f"Invalid input data type '{self._inputs_data_type}'"), PipelineState.ABORTED)
    
    def _load_network(self) -> None:
        """
        Load the SyNAP model and initialize the network.
        
        :raises RuntimeError: If the model file is invalid or missing metadata.
        """
        if not check_model_file(self._model):
            self._raise_with_lock(RuntimeError(f"Fatal: Invalid SyNAP model"), PipelineState.ABORTED)
        if not (model_inp_dims := get_model_input_dims(self._model)):
            self._raise_with_lock(RuntimeError(f"Fatal: Invalid SyNAP model"), PipelineState.ABORTED)
        self._model_inp_width, self._model_inp_height = model_inp_dims
        try:
            self._network = Network(self._model)
        except RuntimeError as e:
            self._raise_with_lock(RuntimeError(f"Fatal: Failed to load model: {e}"), PipelineState.ABORTED, e)

    def _raise_with_lock(self, e: Exception, state: PipelineState, cause: Exception | None = None) -> None:
        """
        Thread safe method to raise an exception and set the pipeline state.

        :param e: Exception to raise
        :type e: Exception
        :param state: Pipeline state to set
        :type state: PipelineState
        :param cause: Optional cause of the exception
        :type cause: Exception, optional
        :raises e: Raises the given exception
        """
        if isinstance(cause, Exception):
            e.__cause__ = cause
        with self._lock:
            self._error = e
            self._state = state
        raise e

    def _validate_inputs(self, *inputs: Any) -> None:
        """
        Validate pipeline inputs.

        :param inputs: Variable length argument list of inputs to the pipeline.
                       Each input must be of a valid data type.
        :type inputs: Any
        :raises ValueError: If no valid inputs are provided, or if there is a mix of input types.
        :raises TypeError: If an input type is invalid
        """
        for input in inputs:
            if not isinstance(input, self._valid_input_types):
                self._raise_with_lock(TypeError(f"Invalid input type '{type(input)}' for {self.name} pipeline"), PipelineState.ABORTED)
            input_type = get_input_type(input)
            if input_type == DataType.INVALID:
                self._raise_with_lock(ValueError(f"Invalid input source '{input}'"), PipelineState.ABORTED)
            else:
                self._inputs_info.append((input, input_type))
        if not self._inputs_info:
            self._raise_with_lock(ValueError(f"No valid inputs for pipeline '{self.name}'"), PipelineState.ABORTED)
        
        input_data_types: list[DataType] = [info[1] for info in self._inputs_info]
        if all(t in (DataType.AUD_FILE, DataType.AUD_MIC) for t in input_data_types):
            self._inputs_data_type = DataType.AUDIO
        elif all(t == DataType.NP_ARRAY for t in input_data_types):
            self._inputs_data_type = DataType.NP_ARRAY
        elif all(t == DataType.IMAGE for t in input_data_types):
            self._inputs_data_type = DataType.IMAGE
        elif all(t in (DataType.VID_CAM, DataType.VID_FILE, DataType.VID_RTSP) for t in input_data_types):
            self._inputs_data_type = DataType.VIDEO
        else:
            self._raise_with_lock(ValueError("Pipeline cannot have a mix of input types"), PipelineState.ABORTED)

    def pause(self) -> None:
        """
        Pause the pipeline.
        """
        with self._lock:
            self._runner.pause()
            self._state = PipelineState.PAUSED

    def poll(self) -> tuple[PipelineState, Any | None]:
        """
        Return pipeline state and last inference results (if any).

        :return: Tuple of running status and inference results
        :rtype: tuple[PipelineState, Any | None]
        """
        if self._state in (PipelineState.ABORTED, PipelineState.ERROR) and threading.current_thread() is threading.main_thread():
            raise self._error or RuntimeError("Unknown pipeline error")
        with self._lock:
            if self._state == PipelineState.PENDING_RESULT:
                self._state = PipelineState.FINISHED
            return self._state, self._results
        
    def resume(self) -> None:
        """
        Resume the pipeline.
        """
        with self._lock:
            self._runner.resume()
            self._state = PipelineState.RUNNING

    def stop(self) -> None:
        """
        Stop the pipeline.
        """
        with self._lock:
            self._runner.stop()
            self._state = PipelineState.PENDING_RESULT if self._results else PipelineState.FINISHED

    def test(self, *sample_data: list[np.ndarray]) -> None:
        """
        Run inference on sample data.

        :param sample_data: Sample data(s) for inference
        :type sample_data: list[np.ndarray]
        """
        for data in sample_data:
            self._infer(data)

    def update_model(self, model: os.PathLike) -> None:
        """
        Update SyNAP model used in the pipeline.
        
        :param model: Path to new SyNAP model file
        """
        self._model = model
        with self._lock:
            self._load_network()

    @abstractmethod
    def preprocess(self, data: list, **kwargs: Any) -> None | list[np.ndarray]:
        """
        Preprocess the input data before inference.

        :param data: Input data for inference, list length must match the number of model inputs
        :type data: list
        :param kwargs: Additional parameters for preprocessing
        :type kwargs: Any
        :return: (optional) Preprocessed input data, list length must match the number of model inputs
        :rtype: None | list[np.ndarray]
        """
        ...

    @abstractmethod
    def postprocess(self, outputs: Tensors, **kwargs: Any) -> None:
        """
        Postprocess the inference outputs.

        :outputs: Inference outputs
        :type outputs: Tensors
        :param kwargs: Additional parameters for postprocessing
        :type kwargs: Any
        """
        ...


class SynapBasePipeline(BasePipeline):
    """
    Base class for an inference pipeline using SyNAP preprocessor and postprocessor.
    
    :param model: Path to SyNAP model file
    :type model: os.PathLike
    :param postprocessor: SyNAP postprocessor instance
    :type postprocessor: Classifier | Detector
    :param handler: An optional callback function to handle post-processed inference results
    :type handler: Callable[[dict[str, Any], float | None], Any], optional
    :param valid_input_types: Tuple of valid input types for the pipeline runner
    :type valid_input_types: tuple[type], optional
    :param infer_params: Additional parameters for the inference pipeline
    :type infer_params: Any
    """
    
    def __init__(
        self,
        model: os.PathLike,
        postprocessor: Classifier | Detector,
        handler: Callable[[dict[str, Any], float | None], Any] | None = None,
        valid_input_types: tuple[type] | None = None,
        **infer_params: Any
    ):
        super().__init__(
            model,
            handler=handler,
            valid_input_types=valid_input_types or DEFAULT_VALID_INPUT_TYPES,
            profile=str(infer_params.get("profile", DEFAULT_EN_PROFILING)).lower() == "true",
            profile_window=int(infer_params.get("profile_window", DEFAULT_PROFILE_WINDOW)),
            skip_frames=int(infer_params.get("skip_frames", DEFAULT_SKIP_FRAMES))
        )

        if not isinstance(postprocessor, (Classifier, Detector)):
            self._raise_with_lock(TypeError(f"Invalid SyNAP postprocessor type '{type(postprocessor)}'"), PipelineState.ABORTED)
        self.postprocessor: Classifier | Detector = postprocessor
        self.preprocessor: Preprocessor = Preprocessor()
        self._assigned_rect: Rect | None = None

    def _result_to_dict(self, result: ClassifierResult | DetectorResult) -> dict:
        """
        Convert ClassifierResult or DetectorResult to a serializable dictionary.

        :param result: Postprocessed result, must be an instance of ClassifierResult or DetectorResult
        :raises TypeError: If the result type is invalid
        """
        if isinstance(result, ClassifierResult):
            return {
                "top_n": [
                    {
                        "class_index": item.class_index,
                        "confidence": item.confidence
                    }
                    for item in result.items
                ]
            }
        elif isinstance(result, DetectorResult):
            return {
                "items": [
                    {
                        "confidence": item.confidence,
                        "class_index": item.class_index,
                        "bounding_box": {
                            "origin": {
                                "x": item.bounding_box.origin.x,
                                "y": item.bounding_box.origin.y
                            },
                            "size": {
                                "x": item.bounding_box.size.x,
                                "y": item.bounding_box.size.y
                            }
                        },
                        "landmarks": [(lm.x, lm.y, lm.z, lm.visibility) for lm in item.landmarks],
                    }
                    for item in result.items
                ]
            }
        else:
            self._raise_with_lock(TypeError(f"Invalid postprocessed result type '{type(result)}'"), PipelineState.ERROR)

    def preprocess(self, data: list) -> None:
        """
        Preprocess the input data before inference using SyNAP preprocessor.

        Input data can be a list of numpy arrays or file paths.

        :param data: Input data for inference, list length must match the number of model inputs
        :type data: list
        """

        for i in range(len(self.inputs)):
            inp_data = data[i]
            if isinstance(inp_data, (os.PathLike, str)):
                self._assigned_rect = self.preprocessor.assign(
                    self.inputs, inp_data, input_index=i
                )
            else:
                if inp_data.ndim == len(list(self.inputs[i].shape)) - 1:
                    data_shape = Shape([1, *inp_data.shape])
                else:
                    data_shape = Shape(inp_data.shape)
                self._assigned_rect = self.preprocessor.assign(
                    self.inputs, inp_data, data_shape, Layout.nhwc, input_index=i
                )

    def postprocess(self, outputs: Tensors) -> None:
        """
        Postprocess the inference outputs using SyNAP postprocessor.
        
        :param outputs: Inference outputs
        :type outputs: Tensors
        :raises TypeError: If the postprocessor type is invalid
        """

        if isinstance(self.postprocessor, Classifier):
            results = self.postprocessor.process(outputs)
        elif isinstance(self.postprocessor, Detector):
            results = self.postprocessor.process(outputs, self._assigned_rect)
        else:
            self._raise_with_lock(TypeError(f"Invalid postprocessor type '{type(self.postprocessor)}'"), PipelineState.ERROR)
        self._results = self._result_to_dict(results)


if __name__ == "__main__":
    pass

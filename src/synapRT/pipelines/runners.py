from abc import ABC, abstractmethod
from signal import SIGINT
from typing import Any, Callable
import logging
import os

import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

from ..constants import DEFAULT_SKIP_FRAMES
from ..utils.datatypes import DataType
from ..utils.input import get_camera_devices
from ..utils.gst import bus_call, handle_sigint, get_gst_elems

__all__ = [
    "BaseRunner",
    "GstVideoRunner",
    "ImageRunner"
]

logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    """
    Abstract base class for inference runners.

    :param inputs_info: List of tuples containing input data and its type
    :type inputs_info: list[tuple[Any, DataType]]
    :param infer_func: Inference function to run on input data
    :type infer_func: Callable[[list[Any]], None]
    """

    def __init__(
        self,
        inputs_info: list[tuple[Any, DataType]],
        infer_func: Callable[[list[Any]], None]
    ):
        self._inputs_info = inputs_info
        self._infer_func = infer_func

    @abstractmethod
    def process_inputs(self) -> None:
        """
        Process input data and prepare for inference
        """
        ...

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the inference runner
        """
        ...

    @abstractmethod
    def pause(self) -> None:
        """
        Pause the inference runner
        """
        ...

    @abstractmethod
    def resume(self) -> None:
        """
        Resume the inference runner
        """
        ...

    @abstractmethod
    def run(self) -> None:
        """
        Start the inference runner
        """
        self.process_inputs()
        self.initialize()

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the inference runner
        """
        ...


class GstVideoRunner(BaseRunner):
    """
    GStreamer-based video inference runner.
    
    :param inputs_info: List of tuples containing input data and its type
    :type inputs_info: list[tuple[str | os.PathLike, DataType]]
    :param infer_func: Inference function to run on input data
    :type infer_func: Callable[[list[Any]], None]
    :param model_inp_width: Model input width
    :type model_inp_width: int
    :param model_inp_height: Model input height
    :type model_inp_height: int
    :param skip_frames: Number of frames to skip between inference
    :type skip_frames: int, optional
    """

    def __init__(
        self,
        inputs_info: list[tuple[str | os.PathLike, DataType]],
        infer_func: Callable[[list[Any]], None],
        model_inp_width: int,
        model_inp_height: int,
        skip_frames: int | None = None
    ):
        super().__init__(inputs_info, infer_func)

        self._model_inp_width = model_inp_width
        self._model_inp_height = model_inp_height
        self._skip_frames = skip_frames or DEFAULT_SKIP_FRAMES

        self._inf_skip_counter: int = self._skip_frames
        self._pipeline_str: str | None = None
        self._pipeline: Gst.Element | None = None
        self._bus_watch_ids: int = 0
        self._main_loop: Gst.Element | None = None
    
    def _cleanup(self) -> None:
        """
        Clean up GStreamer pipeline and exit the program.
        """
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)

        if self._bus_watch_id > 0:
            if GLib.Source.remove(self._bus_watch_id):
                self._bus_watch_id = 0

        if self._main_loop:
            self._main_loop.quit()
    
    def _on_new_sample(self, app_sink: Gst.Element) -> Gst.FlowReturn:
        """
        Callback function for new samples from GStreamer appsink.
        
        :param app_sink: GStreamer appsink element
        :type app_sink: Gst.Element
        :return: GStreamer flow return status
        :rtype: Gst.FlowReturn
        """
        if self._inf_skip_counter > 0:
            self._inf_skip_counter -= 1
            return Gst.FlowReturn.OK
        
        self._inf_skip_counter = self._skip_frames
        sample = app_sink.emit("pull-sample")
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        height = structure.get_value("height")
        width = structure.get_value("width")
        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            raise RuntimeError("Error: Could not map buffer data")

        data = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data)

        buffer.unmap(map_info)

        try:
            self._infer_func([data])
        except RuntimeError as e:
            logger.error(f"Fatal: Inference failed: {e}")
            return Gst.FlowReturn.ERROR

        return Gst.FlowReturn.OK

    def process_inputs(self) -> None:
        """
        Process input and prepare for inference.
        
        :raises TypeError: If non-video input is received
        :raises ValueError: If multiple inputs are received
        :raises ValueError: If input is "cam" and no available cameras are detected
        """
        if len(self._inputs_info) > 1:
            raise ValueError("Video runner does not support multiple inputs")
        input, input_type = self._inputs_info[0]
        if input_type not in (DataType.VID_CAM, DataType.VID_FILE, DataType.VID_RTSP):
            raise TypeError(f"Non-video input '{input}' received in video runner")
        else:
            if input == "cam":
                cams = get_camera_devices()
                try:
                    input = cams.pop()
                except IndexError:
                    raise ValueError(
                        "Received 'cam' input but no available cameras detected"
                    )
            self._pipeline_str = get_gst_elems(
                input, input_type, self._model_inp_width, self._model_inp_height
            )

    def initialize(self) -> None:
        """
        Initialize the GStreamer pipeline.
        
        :raises RuntimeError: If pipeline initialization fails
        """
        Gst.init(None)
        self._main_loop = GLib.MainLoop()

        appsink_name = f"infer_sink"
        # TODO: Add low buffer `! queue` before appsink for smoother performance
        pipeline_str_full = f"{self._pipeline_str} ! appsink name={appsink_name}"
        self._pipeline = Gst.parse_launch(pipeline_str_full)
        if not self._pipeline:
            self._cleanup()
            raise RuntimeError(
                f"Fatal: Failed to initialize GStreamer pipeline:\n\"{pipeline_str_full}\""
            )
        
        bus = self._pipeline.get_bus()
        self._bus_watch_id = bus.add_watch(GLib.PRIORITY_DEFAULT, bus_call, self._main_loop)

        appsink = self._pipeline.get_by_name(appsink_name)
        if not appsink:
            self._cleanup()
            raise RuntimeError(
                f"Fatal: Failed to get appsink for pipeline:\n\"{pipeline_str_full}\""
            )
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", True)
        appsink.connect("new-sample", self._on_new_sample)
    
        GLib.unix_signal_add(
            GLib.PRIORITY_HIGH, int(SIGINT), handle_sigint, self._main_loop, self._pipeline
        )

        self._pipeline.set_state(Gst.State.PLAYING)
        ret, state, _ = self._pipeline.get_state(timeout=5 * Gst.SECOND)
        if ret == Gst.StateChangeReturn.FAILURE or state != Gst.State.PLAYING:
            logger.error(f"Error: Failed to set pipeline to PLAYING. Current state: {state}")
            self._cleanup()
            raise RuntimeError(f"Fatal: Failed to start GStreamer pipeline")
        
    def pause(self):
        """
        Pause the GStreamer pipeline if pipeline is valid.

        Stops pipeline and cleans up resources if pipeline fails to pause.
        """
        if self._pipeline:
            self._pipeline.set_state(Gst.State.PAUSED)
            ret, state, _ = self._pipeline.get_state(timeout=5 * Gst.SECOND)
            if ret == Gst.StateChangeReturn.FAILURE or state != Gst.State.PAUSED:
                logger.error(f"Error: Failed to set pipeline to PAUSED. Current state: {state}")
                self._cleanup()
                raise RuntimeError(f"Fatal: Failed to pause GStreamer pipeline")

    def resume(self):
        """
        Resume the GStreamer pipeline if pipeline is valid.

        Stops pipeline and cleans up resources if pipeline fails to resume.
        """
        if self._pipeline:
            self._pipeline.set_state(Gst.State.PLAYING)
            ret, state, _ = self._pipeline.get_state(timeout=5 * Gst.SECOND)
            if ret == Gst.StateChangeReturn.FAILURE or state != Gst.State.PLAYING:
                logger.error(f"Error: Failed to set pipeline to PLAYING. Current state: {state}")
                self._cleanup()
                raise RuntimeError(f"Fatal: Failed to resume GStreamer pipeline")
            

    def run(self) -> None:
        """
        Start the GStreamer pipelines.
        """
        super().run()
        self._main_loop.run()
        self._cleanup()

    def stop(self) -> None:
        self._cleanup()


class ImageRunner(BaseRunner):
    """
    Image inference runner.

    :param inputs_info: List of tuples containing input data and its type
    :type inputs_info: list[tuple[os.PathLike, DataType]]
    :param infer_func: Inference function to run on input data
    :type infer_func: Callable[[list[np.ndarray]], None]
    """

    def __init__(
        self,
        inputs_info: list[tuple[os.PathLike, DataType]],
        infer_func: Callable[[list[np.ndarray]], None]
    ):
        super().__init__(inputs_info, infer_func)

        self._images: list[os.PathLike] = []

    def process_inputs(self) -> None:
        """
        Process input and prepare for inference.
        
        :raises TypeError: If non-image input is received
        :raises ValueError: If no valid image inputs are received
        """
        for input_info in self._inputs_info:
            input, input_type = input_info
            if input_type != DataType.IMAGE:
                raise TypeError(f"Non-image input '{input}' received in image runner")
            self._images.append(input)
        if not self._images:
            raise ValueError(f"No valid image inputs")

    def initialize(self) -> None:
        pass

    def pause(self) -> None:
        pass
    
    def resume(self) -> None:
        pass

    def run(self) -> None:
        """
        Start the image inference runner.
        """
        super().run()
        for image in self._images:
            self._infer_func([image])

    def stop(self) -> None:
        pass

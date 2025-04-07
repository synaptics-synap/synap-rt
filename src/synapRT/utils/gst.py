import logging
import sys

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

from .datatypes import DataType

__all__ = [
    "bus_call",
    "handle_sigint",
    "get_video_input_elems",
    "get_video_pre_elems",
]

logger = logging.getLogger(__name__)


def handle_sigint(loop: GLib.MainLoop, pipeline: Gst.Element) -> bool:
    """
    Handle interrupt signal by terminating GStreamer mainloop
    :param loop: GStreamer mainloop
    :type loop: GLib.MainLoop
    :param pipeline: GStreamer pipeline
    :type pipeline: Gst.Element
    :return: `False` to remove the signal handler.
    :rtype: bool
    """

    logger.info("Caught Ctrl+C, stopping playback")
    if pipeline is not None:
        pipeline.set_state(Gst.State.NULL)
    loop.quit()
    return GLib.SOURCE_REMOVE


def bus_call(bus: Gst.Bus, msg: Gst.Message, loop: GLib.MainLoop) -> bool:
    """
    Bus callback function.

    :param bus: GStreamer bus from which the Gst.Message originated
    :type bus: Gst.Bus
    :param msg: Message to handle
    :type msg: Gst.Message
    :param loop: GStreamer mainloop
    :type loop: GLib.MainLoop
    :return: True to keep this callback active
    :rtype: bool
    """

    msg_type = msg.type
    if msg_type == Gst.MessageType.EOS:
        logger.info("GStreamer: End of stream, stopping playback")
        loop.quit()
    elif msg_type == Gst.MessageType.WARNING:
        warn, debug = msg.parse_warning()
        logger.warning(f"GStreamer Warning: {warn}, debug: {debug}")
    elif msg_type == Gst.MessageType.ERROR:
        error, debug = msg.parse_error()
        logger.error(f"GStreamer Error: {error}, debug: {debug}", file=sys.stderr)
        loop.quit()

    return True


def get_audio_elems(input: str, input_type: DataType, chunk_duration: float, sample_rate: int, channels: int, sample_width: int) -> str:
    """
    Get suitable GStreamer elements for audio inputs.
    
    :param input: Audio input source (microphone or file)
    :type input: str
    :param input_type: Audio input source type (microphone or file)
    :type input_type: DataType
    :param chunk_duration: Duration of audio chunks (seconds)
    :type chunk_duration: float
    :param sample_rate: Audio sample rate (Hz)
    :type sample_rate: int
    :param channels: Number of audio channels
    :type channels: int
    :param sample_width: Audio sample width (bytes)
    :type sample_width: int
    :return: GStreamer elements for audio input source
    :rtype: str
    """

    if input_type == DataType.AUD_MIC:
        inp_elem = f"alsasrc device={input} buffer-time={int(chunk_duration * 1e6)}"
    else:
        blocksize = int(sample_rate * chunk_duration * channels * sample_width)
        inp_elem = f"filesrc location={input} blocksize={blocksize}"
    return f"{inp_elem} ! decodebin ! audioconvert ! audioresample " \
        f"! audio/x-raw,format=S16LE,channels={int(channels)},rate={int(sample_rate)}"


def get_video_input_elems(input: str, input_type: DataType) -> str:
    """
    Get suitable GStreamer elements based on the input source and type.
    
    :param input: Input source
    :type input: str
    :param input_type: Input source type
    :type input_type: DataType
    :return: GStreamer elements for the input source
    :rtype: str
    """

    if input_type == DataType.VID_CAM:
        return f"v4l2src device={input} ! video/x-raw,framerate=30/1,format=YUY2,width=640,height=480"
    elif input_type == DataType.VID_FILE:
        return f"filesrc location={input} ! qtdemux name=demux demux.video_0 ! h264parse ! avdec_h264"
    elif input_type == DataType.VID_RTSP:
        return f"rtspsrc location={input} latency=2000 ! rtpjitterbuffer ! rtph264depay wait-for-keyframe=true ! video/x-h264"


def get_video_pre_elems(input: str, input_type: DataType, model_inp_width: int, model_inp_height: int) -> str:
    """
    Get GStreamer elements for preprocessing the input source.
    
    :param input: Input source
    :type input: str
    :param input_type: Input source type
    :type input_type: DataType
    :param model_inp_width: Model input width
    :type model_inp_width: int
    :param model_inp_height: Model input height
    :type model_inp_height: int
    :return: GStreamer elements for preprocessing the input source
    :rtype: str
    """

    return f"{get_video_input_elems(input, input_type)}" \
        "! videoconvert " \
        "! videoscale " \
        f"! video/x-raw,width={model_inp_width},height={model_inp_height},format=RGB "
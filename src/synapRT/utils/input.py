import json
import logging
import mimetypes
import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import gi
gi.require_version("GUdev", "1.0")
from gi.repository import GUdev
from synap import Network, Tensor
from synap.types import Layout

from .datatypes import DataType, PipelineParameters
from ..constants._internal import MODEL_META_FILE

__all__ = [
    "check_model_file",
    "get_camera_devices",
    "get_input_type",
    "get_model_input_dims",
    "parse_params_file",
]

logger = logging.getLogger(__name__)

def _is_valid_image(file: os.PathLike) -> bool:
    if not file:
        return False
    try:
        img = cv2.imread(str(file))
        return img is not None
    except Exception as e:
        logger.error(f"Error: Unable to read image file '{file}': {e}")
        return False


def _is_valid_audio(file: os.PathLike) -> bool:
    if not file:
        return False
    try:
        subprocess.run(
            [
                "gst-launch-1.0",
                "filesrc", f"location={file}",
                "!", "decodebin",
                "!", "fakesink"
            ],
            capture_output=True,
            text=True, 
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: Unable to read audio file '{file}': {e.stderr}")
        return False


def _is_valid_video(file: os.PathLike) -> bool:
    if not file:
        return False
    try:
        cap = cv2.VideoCapture(file)
        valid = cap.isOpened()
        cap.release()
        return valid
    except Exception as e:
        logger.error(f"Error: Unable to read video file '{file}': {e}")
        return False


def check_model_file(model_file: os.PathLike) -> bool:
    if not model_file:
        logger.error("Error: SyNAP model not provided")
        return False
    if not Path(model_file).exists():
        logger.error(f"Error: SyNAP model '{model_file}' not found")
        return False
    model_file: Path = Path(model_file).resolve()
    try:
        subprocess.run([
            "synap_cli", "-m", str(model_file), "random"
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: Invalid SyNAP model '{model_file}': {e.stderr.decode()}")
        return False
    return True


def get_camera_devices(cam_subsys: str = "video4linux") -> list[str]:
    camera_paths: list[str] = []
    client = GUdev.Client(subsystems=[cam_subsys])
    devices = client.query_by_subsystem(cam_subsys)

    for device in devices:
        bus = device.get_property("ID_BUS")
        if bus == "usb":
            sys_path = device.get_sysfs_path()
            if sys_path:
                index_path = f"{sys_path}/index"
                try:
                    with open(index_path, "r") as f:
                        contents = f.read().strip()
                        index_val = int(contents)

                        if index_val == 0:
                            dev_node = device.get_device_file()
                            if dev_node:
                                camera_paths.append(dev_node)
                except OSError as e:
                    logger.warning(f"Warning: Error reading {index_path}: {e}")
                except ValueError:
                    logger.warning(f"Warning: Unexpected contents in {index_path}: {contents}")

    return camera_paths


def get_microphone_devices(connector: str = "usb") -> list[str]:
    mic_devices = []
    client = GUdev.Client(subsystems=["sound"])
    devices = client.query_by_subsystem("sound")

    for device in devices:
        dev_node = device.get_device_file()
        if not dev_node:
            continue
        # match PCM capture devices: /dev/snd/pcmC{card}D{device}c
        match = re.search(r"pcmC(\d+)D(\d+)c", dev_node)
        if not match:
            continue
        sys_path = device.get_sysfs_path()
        if sys_path and connector not in sys_path.lower():
            continue
        card_idx, device_idx = int(match.group(1)), int(match.group(2))
        mic_devices.append(f"plughw:{card_idx},{device_idx}")

    return mic_devices


def get_input_type(input_src: str | os.PathLike) -> DataType:
    input_src: str = str(input_src) if input_src else None
    if not input_src:
        logger.error(f"Error: Input source not provided")
        return DataType.INVALID
    
    if isinstance(input_src, np.ndarray):
        return DataType.NP_ARRAY

    if input_src.startswith("/dev/video") or input_src == "cam":
        return DataType.VID_CAM
    elif input_src.startswith("rtsp://"):
        return DataType.VID_RTSP
    elif input_src == "mic":
        return DataType.AUD_MIC

    mime_type, _ = mimetypes.guess_type(input_src)
    if mime_type:
        if mime_type.startswith("audio") and _is_valid_audio(input_src):
            return DataType.AUD_FILE
        elif mime_type.startswith("image") and _is_valid_image(input_src):
            return DataType.IMAGE
        elif mime_type.startswith("video") and _is_valid_video(input_src):
            return DataType.VID_FILE

    logger.error(f"Error: Invalid input source {input_src}")
    return DataType.INVALID


def get_model_input_dims(model: str | os.PathLike | Network) -> dict[str, tuple[int, int]]:
    """
    Gets model input dimensions by parsing .synap file.
    """
    def _get_size(inp_tensor: Tensor) -> tuple[int, int]:
        if inp_tensor.layout == Layout.nchw:
            return inp_tensor.shape[3], inp_tensor.shape[2]
        elif inp_tensor.layout == Layout.nhwc:
            return inp_tensor.shape[2], inp_tensor.shape[1]
        else:
            raise ValueError(f"Input tensor '{inp_tensor.name}' has invalid layout (model: '{str(model)}')")

    input_sizes = {}
    if not isinstance(model, Network):
        model = Network(str(model))
    for inp in model.inputs:
        input_sizes[inp.name] = _get_size(inp)
    return input_sizes


def parse_params_file(file: os.PathLike, **params: str) -> PipelineParameters:
    with open(file, "r") as f:
        params = json.load(f)
    return PipelineParameters.from_json(params)

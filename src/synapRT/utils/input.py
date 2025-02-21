import json
import logging
import mimetypes
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import zipfile
import gi
gi.require_version("GUdev", "1.0")
from gi.repository import GUdev

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

    mime_type, _ = mimetypes.guess_type(input_src)
    if mime_type:
        if mime_type.startswith("audio") and _is_valid_audio(input_src):
            return DataType.AUDIO
        elif mime_type.startswith("image") and _is_valid_image(input_src):
            return DataType.IMAGE
        elif mime_type.startswith("video") and _is_valid_video(input_src):
            return DataType.VID_FILE

    logger.error(f"Error: Invalid input source {input_src}")
    return DataType.INVALID


def get_model_input_dims(model: str) -> tuple[int, int] | None:
    """
    Attempts to find model input dimensions by parsing .synap file.
    """
    try:
        with zipfile.ZipFile(model, "r") as mod_info:
            if MODEL_META_FILE not in mod_info.namelist():
                raise FileNotFoundError("Missing model metadata")
            with mod_info.open(MODEL_META_FILE, "r") as meta_f:
                metadata = json.load(meta_f)
                inputs = metadata["Inputs"]
                if len(inputs) > 1:
                    raise NotImplementedError("Multiple input models not supported")
                input_info = inputs[list(inputs.keys())[0]]
                if input_info["format"] == "nhwc":
                    inp_w, inp_h = input_info["shape"][2], input_info["shape"][1]
                elif input_info["format"] == "nchw":
                    inp_w, inp_h = input_info["shape"][3], input_info["shape"][2]
                else:
                    raise ValueError(
                        f"Invalid metadata: unknown format \"{input_info['format']}\""
                    )
                logger.debug(f"Extracted model input size: {inp_w}x{inp_h}")
                return inp_w, inp_h
    except (zipfile.BadZipFile, FileNotFoundError) as e:
        logger.error(f"Error: Invalid SyNAP model '{model}': {e.args[0]}")
    except KeyError as e:
        logger.error(f"Error: Missing model metadata '{e.args[0]}' for SyNAP model '{model}'")
    except (NotImplementedError, ValueError) as e:
        logger.error(f"Error: Invalid SyNAP model '{model}': {e.args[0]}")


def parse_params_file(file: os.PathLike, **params: str) -> PipelineParameters:
    with open(file, "r") as f:
        params = json.load(f)
    return PipelineParameters.from_json(params)

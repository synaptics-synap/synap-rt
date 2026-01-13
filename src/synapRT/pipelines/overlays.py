from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Callable
import logging
import math
import os

import cairo
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from synap.postprocessor import DetectorResult, DetectorResultItem

from ..utils.gst.overlay import (
    FontConfig,
    StyleConfig,
    LabelCache,
    CairoSegMaskRenderer,
    DEFAULT_COLORMAP,
)

__all__ = [
    "GstVideoOverlay",
    "SynapODOverlay",
    "overlay_factory",
]

logger = logging.getLogger(__name__)


class GstVideoOverlay(ABC):

    def __init__(
        self,
        results_provider: Callable[[], Any],
        model_inp_width: int,
        model_inp_height: int,
        fullscreen: bool = True
    ):
        self._results_provider = results_provider
        self._model_inp_width = model_inp_width
        self._model_inp_height = model_inp_height
        self._fullscreen = fullscreen
        self._disp_w = None
        self._disp_h = None
        self._pipeline = \
            f"videoconvert ! videoscale ! video/x-raw,format=BGRA" \
            f"! cairooverlay name=overlay " \
            f"! waylandsink fullscreen={str(self._fullscreen).lower()}"

        os.environ["XDG_RUNTIME_DIR"] = "/var/run/user/0"
        os.environ["WESTON_DISABLE_GBM_MODIFIERS"] = "true"
        os.environ["WAYLAND_DISPLAY"] = "wayland-1"
        os.environ["QT_QPA_PLATFORM"] = "wayland"

    @property
    def pipeline(self) -> str:
        return self._pipeline

    def on_caps_changed(
        self,
        overlay: Gst.Element,
        caps: Gst.Caps
    ):
        s = caps.get_structure(0)
        self._disp_w = s.get_int("width")[1]
        self._disp_h = s.get_int("height")[1]

    @abstractmethod
    def on_draw(
        self,
        overlay: Gst.Element,
        context: cairo.Context,
        timestamp: int,
        duration: int
    ): ...


class SynapODOverlay(GstVideoOverlay):

    # YOLO(v5/8) pose edges (17-keypoint COCO order)
    _POSE_EDGES: tuple[tuple[int,int], ...] = (
        (0, 1),     # nose - left eye
        (1, 3),     # left eye - left ear
        (3, 5),     # left ear - left shoulder
        (5, 7),     # left shoulder - left elbow
        (7, 9),     # left elbow - left wrist
        (5, 11),    # left shoulder - left hip
        (11, 13),   # left hip - left knee
        (13, 15),   # left knee - left ankle

        (0, 2),     # nose - right eye
        (2, 4),     # right eye - right ear
        (4, 6),     # right ear - right shoulder
        (6, 8),     # right shoulder - right elbow
        (8, 10),    # right elbow - right wrist
        (6, 12),    # right shoulder - right hip
        (12, 14),   # right hip - right knee
        (14, 16),   # right knee - right ankle

        (5, 6),     # left shoulder - right shoulder
        (11, 12)    # left hip - right hip
    )

    def __init__(
        self,
        results_provider: Callable[[], DetectorResult | dict[str, Any]],
        model_inp_width: int,
        model_inp_height: int,
        fullscreen: bool = True,
        font_config: FontConfig | None = None,
        style_config: StyleConfig | None = None,
        seg_colormap: dict[int, tuple[float, float, float]] | None = None
    ):
        super().__init__(
            results_provider,
            model_inp_width,
            model_inp_height,
            fullscreen
        )
        self._label_cache = LabelCache(font_config)
        self._style_config = style_config or StyleConfig()
        self._mask_renderer = CairoSegMaskRenderer()
        self._class_colors = seg_colormap or DEFAULT_COLORMAP

    def _class_rgb(self, idx: int) -> tuple[float,float,float]:
        return self._class_colors.get(idx, (0.2, 0.8, 0.4))  # default mint green

    def _render_json_rect(
        self,
        res: dict[str, Any],
        scale_x: float,
        scale_y: float,
    ) -> tuple[int, int, int, int, str]:
        bbox = res["bounding_box"]
        x = int(round(bbox["origin"]["x"] * scale_x))
        y = int(round(bbox["origin"]["y"] * scale_y))
        w = int(round(bbox["size"]["x"]   * scale_x))
        h = int(round(bbox["size"]["y"]   * scale_y))
        idx = res["class_index"]
        conf = float(res["confidence"])
        # round confidence to improve cache hits (0.05 steps)
        conf_q = round(conf * 20) / 20.0
        label = f"{idx}:{conf_q:.2f}"
        return x, y, w, h, label

    def _render_item_rect(
        self,
        res: DetectorResultItem,
        scale_x: float,
        scale_y: float,
    ) -> tuple[int, int, int, int, str]:
        b = res.bounding_box
        x = int(round(b.origin.x * scale_x))
        y = int(round(b.origin.y * scale_y))
        w = int(round(b.size.x   * scale_x))
        h = int(round(b.size.y   * scale_y))
        conf_q = round(float(res.confidence) * 20) / 20.0
        label = f"{res.class_index}:{conf_q:.2f}"
        return x, y, w, h, label

    def _extract_landmarks(self, obj: DetectorResultItem | dict[str, Any]) -> list[tuple[float,float,float | None]] | None:
        pts = None
        if isinstance(obj, dict):
            lm = obj.get("landmarks") or {}
            pts = lm.get("points") if isinstance(lm, dict) else None
            if isinstance(pts, list) and len(pts) > 0 and isinstance(pts[0], dict):
                out = []
                for p in pts:
                    out.append((float(p.get("x", 0.0)), float(p.get("y", 0.0)), p.get("score")))
                return out
        else:
            try:
                pts = obj.landmarks
                if not pts:
                    return None
                return [
                    (float(p.x), float(p.y), p.visibility)
                    for p in pts
                ]
            except Exception as e:
                logger.warning("Failed to parse pose landmarks: %s", e)
                return None
        return None

    def _extract_mask(self, obj: DetectorResultItem | dict[str, Any]) -> tuple[int,int, Iterable[float]] | None:
        try:
            if isinstance(obj, dict):
                m = obj.get("mask", None)
                if not m:
                    return None
                data = m.get("data", None)
                if not data:
                    return None
                return int(m["width"]), int(m["height"]), data
            else:
                m = obj.mask
                if not m:
                    return None
                return m.width, m.height, m.buffer()
        except Exception:
            logger.warning("Failed to parse segmentation mask data")
            return None

    def _draw_pose(
        self,
        ctx: cairo.Context,
        scale_x: float, scale_y: float,
        landmarks: list[tuple[float, float, float | None]]
    ):
        if not landmarks:
            return

        scaled: list[tuple[float, float, bool]] = []
        for (x, y, score) in landmarks:
            xd = x * scale_x
            yd = y * scale_y
            vis_ok = True
            if score is not None:
                vis_ok = float(score) > 0.3
            scaled.append((xd, yd, vis_ok))

        ctx.set_antialias(cairo.ANTIALIAS_NONE if not self._style_config.antialias else cairo.ANTIALIAS_DEFAULT)
        ctx.set_line_width(self._style_config.kp_line_w)
        ctx.set_source_rgba(*self._style_config.kp_rgba)

        for (s_idx, e_idx) in self._POSE_EDGES:
            xs, ys, vs = scaled[s_idx]
            xe, ye, ve = scaled[e_idx]
            if vs and ve:
                ctx.move_to(xs, ys)
                ctx.line_to(xe, ye)
            ctx.stroke()

        r = self._style_config.kp_radius
        ctx.set_source_rgba(*self._style_config.kp_rgba)
        ctx.new_path()
        for (x, y, v) in scaled:
            if v:
                ctx.arc(x, y, r, 0, 2 * math.pi)
                ctx.fill()

    def on_draw(
        self,
        overlay: Gst.Element,
        context: cairo.Context,
        timestamp: int,
        duration: int
    ):
        results = self._results_provider()
        if not results:
            return
        if not (self._disp_w and self._disp_h):
            logger.warning("Could not determine display dims, skipping")
            return
        if not isinstance(results, (DetectorResult, dict)):
            logger.warning("Invalid results type '%s', skipping", type(results))
            return

        items = results.items if isinstance(results, DetectorResult) else results.get("items", None)
        if not items:
            logger.debug("Empty results, skipping")
            return

        sx = self._disp_w / float(self._model_inp_width)
        sy = self._disp_h / float(self._model_inp_height)
        rects: list[tuple[int,int,int,int]] = []
        labels: list[tuple[str,int,int]] = []

        for r in items:
            if isinstance(r, DetectorResultItem):
                x, y, w, h, label = self._render_item_rect(r, sx, sy)
                cls_idx = r.class_index
            elif isinstance(r, dict):
                x, y, w, h, label = self._render_json_rect(r, sx, sy)
                cls_idx = int(r["class_index"])
            else:
                logger.warning("Invalid result object type '%s', skipping", type(r))
                continue

            # skip fully off-screen
            if x + w < 0 or y + h < 0 or x > self._disp_w or y > self._disp_h:
                continue

            m = self._extract_mask(r)
            if m is not None:
                (mw, mh, data) = m
                self._mask_renderer.add_mask(
                    mw, mh, data,
                    bbox=(x, y, w, h),
                    thresh=self._style_config.mask_threshold,
                    color=self._class_rgb(cls_idx),
                    alpha=self._style_config.mask_alpha,
                )

            pts = self._extract_landmarks(r)
            if pts:
                if len(pts) != 17:
                    logger.warning("Malformed YOLO pose landmarks: expected 17 points, got %d", len(pts))
                    continue
                try:
                    self._draw_pose(context, sx, sy, pts)
                except Exception as e:
                    logger.warning("Pose draw failed: %s", e)

            rects.append((x, y, w, h))
            # label position (top-left)
            labels.append((label, x + 2, max(0, y - 2)))

        try:
            self._mask_renderer.draw_all_masks(context, self._disp_w, self._disp_h)
        except Exception as e:
            logger.warning("Mask draw failed: %s", e)

        if not rects:
            return

        # draw all bboxes in single stroke
        context.set_antialias(cairo.ANTIALIAS_NONE if not self._style_config.antialias else cairo.ANTIALIAS_DEFAULT)
        context.set_line_width(self._style_config.box_line_w)
        context.new_path()
        for (x, y, w, h) in rects:
            context.rectangle(x, y, w, h)
        context.set_source_rgba(*self._style_config.box_rgba)
        context.stroke()

        # blit font surfaces onto draw surface
        for (text, lx, ly_top) in labels:
            surf, W, H = self._label_cache.get(text)
            draw_y = ly_top - H
            context.set_source_surface(surf, lx, draw_y)
            context.paint()


def overlay_factory(
    overlay_type: str,
    results_func: Callable[[], Any],
    model_inp_width: int,
    model_inp_height: int,
    fullscreen: bool = True,
    font_config: FontConfig | None = None
):
    if overlay_type.lower() == "object-detection":
        return SynapODOverlay(
            results_func,
            model_inp_width,
            model_inp_height,
            fullscreen,
            font_config
        )

    else:
        raise ValueError(f"Unsupported overlay type '{overlay_type}'")

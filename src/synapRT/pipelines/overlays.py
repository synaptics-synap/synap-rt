from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable
import logging
import os
import math
import threading

import cairo
import gi
gi.require_version('Gst', '1.0')
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Gst, Pango, PangoCairo
from synap.postprocessor import DetectorResult, DetectorResultItem

__all__ = [
    "FontConfig",
    "GstVideoOverlay",
    "SynapODOverlay",
    "overlay_factory"
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FontConfig:
    desc: str
    fg_rgba: tuple[float, float, float, float]
    bg_rgba: tuple[float, float, float, float]
    pad_xy: tuple[int, int]

    @classmethod
    def default(cls) -> "FontConfig":
        return cls(
            desc="Sans Bold 10",
            fg_rgba=(0, 1, 0, 1),
            bg_rgba=(0, 0, 0, 0),
            pad_xy=(4, 3)
        )


class _LabelCache:
    """Cache: text -> (ImageSurface, width, height). Thread-safe."""

    def __init__(
        self,
        font_config: FontConfig | None = None,
    ):
        self._cache: dict[str, tuple[cairo.ImageSurface, int, int]] = {}
        self._lock = threading.Lock()
        font_config = font_config or FontConfig.default()
        self._font = Pango.FontDescription(font_config.desc)
        self._fg = font_config.fg_rgba
        self._bg = font_config.bg_rgba
        self._pad_x, self._pad_y = font_config.pad_xy

    def get(self, text: str) -> tuple[cairo.ImageSurface, int, int]:
        with self._lock:
            hit = self._cache.get(text)
            if hit is not None:
                return hit

        # measure using a throwaway surface
        tmp = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
        ctx = cairo.Context(tmp)
        layout = PangoCairo.create_layout(ctx)
        layout.set_font_description(self._font)
        layout.set_text(text, -1)
        tw, th = layout.get_pixel_size()
        W = max(1, int(math.ceil(tw + 2 * self._pad_x)))
        H = max(1, int(math.ceil(th + 2 * self._pad_y)))

        # render to real surface
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        ctx = cairo.Context(surf)
        if self._bg[3] > 0:
            ctx.set_source_rgba(*self._bg)
            ctx.rectangle(0, 0, W, H)
            ctx.fill()

        ctx.set_source_rgba(*self._fg)
        layout = PangoCairo.create_layout(ctx)
        layout.set_font_description(self._font)
        layout.set_text(text, -1)
        ctx.move_to(self._pad_x, self._pad_y)
        PangoCairo.show_layout(ctx, layout)
        surf.flush()

        with self._lock:
            self._cache[text] = (surf, W, H)
            return self._cache[text]


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

    def __init__(
        self,
        results_provider: Callable[[], DetectorResult | dict[str, Any]],
        model_inp_width: int,
        model_inp_height: int,
        fullscreen: bool = True,
        font_config: FontConfig | None = None
    ):
        super().__init__(
            results_provider,
            model_inp_width,
            model_inp_height,
            fullscreen
        )
        self._label_cache = _LabelCache(font_config)

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
            elif isinstance(r, dict):
                x, y, w, h, label = self._render_json_rect(r, sx, sy)
            else:
                logger.warning("Invalid result object type '%s', skipping", type(r))
                continue

            # skip fully off-screen
            if x + w < 0 or y + h < 0 or x > self._disp_w or y > self._disp_h:
                continue

            rects.append((x, y, w, h))
            # label position (top-left)
            labels.append((label, x + 2, max(0, y - 2)))

        if not rects:
            return

        # draw all bboxes in single stroke
        context.set_antialias(cairo.ANTIALIAS_NONE)
        context.set_line_width(2.0)
        context.new_path()
        for (x, y, w, h) in rects:
            context.rectangle(x, y, w, h)
        context.set_source_rgba(0, 1, 0, 1)
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

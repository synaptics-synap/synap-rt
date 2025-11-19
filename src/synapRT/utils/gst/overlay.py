import math
import logging
import os
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from time import perf_counter_ns
from typing import Final

import cairo
import numpy as np
import numpy.typing as npt

import gi
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Pango, PangoCairo

__all__ = [
    "FontConfig",
    "StyleConfig",
    "LabelCache",
]

logger = logging.getLogger(__name__)

_COLORS_COCO_BGR: Final[dict[int, list[int]]] = {
    0: [189, 114, 0], 1: [25, 83, 217], 2: [32, 176, 237], 3: [142, 47, 126],
    4: [48, 172, 119], 5: [238, 190, 77], 6: [47, 20, 162], 7: [77, 77, 77],
    8: [153, 153, 153], 9: [0, 0, 255], 10: [0, 128, 255], 11: [0, 191, 191],
    12: [0, 255, 0], 13: [255, 0, 0], 14: [255, 0, 170], 15: [0, 85, 85],
    16: [0, 170, 85], 17: [0, 255, 85], 18: [0, 85, 170], 19: [0, 170, 170],
    20: [0, 255, 170], 21: [0, 85, 255], 22: [0, 170, 255], 23: [0, 255, 255],
    24: [128, 85, 0], 25: [128, 170, 0], 26: [128, 255, 0], 27: [128, 0, 85],
    28: [128, 85, 85], 29: [128, 170, 85], 30: [128, 255, 85], 31: [128, 0, 170],
    32: [128, 85, 170], 33: [128, 170, 170], 34: [128, 255, 170], 35: [128, 0, 255],
    36: [128, 85, 255], 37: [128, 170, 255], 38: [128, 255, 255], 39: [255, 85, 0],
    40: [255, 170, 0], 41: [255, 255, 0], 42: [255, 0, 85], 43: [255, 85, 85],
    44: [255, 170, 85], 45: [255, 255, 85], 46: [255, 0, 170], 47: [255, 85, 170],
    48: [255, 170, 170], 49: [255, 255, 170], 50: [255, 0, 255], 51: [255, 85, 255],
    52: [255, 170, 255], 53: [255, 255, 255], 54: [0, 0, 85], 55: [0, 0, 128],
    56: [0, 0, 170], 57: [0, 0, 212], 58: [0, 0, 255], 59: [0, 85, 0],
    60: [0, 128, 0], 61: [0, 170, 0], 62: [0, 212, 0], 63: [0, 255, 0],
    64: [85, 0, 0], 65: [128, 0, 0], 66: [170, 0, 0], 67: [212, 0, 0],
    68: [255, 0, 0], 69: [0, 85, 85], 70: [0, 85, 128], 71: [0, 85, 170],
    72: [0, 85, 212], 73: [0, 85, 255], 74: [0, 128, 85], 75: [0, 128, 128],
    76: [0, 128, 170], 77: [0, 128, 212], 78: [0, 128, 255], 79: [0, 170, 85],
}

DEFAULT_COLORMAP: Final[dict[int, tuple[float,float,float]]] = {
    k: (r/255.0, g/255.0, b/255.0) for k, (b, g, r) in _COLORS_COCO_BGR.items()
}

class RenderingThreadingPolicy(Enum):
    NONE  = auto()
    SMART = auto()
    MAX_N = auto()


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


@dataclass(frozen=True)
class StyleConfig:
    box_rgba: tuple[float,float,float,float] = (0, 1, 0, 1)
    box_line_w: float = 2.0
    mask_alpha: float = 0.35
    mask_threshold: float = 0.5
    kp_radius: float = 3.0
    kp_line_w: float = 2.0
    kp_rgba: tuple[float,float,float,float] = (1, 0.9, 0.3, 1.0)  # light yellow
    antialias: bool = False  # disable for faster rasterization


@dataclass
class MaskData:
    mask_w: int
    mask_h: int
    data: npt.NDArray[np.float32 | np.uint8]
    bbox: tuple[int, int, int, int] | None = None
    thresh: float = 0.5
    color: tuple[float, float, float] = (1.0, 0.0, 0.0)
    alpha: float = 0.35


class LabelCache:
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


class CairoSegMaskRenderer:
    """Render segmentation masks onto a Cairo surface"""

    def __init__(
        self,
        threading: RenderingThreadingPolicy = RenderingThreadingPolicy.SMART,
        max_workers: int | None = None
    ):
        self._threading_policy = threading
        self._max_workers = max_workers
        self._masks_data: list[MaskData] = []

        # for debugging
        self._mask_add_time = 0.0
        self._mask_raster_time = 0.0
        self._mask_draw_time = 0.0

    @staticmethod
    def rasterize_mask(mask_data: MaskData):
        buf = np.zeros((mask_data.mask_h, mask_data.mask_w), dtype=np.uint8)
        # Write comparison results as 0/1 bytes, then scale in-place to 0/255
        np.greater(mask_data.data, mask_data.thresh, out=buf.view(np.bool_))
        buf *= 255
        mask_data.data = buf

    def _rasterize(self):
        if self._masks_data:
            n_masks: int = len(self._masks_data)
            if self._threading_policy == RenderingThreadingPolicy.NONE:
                use_threads: bool = False
            elif self._threading_policy == RenderingThreadingPolicy.MAX_N:
                use_threads: bool = True
            else:
                if n_masks > 3:
                    use_threads = True
                else:
                    use_threads = False
            if use_threads:
                n_threads: int = self._max_workers or max(n_masks, os.cpu_count() or 4)
                with ThreadPoolExecutor(max_workers=n_threads) as ex:
                    ex.map(CairoSegMaskRenderer.rasterize_mask, self._masks_data)
            else:
                map(CairoSegMaskRenderer.rasterize_mask, self._masks_data)

    def add_mask(
        self,
        mask_w: int,
        mask_h: int,
        data: Iterable[float],
        *,
        bbox: tuple[int, int, int, int] | None = None,
        thresh: float = 0.5,
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        alpha: float = 0.35

    ):
        st = perf_counter_ns()
        self._masks_data.append(MaskData(
            mask_w,
            mask_h,
            np.asarray(data, dtype=np.float32).reshape(mask_h, mask_w),
            bbox=bbox,
            thresh=thresh,
            color=color,
            alpha=alpha
        ))
        self._mask_add_time += perf_counter_ns() - st

    def draw_all_masks(
        self,
        ctx: cairo.Context,
        frame_w: int,
        frame_h: int,
    ) -> None:
        if frame_w is None or frame_h is None:
            return

        st = perf_counter_ns()
        self._rasterize()
        self._mask_raster_time = perf_counter_ns() - st

        st = perf_counter_ns()
        for m in self._masks_data:
            ctx.save()
            if m.bbox:
                x, y, w, h = m.bbox
                ctx.rectangle(x, y, w, h)
                ctx.clip()

            surf = cairo.ImageSurface.create_for_data(
                m.data, cairo.FORMAT_A8, m.mask_w, m.mask_h
            )
            surf.mark_dirty()

            ctx.set_source_rgba(m.color[0], m.color[1], m.color[2], m.alpha)

            if m.mask_w == frame_w and m.mask_h == frame_h:
                ctx.mask_surface(surf, 0, 0)
                ctx.restore()
                continue

            # Scale mask to the frame
            pattern = cairo.SurfacePattern(surf)
            pattern.set_filter(cairo.FILTER_NEAREST)
            pattern.set_extend(cairo.EXTEND_NONE)

            M = cairo.Matrix()
            M.xx = m.mask_w / float(frame_w)
            M.yy = m.mask_h / float(frame_h)
            M.x0 = 0.0
            M.y0 = 0.0
            pattern.set_matrix(M)

            ctx.mask(pattern)
            ctx.restore()
        self._mask_draw_time = perf_counter_ns() - st

        logger.debug(
            "%s: Seg mask overlay time (ms): add = %.3f, raster = %.3f, draw = %.3f",
            self.__class__.__name__,
            self._mask_add_time / 1e6,
            self._mask_raster_time / 1e6,
            self._mask_draw_time / 1e6
        )
        self._mask_add_time = 0
        self._mask_raster_time = 0
        self._mask_draw_time = 0

        self._masks_data.clear()

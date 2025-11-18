import math
import threading
from dataclasses import dataclass

import cairo
import gi
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Pango, PangoCairo

__all__ = [
    "FontConfig",
    "LabelCache",
]


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

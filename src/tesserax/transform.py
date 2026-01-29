from __future__ import annotations
import math
from dataclasses import dataclass
from .core import Shape, Bounds, Point


@dataclass
class Transform:
    tx: float = 0.0
    ty: float = 0.0
    rotation: float = 0.0
    scale: float = 1.0

    def map(self, p: Point) -> Point:
        """Applies this local transform to a point."""
        rad = math.radians(self.rotation)
        # Scale
        nx, ny = p.x * self.scale, p.y * self.scale
        # Rotate
        rx = nx * math.cos(rad) - ny * math.sin(rad)
        ry = nx * math.sin(rad) + ny * math.cos(rad)
        # Translate
        return Point(rx + self.tx, ry + self.ty)

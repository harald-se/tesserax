from __future__ import annotations
import copy
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod
import math
from typing import Literal, Self
import typing


type Anchor = Literal[
    "top",
    "bottom",
    "left",
    "right",
    "center",
    "topleft",
    "topright",
    "bottomleft",
    "bottomright",
]


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def apply(self, tx=0.0, ty=0.0, r=0.0, s=1.0) -> Point:
        """Applies transformation parameters to this point."""
        rad = math.radians(r)
        # Scale
        nx, ny = self.x * s, self.y * s
        # Rotate
        rx = nx * math.cos(rad) - ny * math.sin(rad)
        ry = nx * math.sin(rad) + ny * math.cos(rad)
        # Translate
        return Point(rx + tx, ry + ty)


@dataclass(frozen=True)
class Bounds:
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> Point:
        return Point(self.x, self.y + self.height / 2)

    @property
    def right(self) -> Point:
        return Point(self.x + self.width, self.y + self.height / 2)

    @property
    def top(self) -> Point:
        return Point(self.x + self.width / 2, self.y)

    @property
    def bottom(self) -> Point:
        return Point(self.x + self.width / 2, self.y + self.height)

    @property
    def topleft(self) -> Point:
        return Point(self.x, self.y)

    @property
    def topright(self) -> Point:
        return Point(self.x + self.width, self.y)

    @property
    def bottomleft(self) -> Point:
        return Point(self.x, self.y + self.height)

    @property
    def bottomright(self) -> Point:
        return Point(self.x + self.width, self.y + self.height)

    def padded(self, amount: float) -> Bounds:
        """Returns a new Bounds expanded by the given padding amount on all sides."""
        return Bounds(
            x=self.x - amount,
            y=self.y - amount,
            width=self.width + 2 * amount,
            height=self.height + 2 * amount,
        )

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2, self.y + self.height / 2)

    def anchor(self, name: Anchor) -> Point:
        """Returns a Point based on a string name for layout flexibility."""
        match name:
            case "top":
                return self.top
            case "bottom":
                return self.bottom
            case "left":
                return self.left
            case "right":
                return self.right
            case "center":
                return self.center
            case "topleft":
                return self.topleft
            case "topright":
                return self.topright
            case "bottomleft":
                return self.bottomleft
            case "bottomright":
                return self.bottomright
            case _:
                raise ValueError(f"Unknown anchor: {name}")

    @classmethod
    def union(cls, *bounds: Bounds) -> Bounds:
        """Computes the minimal bounding box that contains all given bounds."""
        if not bounds:
            return Bounds(0, 0, 0, 0)

        x_min = min(b.x for b in bounds)
        y_min = min(b.y for b in bounds)
        x_max = max(b.x + b.width for b in bounds)
        y_max = max(b.y + b.height for b in bounds)

        return Bounds(x_min, y_min, x_max - x_min, y_max - y_min)


if typing.TYPE_CHECKING:
    from .transform import Transform
    from .base import Group


class Shape(ABC):
    """Base class for all renderable SVG components."""
    def __init__(self) -> None:
        self.transform = Transform()
        self.parent: Group | None = None

    @abstractmethod
    def local_bounds(self) -> Bounds:
        """Calculates the bounding box of the shape in the coordinate space."""
        pass

    def bounds(self) -> Bounds:
        """Computes the AABB of the transformed local bounds."""
        base = self.local_bounds()
        corners = [base.topleft, base.topright, base.bottomleft, base.bottomright]
        transformed = [self.transform.map(p) for p in corners]

        xs = [p.x for p in transformed]
        ys = [p.y for p in transformed]

        return Bounds(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    @abstractmethod
    def render(self) -> str:
        """Returns the SVG XML string representation of the shape."""
        pass

    def resolve(self, p: Point) -> Point:
        """Recursively resolves a point to global coordinates."""
        world_p = self.transform.map(p)
        if self.parent:
            return self.parent.resolve(world_p)
        return world_p

    def anchor(self, name: Anchor) -> Point:
        """Gets a global coordinate for a named anchor."""
        return self.resolve(self.local_bounds().anchor(name))

    def translated(self, dx: float, dy: float) -> Self:
        self.transform.tx += dx
        self.transform.ty += dy
        return self

    def rotated(self, r: float) -> Self:
        self.transform.rotation += r
        return self

    def scaled(self, s: float) -> Self:
        self.transform.scale += s
        return self

    def __add__(self, other: Shape) -> Group:
        """Enables the 'shape + shape' syntax to create groups."""
        return Group().add(self, other)

    def clone(self) -> typing.Self:
        """
        Returns a deep copy of the shape.
        Essential for creating variations of a base structure without side effects.
        """
        return copy.deepcopy(self)

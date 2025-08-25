import math
import sys
from typing import List, NamedTuple


class Point(NamedTuple):
    x: float
    y: float


EPSILON = math.sqrt(sys.float_info.epsilon)


def _is_clockwise(polygon: List[Point]) -> bool:
    s = 0
    polygon_count = len(polygon)
    for i in range(polygon_count):
        point = polygon[i]
        point2 = polygon[(i + 1) % polygon_count]
        s += (point2.x - point.x) * (point2.y + point.y)
    return s > 0


def _is_convex(prev: Point, point: Point, next: Point) -> bool:
    return _triangle_sum(prev.x, prev.y, point.x, point.y, next.x, next.y) < 0


def _is_ear(p1: Point, p2: Point, p3: Point, polygon: List[Point]) -> bool:
    ear = (
        _contains_no_points(p1, p2, p3, polygon)
        and _is_convex(p1, p2, p3)
        and _triangle_area(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y) > 0
    )
    return ear


def _contains_no_points(p1: Point, p2: Point, p3: Point, polygon: List[Point]) -> bool:
    for pn in polygon:
        if pn in (p1, p2, p3):
            continue
        elif _is_point_inside(pn, p1, p2, p3):
            return False
    return True


def _is_point_inside(p: Point, a: Point, b: Point, c: Point) -> bool:
    area = _triangle_area(a.x, a.y, b.x, b.y, c.x, c.y)
    area1 = _triangle_area(p.x, p.y, b.x, b.y, c.x, c.y)
    area2 = _triangle_area(p.x, p.y, a.x, a.y, c.x, c.y)
    area3 = _triangle_area(p.x, p.y, a.x, a.y, b.x, b.y)
    areadiff = abs(area - sum([area1, area2, area3])) < EPSILON
    return areadiff


def _triangle_area(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def _triangle_sum(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    return x1 * (y3 - y2) + x2 * (y1 - y3) + x3 * (y2 - y1)

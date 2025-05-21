import math
import sys
from collections import namedtuple
from typing import List, Tuple, TypeVar, Generic
import numpy as np
from numpy.typing import NDArray

Point = namedtuple("Point", ["x", "y"])

EPSILON = math.sqrt(sys.float_info.epsilon)

T = TypeVar("T")


def earclip(polygon: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Simple earclipping algorithm for a given polygon p.
    polygon is expected to be a numpy array of shape (n,2) containing the cartesian points of the polygon

    For a polygon with n points it will return n-2 triangles.
    The triangles are returned as a numpy array of shape (n-2, 3, 2) where:
    - n-2 is the number of triangles
    - 3 represents the three points of each triangle
    - 2 represents the x,y coordinates of each point

    e.g
    >>> polygon = np.array([[0,1], [-1, 0], [0, -1], [1, 0]])
    >>> triangles = tripy.earclip(polygon)
    >>> triangles.shape
    (2, 3, 2)
    """
    ear_vertex: List[Point] = []
    triangles_list: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = []

    # Convert numpy array to list of Points
    polygon_points = [Point(float(x), float(y)) for x, y in polygon]

    if _is_clockwise(polygon_points):
        polygon_points.reverse()

    point_count = len(polygon_points)
    for i in range(point_count):
        prev_index = i - 1
        prev_point = polygon_points[prev_index]
        point = polygon_points[i]
        next_index = (i + 1) % point_count
        next_point = polygon_points[next_index]

        if _is_ear(prev_point, point, next_point, polygon_points):
            ear_vertex.append(point)

    while ear_vertex and point_count >= 3:
        ear = ear_vertex.pop(0)
        i = polygon_points.index(ear)
        prev_index = i - 1
        prev_point = polygon_points[prev_index]
        next_index = (i + 1) % point_count
        next_point = polygon_points[next_index]

        polygon_points.remove(ear)
        point_count -= 1
        triangles_list.append(((prev_point.x, prev_point.y), (ear.x, ear.y), (next_point.x, next_point.y)))
        if point_count > 3:
            prev_prev_point = polygon_points[prev_index - 1]
            next_next_index = (i + 1) % point_count
            next_next_point = polygon_points[next_next_index]

            groups = [
                (prev_prev_point, prev_point, next_point, polygon_points),
                (prev_point, next_point, next_next_point, polygon_points),
            ]
            for group in groups:
                p = group[1]
                if _is_ear(*group):
                    if p not in ear_vertex:
                        ear_vertex.append(p)
                elif p in ear_vertex:
                    ear_vertex.remove(p)

    # Convert list of tuples to numpy array of shape (n, 3, 2)
    triangles_array = np.array(triangles_list)
    return triangles_array


def earclip_indices(polygon: NDArray[np.float64]) -> NDArray[np.int64]:
    """
    Simple earclipping algorithm for a given polygon p that returns indices instead of points.
    polygon is expected to be a numpy array of shape (n,2) containing the cartesian points of the polygon

    For a polygon with n points it will return n-2 triangles.
    The triangles are returned as a numpy array of shape (n-2, 3) containing the indices of the points
    in the original polygon that form each triangle.

    e.g
    >>> polygon = np.array([[0,1], [-1, 0], [0, -1], [1, 0]])
    >>> triangles = tripy.earclip_indices(polygon)
    >>> triangles.shape
    (2, 3)
    >>> triangles
    array([[0, 1, 2],
           [0, 2, 3]])
    """
    ear_vertex: List[Tuple[Point, int]] = []
    triangles_list: List[Tuple[int, int, int]] = []

    # Convert numpy array to list of Points with their indices
    polygon_points = [(Point(float(x), float(y)), i) for i, (x, y) in enumerate(polygon)]
    points, indices = zip(*polygon_points)

    if _is_clockwise(points):
        points = list(points)
        indices = list(indices)
        points.reverse()
        indices.reverse()
        polygon_points = list(zip(points, indices))

    point_count = len(polygon_points)
    for i in range(point_count):
        prev_index = i - 1
        prev_point, prev_idx = polygon_points[prev_index]
        point, point_idx = polygon_points[i]
        next_index = (i + 1) % point_count
        next_point, next_idx = polygon_points[next_index]

        if _is_ear(prev_point, point, next_point, [p for p, _ in polygon_points]):
            ear_vertex.append((point, point_idx))

    while ear_vertex and point_count >= 3:
        ear, ear_idx = ear_vertex.pop(0)
        i = next(idx for idx, (p, _) in enumerate(polygon_points) if p == ear)
        prev_index = i - 1
        prev_point, prev_idx = polygon_points[prev_index]
        next_index = (i + 1) % point_count
        next_point, next_idx = polygon_points[next_index]

        polygon_points.remove((ear, ear_idx))
        point_count -= 1
        triangles_list.append((prev_idx, ear_idx, next_idx))
        if point_count > 3:
            prev_prev_point, prev_prev_idx = polygon_points[prev_index - 1]
            next_next_index = (i + 1) % point_count
            next_next_point, next_next_idx = polygon_points[next_next_index]

            groups = [
                (prev_prev_point, prev_point, next_point, [p for p, _ in polygon_points]),
                (prev_point, next_point, next_next_point, [p for p, _ in polygon_points]),
            ]
            for group in groups:
                p = group[1]
                if _is_ear(*group):
                    if not any(p == point for point, _ in ear_vertex):
                        ear_vertex.append((p, next(idx for idx, (pt, _) in enumerate(polygon_points) if pt == p)))
                else:
                    ear_vertex = [(point, idx) for point, idx in ear_vertex if point != p]

    # Convert list of tuples to numpy array of shape (n, 3)
    triangles_array = np.array(triangles_list, dtype=np.int64)
    return triangles_array


def get_triangulation_points(polygon: NDArray[np.float64], triangles: NDArray[np.int64]) -> NDArray[np.float64]:
    """
    Convert triangle indices to actual points.

    Args:
        polygon: Input polygon points as numpy array of shape (n, 2)
        triangles: Triangle indices as numpy array of shape (m, 3)

    Returns:
        numpy array of shape (m, 3, 2) containing the actual points for each triangle
    """
    return polygon[triangles]


def calculate_total_area_points(triangles: NDArray[np.float64]) -> float:
    """
    Calculate the total area of all triangles.
    triangles is expected to be a numpy array of shape (n, 3, 2) where:
    - n is the number of triangles
    - 3 represents the three points of each triangle
    - 2 represents the x,y coordinates of each point
    """
    result: List[Tuple[float, float, float, float]] = []
    for triangle in triangles:
        sides: List[float] = []
        for i in range(3):
            next_index = (i + 1) % 3
            pt = triangle[i]
            pt2 = triangle[next_index]
            # Distance between two points
            side = math.sqrt(math.pow(pt2[0] - pt[0], 2) + math.pow(pt2[1] - pt[1], 2))
            sides.append(side)
        # Heron's numerically stable forumla for area of a triangle:
        # https://en.wikipedia.org/wiki/Heron%27s_formula
        # However, for line-like triangles of zero area this formula can produce an infinitesimally negative value
        # as an input to sqrt() due to the cumulative arithmetic errors inherent to floating point calculations:
        # https://people.eecs.berkeley.edu/~wkahan/Triangle.pdf
        # For this purpose, abs() is used as a reasonable guard against this condition.
        c, b, a = sorted(sides)
        area = 0.25 * math.sqrt(abs((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))))
        result.append((area, a, b, c))
    triangle_area = sum(tri[0] for tri in result)
    return triangle_area


def calculate_total_area_indices(polygon: NDArray[np.float64], triangles: NDArray[np.int64]) -> float:
    """
    Calculate the total area of all triangles using indices.

    Args:
        polygon: Input polygon points as numpy array of shape (n, 2)
        triangles: Triangle indices as numpy array of shape (m, 3)
    """
    triangle_points = get_triangulation_points(polygon, triangles)
    return calculate_total_area_points(triangle_points)


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

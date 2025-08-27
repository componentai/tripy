import math
import sys
from collections import namedtuple
from typing import List, Tuple, Union, NamedTuple, Sequence, TypeVar, cast
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class Point(NamedTuple):
    x: float
    y: float


EPSILON = math.sqrt(sys.float_info.epsilon)


def earclip(
    polygon: Sequence[Tuple[float, float]],
) -> List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """
    Simple earclipping algorithm for a given polygon p.
    polygon is expected to be an array of 2-tuples of the cartesian points of the polygon

    For a polygon with n points it will return n-2 triangles.
    The triangles are returned as an array of 3-tuples where each item in the tuple is a 2-tuple of the cartesian point.

    e.g
    >>> polygon = [(0,1), (-1, 0), (0, -1), (1, 0)]
    >>> triangles = tripy.earclip(polygon)
    >>> triangles
    [((1, 0), (0, 1), (-1, 0)), ((1, 0), (-1, 0), (0, -1))]

    Implementation Reference:
        - https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
    """
    ear_vertex = []
    triangles = []

    # Convert input tuples to Points
    polygon_points = [Point(x, y) for x, y in polygon]

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
        triangles.append(((prev_point.x, prev_point.y), (ear.x, ear.y), (next_point.x, next_point.y)))
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
    return triangles


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


def calculate_total_area(
    triangles: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
) -> float:
    result = []
    for triangle in triangles:
        sides = []
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


def earclip_np(polygon: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    NumPy version of the earclipping algorithm for a given polygon.
    polygon is expected to be a numpy array of shape (n, 2) containing the cartesian points of the polygon

    For a polygon with n points it will return n-2 triangles.
    The triangles are returned as a numpy array of shape (n-2, 3, 2) where each item is a 2D point.

    e.g
    >>> polygon = np.array([[0,1], [-1, 0], [0, -1], [1, 0]])
    >>> triangles = tripy.earclip_np(polygon)
    >>> triangles.shape
    (2, 3, 2)
    """
    ear_vertex = []
    triangles = []

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
        triangles.append(((prev_point.x, prev_point.y), (ear.x, ear.y), (next_point.x, next_point.y)))
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

    # Convert the list of triangles to a numpy array
    return np.array(triangles)


def earclip_np_indices(
    polygon: npt.NDArray[np.float64], vertex_indices_map: npt.NDArray[np.int64] | None = None
) -> npt.NDArray[np.int64]:
    """
    NumPy version of the earclipping algorithm that returns indices instead of coordinates.
    polygon is expected to be a numpy array of shape (n, 2) containing the cartesian points of the polygon

    For a polygon with n points it will return n-2 triangles.
    The triangles are returned as a numpy array of shape (n-2, 3) containing the indices of the points
    that form each triangle.

    e.g
    >>> polygon = np.array([[0,1], [-1, 0], [0, -1], [1, 0]])
    >>> triangles = tripy.earclip_np_indices(polygon)
    >>> triangles.shape
    (2, 3)
    >>> triangles
    array([[0, 1, 2],
           [0, 2, 3]])
    """
    ear_vertex = []
    triangles = []
    indices = list(range(len(polygon)))  # Keep track of original indices

    # Convert numpy array to list of Points
    polygon_points = [Point(float(x), float(y)) for x, y in polygon]

    if _is_clockwise(polygon_points):
        polygon_points.reverse()
        indices.reverse()

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

        # Store the indices of the triangle
        triangles.append((indices[prev_index], indices[i], indices[next_index]))

        polygon_points.remove(ear)
        indices.pop(i)  # Remove the corresponding index
        point_count -= 1

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

    if vertex_indices_map is not None:
        # Map each index in each triangle tuple to its corresponding vertex index
        triangles = [(vertex_indices_map[i], vertex_indices_map[j], vertex_indices_map[k]) for i, j, k in triangles]
        return np.array(triangles, dtype=np.int64)

    return np.array(triangles, dtype=np.int64)


def test_triangulation(polygon, name="Test case"):
    """
    Test all three triangulation functions on a given polygon and verify their outputs are equivalent.

    Args:
        polygon: Either a list of (x,y) tuples or a numpy array of shape (n,2)
        name: Name of the test case for printing
    """
    import numpy as np

    # Convert to numpy array if it's a list
    if isinstance(polygon, list):
        polygon_np = np.array(polygon)
    else:
        polygon_np = polygon

    # Test all three functions
    triangles = earclip(polygon)
    triangles_np = earclip_np(polygon_np)
    triangles_indices = earclip_np_indices(polygon_np)

    # Convert all outputs to numpy arrays for comparison
    triangles_np_from_list = np.array(triangles)
    triangles_np_from_indices = polygon_np[triangles_indices]

    # Function to compare triangles in an order-invariant way
    def compare_triangles(tri1, tri2):
        # Sort vertices of each triangle by x-coordinate, then y-coordinate
        tri1_sorted = np.sort(tri1, axis=0)
        tri2_sorted = np.sort(tri2, axis=0)
        return np.allclose(tri1_sorted, tri2_sorted)

    # Verify all outputs are equivalent
    list_np_match = np.allclose(triangles_np_from_list, triangles_np)
    list_indices_match = all(
        compare_triangles(t1, t2) for t1, t2 in zip(triangles_np_from_list, triangles_np_from_indices)
    )
    np_indices_match = all(compare_triangles(t1, t2) for t1, t2 in zip(triangles_np, triangles_np_from_indices))

    print(f"\n{name}:")
    print(f"Number of points: {len(polygon_np)}")
    print(f"Number of triangles: {len(triangles)}")
    print(f"All methods produce equivalent results: {list_np_match and list_indices_match and np_indices_match}")

    # print(f"polygon: {polygon_np}")
    # print(f"Earclip: {triangles}")
    # print(f"Earclip_np: {triangles_np}")
    # print(f"Earclip_np_indices: {triangles_indices}")

    return list_np_match and list_indices_match and np_indices_match


def generate_random_polygon(n_points, radius=1.0, center=(0, 0)):
    """
    Generate a random simple polygon with n_points vertices.
    The polygon will be approximately circular with points distributed around the center.

    Args:
        n_points: Number of points in the polygon
        radius: Approximate radius of the polygon
        center: Center point of the polygon

    Returns:
        numpy array of shape (n_points, 2) containing the polygon vertices
    """
    import numpy as np

    # Generate random angles
    angles = np.sort(np.random.uniform(0, 2 * np.pi, n_points))

    # Add some random variation to the radius
    radii = radius * (1 + 0.2 * np.random.randn(n_points))

    # Convert to cartesian coordinates
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)

    return np.column_stack((x, y))

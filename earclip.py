from typing import List, Tuple, Sequence
from core import Point, _is_clockwise, _is_ear


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

import numpy as np
import numpy.typing as npt
from core import Point, _is_clockwise, _is_ear


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

    If vertex_indices_map is provided, it is expected to be a numpy array of shape (n) containing the indices of each vertex.
    The returned triangles will then be the indices of the vertices in the vertex_indices_map array.

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

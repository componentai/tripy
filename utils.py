import math
import numpy as np
from typing import List, Tuple
from earclip import earclip
from earclip_np import earclip_np, earclip_np_indices


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


def test_triangulation(polygon, name="Test case"):
    """
    Test all three triangulation functions on a given polygon and verify their outputs are equivalent.

    Args:
        polygon: Either a list of (x,y) tuples or a numpy array of shape (n,2)
        name: Name of the test case for printing
    """
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
    # Generate random angles
    angles = np.sort(np.random.uniform(0, 2 * np.pi, n_points))

    # Add some random variation to the radius
    radii = radius * (1 + 0.2 * np.random.randn(n_points))

    # Convert to cartesian coordinates
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)

    return np.column_stack((x, y))

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from earclip import earclip
from earclip_np import earclip_np, earclip_np_indices


__all__ = [
    "earclip",
    "earclip_np",
    "earclip_np_indices",
]


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


def plot_polygon_and_triangulation(
    polygon: npt.NDArray[np.float64],
    triangles: npt.NDArray[np.float64],
    title: str = "Earclip_np triangulation",
):
    plt.figure(figsize=(8, 8))
    # Scatter the polygon vertices
    plt.scatter(polygon[:, 0], polygon[:, 1], s=10, c="black", label="Vertices")

    # Draw triangle edges
    for tri in triangles:
        xs = [tri[0][0], tri[1][0], tri[2][0], tri[0][0]]
        ys = [tri[0][1], tri[1][1], tri[2][1], tri[0][1]]
        plt.plot(xs, ys, color="tab:blue", linewidth=0.8)

    # Draw polygon boundary
    boundary = np.vstack([polygon, polygon[0]])
    plt.plot(boundary[:, 0], boundary[:, 1], color="tab:orange", linewidth=1.0, alpha=0.6, label="Boundary")

    # Compute independent x/y ranges with their own padding (independent scaling)
    x_min, x_max = float(np.min(polygon[:, 0])), float(np.max(polygon[:, 0]))
    y_min, y_max = float(np.min(polygon[:, 1])), float(np.max(polygon[:, 1]))
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_padding = (0.05 * x_span) if x_span > 0 else 1.0
    y_padding = (0.05 * y_span) if y_span > 0 else 1.0

    ax = plt.gca()
    ax.set_xlim(x_center - (0.5 * x_span + x_padding), x_center + (0.5 * x_span + x_padding))
    ax.set_ylim(y_center - (0.5 * y_span + y_padding), y_center + (0.5 * y_span + y_padding))
    ax.set_aspect("auto")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    i = 2
    np.set_printoptions(threshold=np.inf)  # type: ignore
    test_polygon = np.load(f"projected_backside_vertices_{i}.npy")
    test_data = np.load(f"backside_vertices_{i}.npy")

    print(f"test_polygon: {test_polygon}")
    print(f"test_data: {test_data}")

    triangulation = earclip_np(test_polygon)

    plot_polygon_and_triangulation(test_polygon, triangulation, title="Test polygon with earclip_np triangulation")

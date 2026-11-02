import numpy as np
from scipy.spatial import cKDTree


def mask_near_points(xgrid, ygrid, xpoints, ypoints, threshold=None) -> np.ndarray:
    """A boolean mask of the points of a 2D grid that are close to at least one data point.

    The nearest data point to each grid point is found with a KD-tree. Use it to restrict a
    model evaluated on a grid to where there are data.

    Args:
        xgrid (numpy.ndarray): Grid coordinates along x (monotonically increasing).
        ygrid (numpy.ndarray): Grid coordinates along y (monotonically increasing).
        xpoints (numpy.ndarray): x coordinates of the data points.
        ypoints (numpy.ndarray): y coordinates of the data points.
        threshold (float | None): A grid point is kept if its nearest data point is closer
            than this. If `None`, 1.5 times the larger of the mean grid spacings in x and y.

    Returns:
        Boolean mask of shape `(len(ygrid), len(xgrid))`, `True` where the grid point is
            near a data point ("keep").

    Example:
        ```python
        import numpy as np
        from lvm_tools.utils.mask import mask_near_points

        xgrid = np.linspace(0, 10, 100)
        ygrid = np.linspace(0, 10, 100)
        xpoints = np.random.uniform(2, 8, 50)
        ypoints = np.random.uniform(2, 8, 50)

        mask = mask_near_points(xgrid, ygrid, xpoints, ypoints)
        data = np.ones((len(ygrid), len(xgrid)))
        masked_data = np.where(mask, data, np.nan)
        ```
    """
    xx, yy = np.meshgrid(xgrid, ygrid, indexing="xy")
    grid_centers = np.column_stack([xx.ravel(), yy.ravel()])

    tree = cKDTree(np.column_stack([xpoints, ypoints]))
    dists, _ = tree.query(grid_centers, k=1)

    if threshold is None:
        dx = np.mean(np.diff(xgrid))
        dy = np.mean(np.diff(ygrid))
        threshold = 1.5 * max(dx, dy)

    mask = (dists < threshold).reshape(xx.shape)
    return mask

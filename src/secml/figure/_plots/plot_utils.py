"""
.. module:: PlotUtils
   :synopsis: Collection of mixed utilities for Plots

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.array import CArray

__all__ = ['create_points_grid']


def create_points_grid(grid_limits, n_grid_points):
    """Creates a grid of points.

    Parameters
    ----------
    grid_limits : list of tuple
        List with a tuple of min/max limits for each axis.
        If None, [(0, 1), (0, 1)] limits will be used.
    n_grid_points : int
        Number of grid points.

    """
    grid_bounds = [(0, 1), (0, 1)] if grid_limits is None else grid_limits
    x_min, x_max = (grid_bounds[0][0], grid_bounds[0][1])
    y_min, y_max = (grid_bounds[1][0], grid_bounds[1][1])

    # Padding x and y grid points
    padding_x, padding_y = (0.05 * (x_max - x_min), 0.05 * (y_max - y_min))
    # Create the equi-spaced indices for each axis
    x_grid_points = CArray.linspace(
        x_min - padding_x, x_max + padding_x, num=n_grid_points)
    y_grid_points = CArray.linspace(
        y_min - padding_y, y_max + padding_y, num=n_grid_points)
    # Create the grid
    pad_xgrid, pad_ygrid = CArray.meshgrid((x_grid_points, y_grid_points))
    pad_grid_point_features = CArray.concatenate(
        pad_xgrid.reshape((pad_xgrid.size, 1)),
        pad_ygrid.reshape((pad_ygrid.size, 1)), axis=1)

    return pad_grid_point_features, pad_xgrid, pad_ygrid

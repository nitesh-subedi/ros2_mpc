import numpy as np
from numba import njit


@njit
def inflate_global(occupancy_grid, inflation_matrix, cells_inflation):
    new_grid = occupancy_grid.copy()
    for i in range(occupancy_grid.shape[0]):
        for j in range(occupancy_grid.shape[1]):
            if occupancy_grid[i, j] == 0:
                if new_grid[max(0, i - cells_inflation):min(occupancy_grid.shape[0], i + cells_inflation + 1), max(
                        0, j - cells_inflation):min(occupancy_grid.shape[1],
                                                    j + cells_inflation + 1)].shape != inflation_matrix.shape:
                    continue
                new_grid[max(0, i - cells_inflation):min(occupancy_grid.shape[0], i + cells_inflation + 1), max(
                    0, j - cells_inflation):min(occupancy_grid.shape[1], j + cells_inflation + 1)] = np.minimum(
                    new_grid[max(0, i - cells_inflation):min(occupancy_grid.shape[0], i + cells_inflation + 1), max(
                        0, j - cells_inflation):min(occupancy_grid.shape[1], j + cells_inflation + 1)],
                    inflation_matrix)
    return new_grid


@njit
def inflate_local(occupancy_grid, inflation_matrix, cells_inflation, robot_position, costmap_size):
    # Extract the occupancy grid around the robot position with the size of the costmap
    occupancy_grid = occupancy_grid[int(robot_position[1] - costmap_size / 2):int(robot_position[1] + costmap_size / 2),
                    int(robot_position[0] - costmap_size / 2):int(robot_position[0] + costmap_size / 2)]
    new_grid = occupancy_grid.copy()
    for i in range(occupancy_grid.shape[0]):
        for j in range(occupancy_grid.shape[1]):
            if occupancy_grid[i, j] == 0:
                if new_grid[max(0, i - cells_inflation):min(occupancy_grid.shape[0], i + cells_inflation + 1), max(
                        0, j - cells_inflation):min(occupancy_grid.shape[1],
                                                    j + cells_inflation + 1)].shape != inflation_matrix.shape:
                    continue
                new_grid[max(0, i - cells_inflation):min(occupancy_grid.shape[0], i + cells_inflation + 1), max(
                    0, j - cells_inflation):min(occupancy_grid.shape[1], j + cells_inflation + 1)] = np.minimum(
                    new_grid[max(0, i - cells_inflation):min(occupancy_grid.shape[0], i + cells_inflation + 1), max(
                        0, j - cells_inflation):min(occupancy_grid.shape[1], j + cells_inflation + 1)],
                    inflation_matrix)
    return new_grid


@njit
def get_inflation_matrix(cells_inflation, factor=1.3):
    inflation_matrix = np.zeros((2 * cells_inflation + 1, 2 * cells_inflation + 1))
    # Make the center cell 100 and gradually decrease the value to 0 as we move away from the center
    inflation_matrix[cells_inflation, cells_inflation] = 100
    decay = (1 / cells_inflation) / factor
    for k in range(cells_inflation):
        inflation_matrix[k: np.shape(inflation_matrix)[
                                0] - k, k] = decay * (k + 1) * 100
        inflation_matrix[k: np.shape(inflation_matrix)[
                                0] - k, np.shape(inflation_matrix)[0] - (k + 1)] = decay * (k + 1) * 100
        inflation_matrix[k, k: np.shape(inflation_matrix)[
                                   1] - k] = decay * (k + 1) * 100
        inflation_matrix[np.shape(inflation_matrix)[
                             1] - (k + 1), k: np.shape(inflation_matrix)[1] - k] = decay * (k + 1) * 100
    return inflation_matrix

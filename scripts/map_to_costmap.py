import numpy as np
import cv2
import yaml
from numba import njit
import time


@njit   # This is the fastest version
def inflate(occupancy_grid, new_grid, inflation_matrix, cells_inflation):
    for i in range(occupancy_grid.shape[0]):
        for j in range(occupancy_grid.shape[1]):
            if occupancy_grid[i, j] == 0:
                if new_grid[max(0, i-cells_inflation):min(occupancy_grid.shape[0], i+cells_inflation+1), max(
                    0, j-cells_inflation):min(occupancy_grid.shape[1], j+cells_inflation+1)].shape != inflation_matrix.shape:
                    continue
                new_grid[max(0, i-cells_inflation):min(occupancy_grid.shape[0], i+cells_inflation+1), max(
                    0, j-cells_inflation):min(occupancy_grid.shape[1], j+cells_inflation+1)] = np.minimum(new_grid[max(0, i-cells_inflation):min(occupancy_grid.shape[0], i+cells_inflation+1), max(
                        0, j-cells_inflation):min(occupancy_grid.shape[1], j+cells_inflation+1)], inflation_matrix)
    return new_grid


def get_inflation_matrix(cells_inflation, inflation_matrix):
    # Make the center cell 100 and gradually decrease the value to 0 as we move away from the center
    inflation_matrix[cells_inflation, cells_inflation] = 100
    decay = (1 / cells_inflation) / 1.3 # 2 is a magic number
    for k in range(cells_inflation):
        inflation_matrix[k: np.shape(inflation_matrix)[
            0] - k, k] = decay * (k + 1) * 100
        inflation_matrix[k: np.shape(inflation_matrix)[
            0] - k, np.shape(inflation_matrix)[0]-(k+1)] = decay * (k + 1) * 100
        inflation_matrix[k, k: np.shape(inflation_matrix)[
            1] - k] = decay * (k + 1) * 100
        inflation_matrix[np.shape(inflation_matrix)[
            1]-(k+1), k: np.shape(inflation_matrix)[1] - k] = decay * (k + 1) * 100
    return inflation_matrix


def main():
    # Load map
    img = cv2.imread(
        './maps/map_carto.pgm', cv2.IMREAD_GRAYSCALE)

    # Load map.yaml file
    with open('./maps/map_carto.yaml', 'r') as file:
        params = yaml.safe_load(file)

    resolution = params['resolution']
    occupancy_thresh = params['occupied_thresh']
    inflation = 0.22  # in meters
    cells_inflation = int(inflation / resolution)

    # Threshold the image into a binary image
    ret, binary = cv2.threshold(img, occupancy_thresh, 255, cv2.THRESH_BINARY)

    # Convert to occupancy grid
    occupancy_grid = np.zeros((img.shape[0], img.shape[1]))
    occupancy_grid[binary == 255] = 100

    new_grid = occupancy_grid.copy()
    inflation_matrix = np.zeros((2*cells_inflation + 1, 2*cells_inflation + 1))
    inflation_matrix = (get_inflation_matrix(cells_inflation, inflation_matrix)) / 100
    # Invert the matrix so that the center cell is 0 and the outer cells are 1
    inflation_matrix = 1 - inflation_matrix
    print(inflation_matrix)
    new_grid = inflate(occupancy_grid, new_grid,
                       inflation_matrix, cells_inflation)
    cv2.imshow('Inflated map', new_grid)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()

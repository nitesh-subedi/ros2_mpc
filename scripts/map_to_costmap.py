import numpy as np
import cv2
import yaml
from numba import njit
from ros2_mpc import get_inflation_matrix, inflate_local


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
    inflation_matrix = np.zeros((2 * cells_inflation + 1, 2 * cells_inflation + 1))
    inflation_matrix = (get_inflation_matrix(cells_inflation)) / 100
    # Invert the matrix so that the center cell is 0 and the outer cells are 1
    inflation_matrix = 1 - inflation_matrix
    print(inflation_matrix)
    new_grid = inflate_local(occupancy_grid, new_grid,
                       inflation_matrix, cells_inflation)
    cv2.imshow('Inflated map', new_grid)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()

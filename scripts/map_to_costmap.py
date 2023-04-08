import numpy as np
import cv2
import yaml
from numba import njit
import time


@njit
def inflate(occupancy_grid, new_grid, cells_inflation):
    for i in range(occupancy_grid.shape[0]):
        for j in range(occupancy_grid.shape[1]):
            if occupancy_grid[i, j] == 0:
                new_grid[max(0, i-cells_inflation):min(occupancy_grid.shape[0], i+cells_inflation), max(
                    0, j-cells_inflation):min(occupancy_grid.shape[1], j+cells_inflation)] = 0
    return new_grid

if __name__ == '__main__':
    # Load map
    img = cv2.imread(
        './maps/map_carto.pgm', cv2.IMREAD_GRAYSCALE)

    # Load map.yaml file
    with open('./maps/map_carto.yaml', 'r') as file:
        params = yaml.safe_load(file)

    resolution = params['resolution']
    origin = params['origin']
    occupancy_thresh = params['occupied_thresh']
    free_thresh = params['free_thresh']
    inflation = 0.5  # in meters
    cells_inflation = int(inflation / resolution)

    # Threshold the image into a binary image
    ret, binary = cv2.threshold(img, occupancy_thresh, 255, cv2.THRESH_BINARY)

    # Convert to occupancy grid
    occupancy_grid = np.zeros((img.shape[0], img.shape[1]))
    occupancy_grid[binary == 255] = 100

    new_grid = occupancy_grid.copy()
    tic = time.time()
    new_grid = inflate(occupancy_grid, new_grid, cells_inflation)
    print("Elapsed time: ", time.time() - tic)
    new_grid = occupancy_grid.copy()
    tic = time.time()
    new_grid = inflate(occupancy_grid, new_grid, cells_inflation)
    print("Elapsed time: ", time.time() - tic)

    cv2.imshow('occupancy_grid', new_grid)
    cv2.waitKey(0)

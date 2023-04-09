import numpy as np
import cv2
import time
from utils import get_inflation_matrix, inflate_local
import yaml
import os


def main():
    map_path = os.path.join(current_path, 'maps', 'map_carto.pgm')
    yaml_path = os.path.join(current_path, 'maps', 'map_carto.yaml')
    # Load map
    img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

    # Load map.yaml file
    with open(yaml_path, 'r') as file:
        params = yaml.safe_load(file)

    resolution = params['resolution']
    occupancy_thresh = params['occupied_thresh']
    origin = -np.array(params['origin']) / resolution
    inflation = 0.22  # in meters
    cells_inflation = int(inflation / resolution)
    size = 3  # in meters
    costmap_size = int(size / resolution)

    rob_x = 0  # in meters
    rob_y = 0  # in meters

    # Threshold the image into a binary image
    ret, binary = cv2.threshold(img, occupancy_thresh, 255, cv2.THRESH_BINARY)

    # Convert to occupancy grid
    occupancy_grid = np.zeros((img.shape[0], img.shape[1]))
    occupancy_grid[binary == 255] = 100

    # Display a red circle in the center of the map
    map_origin = np.array([origin[0], occupancy_grid.shape[0] - origin[1]])
    robot_pos = np.array([map_origin[0] + int(rob_x / resolution),
                          map_origin[1] - int(rob_y / resolution)])
    robot_position = tuple(robot_pos.astype(int))
    inflation_matrix = 1 - ((get_inflation_matrix(cells_inflation, factor=1.5)) / 100)

    # Inflate the map
    local_costmap = inflate_local(occupancy_grid, inflation_matrix, cells_inflation, robot_position, costmap_size)
    cv2.imshow('Local costmap', local_costmap)
    cv2.waitKey(0)


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    # Get the parent path
    current_path = os.path.dirname(current_path)
    main()

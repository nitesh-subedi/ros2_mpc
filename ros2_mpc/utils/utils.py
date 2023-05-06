import numpy as np
from numba import njit


@njit
def convert_laser_scan_to_occupancy_grid(laser_scan_data, angles, map_resolution, map_size):
    # Calculate the size of each cell in the occupancy grid
    cell_size = map_resolution
    angle_min = angles[0]
    angle_max = angles[1]

    # Calculate the number of cells in the occupancy grid
    num_cells = int(map_size / cell_size)

    # Create an empty occupancy grid
    occupancy_grid = np.zeros((num_cells, num_cells))

    # Convert laser scan data to Cartesian coordinates
    angles = np.arange(len(laser_scan_data)) * (angle_max - angle_min) / len(laser_scan_data) + angle_min
    x_coords = laser_scan_data * np.cos(angles)
    y_coords = laser_scan_data * np.sin(angles)

    x_indices = (x_coords + (map_size / 2))
    y_indices = (y_coords + (map_size / 2))

    # Set occupied cells in the occupancy grid
    for i in range(len(x_indices)):
        x, y = x_indices[i], y_indices[i]
        if 0 <= x < num_cells and 0 <= y < num_cells:
            occupancy_grid[int(x), int(y)] = 1

    return occupancy_grid


@njit
def convert_to_map_coordinates(occ_grid, map_resolution=0.8):
    map_origin = np.array([occ_grid.shape[0] // 2, occ_grid.shape[1] // 2]) * map_resolution
    meter_x = occ_grid.copy()
    meter_y = occ_grid.copy()
    for i in range(occ_grid.shape[0]):
        for j in range(occ_grid.shape[1]):
            meter_x[i, j] = - j * map_resolution + map_origin[1]
            meter_y[i, j] = - i * map_resolution + map_origin[0]

    return meter_y, meter_x


@njit
def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


@njit
def rotate_coordinates(coordinates, rotation):
    rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                           [np.sin(rotation), np.cos(rotation)]])

    rotated = np.dot(rot_matrix, coordinates)

    return rotated


def world_to_map(world_x, world_y, map_image, map_info):
    map_coordinates = ((np.array([world_x, world_y]) - map_info['origin']) / map_info['resolution']).astype(np.int32)
    return map_image.shape[0] - map_coordinates[1]


def map_to_world(path, map_image, map_info):
    path = np.array(path)
    path = np.column_stack((path[:, 1], map_image.shape[0] - path[:, 0]))
    # Convert back to world coordinates
    path_xy = path * map_info['resolution'] + map_info['origin']
    return path_xy

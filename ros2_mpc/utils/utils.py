import numpy as np
from numba import njit


@njit
def convert_laser_scan_to_occupancy_grid(laser_scan_data, angles, map_resolution, map_size, rotation=0.0):
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
    # print(type(x_coords))
    coordinates = np.vstack((x_coords, y_coords))
    coordinates_rotated = rotate_coordinates(coordinates, rotation)
    x_coords = coordinates_rotated[0, :]
    y_coords = coordinates_rotated[1, :]
    # Convert nan values to 0
    x_coords[np.isnan(x_coords)] = 0
    y_coords[np.isnan(y_coords)] = 0
    # Convert inf values to max range
    x_coords[np.isinf(x_coords)] = np.max(x_coords[~np.isinf(x_coords)])
    y_coords[np.isinf(y_coords)] = np.max(y_coords[~np.isinf(y_coords)])

    x_indices = (x_coords + (map_size / 2))
    y_indices = (y_coords + (map_size / 2))

    # Set occupied cells in the occupancy grid
    for i in range(len(x_indices)):
        x, y = int(x_indices[i] / cell_size), int(y_indices[i] / cell_size)
        if 0 <= x < num_cells and 0 <= y < num_cells:
            occupancy_grid[int(y), int(x)] = 100

    return occupancy_grid


@njit
def convert_laser_scan_to_xy_coordinates(laser_scan_data, angles, rotation=0.0):
    # Calculate the size of each cell in the occupancy grid
    angle_min = angles[0]
    angle_max = angles[1]

    # Convert laser scan data to Cartesian coordinates
    angles = np.arange(len(laser_scan_data)) * (angle_max - angle_min) / len(laser_scan_data) + angle_min
    x_coords = laser_scan_data * np.cos(angles)
    y_coords = laser_scan_data * np.sin(angles)
    # print(type(x_coords))
    coordinates = np.vstack((x_coords, y_coords))
    coordinates_rotated = rotate_coordinates(coordinates, rotation)
    x_coords = coordinates_rotated[0, :]
    y_coords = coordinates_rotated[1, :]
    # Convert nan values to 0
    x_coords[np.isnan(x_coords)] = np.min(x_coords)
    y_coords[np.isnan(y_coords)] = np.min(y_coords)
    # Convert inf values to max range
    x_coords[np.isinf(x_coords)] = np.max(x_coords[~np.isinf(x_coords)])
    y_coords[np.isinf(y_coords)] = np.max(y_coords[~np.isinf(y_coords)])

    return x_coords, y_coords


@njit
def convert_xy_coordinates_to_occ_grid(x_coords, y_coords, map_size, map_resolution, map_origin):
    # Calculate the size of each cell in the occupancy grid
    cell_size = map_resolution
    # Calculate the number of cells in the occupancy grid
    num_cells_x = map_size[0]
    num_cells_y = map_size[1]

    # Create an empty occupancy grid
    occupancy_grid = np.zeros((num_cells_x, num_cells_y))

    # Convert laser scan data to Cartesian coordinates
    x_indices = (x_coords - map_origin[0]) / cell_size
    y_indices = (y_coords - map_origin[1]) / cell_size

    # Set occupied cells in the occupancy grid
    for i in range(len(x_indices)):
        x, y = int(x_indices[i]), int(y_indices[i])
        if 0 <= x < num_cells_x and 0 <= y < num_cells_y:
            occupancy_grid[int(y), int(x)] = 100

    return occupancy_grid


@njit
def convert_occ_grid_to_xy_coordinates(occ_grid, map_resolution, map_origin):
    occ_grid = np.flipud(occ_grid)
    # Calculate the number of cells in the occupancy grid
    num_cells_x = int(occ_grid.shape[0])
    num_cells_y = int(occ_grid.shape[1])

    # Create an empty occupancy grid
    x_coords = []
    y_coords = []
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            if occ_grid[i, j] == 255:
                x_coords.append(j * map_resolution + map_origin[0])
                y_coords.append(i * map_resolution + map_origin[1])

    return x_coords, y_coords


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
    map_coordinates[1] = map_image.shape[0] - map_coordinates[1]
    return map_coordinates


def map_to_world(path, map_image, map_info):
    path = np.array(path)
    try:
        path = np.column_stack((path[:, 1], map_image.shape[0] - path[:, 0]))
    except IndexError:
        return None
    # Convert back to world coordinates
    path_xy = path * map_info['resolution'] + map_info['origin']
    return path_xy

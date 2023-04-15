import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import cv2


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
    angles = np.linspace(angle_min, angle_max, len(laser_scan_data))
    x_coords = laser_scan_data * np.cos(angles)
    y_coords = laser_scan_data * np.sin(angles)

    # Convert Cartesian coordinates to occupancy grid indices
    x_indices = np.array((x_coords + (map_size / 2)) / cell_size, dtype=int)
    y_indices = np.array((y_coords + (map_size / 2)) / cell_size, dtype=int)

    # Set occupied cells in the occupancy grid
    for x, y in zip(x_indices, y_indices):
        if 0 <= x < num_cells and 0 <= y < num_cells:
            occupancy_grid[x, y] = 1

    return occupancy_grid


class LaserSubscriber(Node):
    def __init__(self):
        super().__init__('laser_subscriber')
        self.angles = None
        self.laser_data = None
        self.map_subscriber = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

    def laser_callback(self, msg):
        self.laser_data = np.array(msg.ranges)
        self.angles = np.array([msg.angle_min, msg.angle_max])
        self.get_logger().info("Data received!")

    def get_scan(self):
        rclpy.spin_once(self)
        time.sleep(0.1)
        return self.laser_data, self.angles


def convert_to_map_coordinates(x, y, occ_grid, map_resolution=0.8):
    map_origin = np.array([occ_grid.shape[0] // 2, occ_grid.shape[1] // 2]) * map_resolution
    x_meters = - x * map_resolution + map_origin[1]
    y_meters = - y * map_resolution + map_origin[0]
    return y_meters, x_meters


# x_per_start, y_per_start = (100 - msg['data']['start']['x']) / \
#             100, (100 - msg['data']['start']['y']) / 100
#         x_per_end, y_per_end = (100 - msg['data']['end']['x']) / \
#             100, (100 - msg['data']['end']['y']) / 100
#         print(x_per_start, y_per_start)
#         x_start, y_start = robot.convert_to_map_coordinates(
#             x_per_start, y_per_start, mode='mapping')


def main(args=None):
    rclpy.init(args=args)
    laser_node = LaserSubscriber()
    scale = 5
    while rclpy.ok():
        scan_data, angles = laser_node.get_scan()
        if scan_data is None:
            continue
        occ_grid = 1 - convert_laser_scan_to_occupancy_grid(scan_data, angles, 0.8, 2 * 2)
        x, y = convert_to_map_coordinates(4, 4, occ_grid=occ_grid)
        print(x, y)
        break
        # occ_grid = cv2.rotate(occ_grid, cv2.ROTATE_180)
        # obstacles = np.where(occ_grid == 100)
        # print(occ_grid)
        # print(occ_grid.shape)
        # center = occ_grid.shape[0] // 2, occ_grid.shape[1] // 2
        # occ_grid = cv2.resize(occ_grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        #
        # cv2.imshow("occ", occ_grid)
        # if (cv2.waitKey(1) & 0xFF) == ord('q'):
        #     break


if __name__ == "__main__":
    main()

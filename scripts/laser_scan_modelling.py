import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import cv2
from numba import njit
import math
from matplotlib import pyplot as plt


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


class OdomSubscriber(Node):
    def __init__(self):
        super().__init__('odom_subscriber')
        self.orientation = None
        self.position = None
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, msg):
        self.position = np.array([msg.pose.pose.position.x + 3.0, msg.pose.pose.position.y - 1.0])
        self.orientation = np.array(euler_from_quaternion(
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w))
        self.get_logger().info("Data received!")

    def get_states(self):
        rclpy.spin_once(self)
        time.sleep(0.1)
        return self.position, self.orientation


def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


@njit
def convert_to_map_coordinates(occ_grid, map_resolution=0.8):
    map_origin = np.array([occ_grid.shape[0] // 2, occ_grid.shape[1] // 2]) * map_resolution
    meter_x = occ_grid.copy()
    meter_y = occ_grid.copy()
    for i in range(occ_grid.shape[0]):
        for j in range(occ_grid.shape[1]):
            meter_x[i, j] = - j * map_resolution + map_origin[1]
            meter_y[i, j] = - i * map_resolution + map_origin[0]

    # rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
    #                        [np.sin(rotation), np.cos(rotation)]])
    #
    # x_new = np.dot(rot_matrix, meter_x)
    # y_new = np.dot(rot_matrix, meter_y)

    return meter_y, meter_x


# x_per_start, y_per_start = (100 - msg['data']['start']['x']) / \
#             100, (100 - msg['data']['start']['y']) / 100
#         x_per_end, y_per_end = (100 - msg['data']['end']['x']) / \
#             100, (100 - msg['data']['end']['y']) / 100
#         print(x_per_start, y_per_start)
#         x_start, y_start = robot.convert_to_map_coordinates(
#             x_per_start, y_per_start, mode='mapping')

def rotate_image(img, angle):
    # get image dimensions
    (h, w) = img.shape[:2]

    # calculate the center of the image
    center = (w // 2, h // 2)

    # get the rotation matrix for our chosen angle
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # rotate the image
    rotated_img = cv2.warpAffine(img, M, (w, h))

    return rotated_img


def rotate_coordinates(coordinates, rotation):
    rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                           [np.sin(rotation), np.cos(rotation)]])

    rotated = np.dot(rot_matrix, coordinates)

    return rotated


def main(args=None):
    rclpy.init(args=args)
    laser_node = LaserSubscriber()
    odom_node = OdomSubscriber()
    scale = 5
    resolution = 0.1
    size = 2.5
    while rclpy.ok():
        scan_data, angles = laser_node.get_scan()
        pos, ori = odom_node.get_states()
        if scan_data is None:
            continue
        occ_grid = 1 - convert_laser_scan_to_occupancy_grid(scan_data, angles, resolution, size * 2)
        occ_grid = cv2.rotate(occ_grid, cv2.ROTATE_180)
        x, y = convert_to_map_coordinates(occ_grid=occ_grid, map_resolution=resolution)
        # print(time.time() - tic)
        obstacles_indices = np.where(occ_grid == 0)
        obs_x, obs_y = x[obstacles_indices], y[obstacles_indices]
        obstacle_array = np.array([obs_x, obs_y])
        rotated_obstacle = rotate_coordinates(obstacle_array, ori[2])
        rotated_obstacle[0, :] += pos[0]
        rotated_obstacle[1, :] += pos[1]
        plt.scatter(rotated_obstacle[0], rotated_obstacle[1])
        # plt.scatter(obs_x, obs_y)
        plt.show()
        pass

        # print(x, y)
        # pass
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

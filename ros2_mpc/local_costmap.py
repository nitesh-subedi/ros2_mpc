import numpy as np
import cv2
import time
from utils import get_inflation_matrix, inflate_local
import yaml
import os
import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_msgs.msg import TFMessage
# from matplotlib import pyplot as plt
from nav_msgs.msg import OccupancyGrid, Odometry
from numba import njit


class MapSubscriber(Node):
    def __init__(self):
        super().__init__("map_subscriber")
        self.map = None
        self.map_info = None
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

    def map_callback(self, msg):
        map_origin = msg.info.origin.position
        map_resolution = msg.info.resolution
        self.map_info = {'resolution': map_resolution, 'origin': map_origin}
        height = msg.info.height
        width = msg.info.width
        self.map = np.array(msg.data).reshape((height, width))

    def get_map(self):
        rclpy.spin_once(self)
        return self.map, self.map_info


class OdomSubscriber(Node):
    def __init__(self):
        super().__init__("odom_subscriber")
        self.robot_pos = None
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, msg):
        x_pos, y_pos = msg.pose.pose.position.x + 3.0, msg.pose.pose.position.y - 1.0
        self.robot_pos = np.array([x_pos, y_pos])

    def get_odom(self):
        rclpy.spin_once(self)
        return self.robot_pos


class RobotPositionSubscriber(Node):
    def __init__(self):
        super().__init__("robot_position_subscriber")
        self.map_to_odom = None
        self.odom_to_base_footprint = None
        self.robot_position = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.subscription = self.create_subscription(TFMessage, "/tf", self.listener_callback, 10)

    def listener_callback(self, data):
        if not data.transforms:
            pass
        else:
            for transform in data.transforms:
                if transform.child_frame_id == "odom" and transform.header.frame_id == "map":
                    self.map_to_odom = np.array([transform.transform.translation.x, transform.transform.translation.y])
                if transform.child_frame_id == "base_footprint" and transform.header.frame_id == "odom":
                    self.odom_to_base_footprint = np.array(
                        [transform.transform.translation.x, transform.transform.translation.y])
                if self.map_to_odom is not None and self.odom_to_base_footprint is not None:
                    self.robot_position = self.map_to_odom + self.odom_to_base_footprint
                    self.get_logger().info(
                        "Robot position on map: x = %f, y = %f" % (self.robot_position[0], self.robot_position[1]))

    def get_robot_position(self):
        rclpy.spin_once(self)
        time.sleep(0.1)
        return self.robot_position


class CostmapPublisher(Node):
    def __init__(self):
        super().__init__("costmap_publisher")
        self.publisher = self.create_publisher(OccupancyGrid, "/my_local_costmap", 10)
        self.msg = OccupancyGrid()

    def publish_costmap(self, costmap, costmap_size, robot_pos):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.msg.header.frame_id = "map"
        self.msg.info.width = costmap.shape[1]
        self.msg.info.height = costmap.shape[0]
        self.msg.info.origin.position.x = robot_pos[0] + (-costmap_size / 2) * 0.05
        self.msg.info.origin.position.y = robot_pos[1] + (-costmap_size / 2) * 0.05
        self.msg.info.resolution = 0.05
        self.msg.data = costmap.flatten().tolist()
        self.publisher.publish(self.msg)
        self.get_logger().info("Costmap Published!")


@njit
def get_grid(data):
    M = 65
    N = 50
    height = data.shape[0]
    width = data.shape[1]
    # data = list(data)
    for y in range(width):
        for x in range(height):
            if data[x, y] >= M:
                data[x, y] = 100
            elif data[x, y] < N:
                data[x, y] = 0

    return data


def main(args=None):
    rclpy.init(args=args)
    robot = OdomSubscriber()
    costmap_publisher = CostmapPublisher()
    map_subscriber = MapSubscriber()
    occupancy_thresh = 65
    inflation = 0.3  # in meters
    costmap_m_size = 2  # in meters

    while rclpy.ok():
        # time.sleep(2)
        # Get the robot position
        robot_position = robot.get_odom()
        while robot_position is None:
            robot_position = robot.get_odom()
            time.sleep(0.1)

        # Get the map
        map_image, map_info = map_subscriber.get_map()
        resolution = map_info['resolution']
        map_origin = map_info['origin']
        costmap_size = int(costmap_m_size / resolution)
        cells_inflation = int(inflation / resolution)
        # map_image = map_image.astype(np.uint8)
        # print("map shape: ", map_image.shape)
        # occupancy_grid = map_image.copy()
        # mask = map_image > 50
        # occupancy_grid[mask] = 0
        # mask = map_image < 25
        # map_image[mask] = 100
        # occupancy_grid = map_image
        occupancy_grid = get_grid(map_image)
        # ret, occupancy_grid = cv2.threshold(map_image, occupancy_thresh, 100, cv2.THRESH_BINARY)
        # cv2.imwrite('occ.png', occupancy_grid)
        # pass
        # occupancy_grid = np.zeros((map_image.shape[0], map_image.shape[1]))
        # occupancy_grid[binary == 255] = 100

        rob_x = robot_position[0]  # in meters
        rob_y = robot_position[1]  # in meters

        robot_pos = np.array([-map_origin.x / resolution + int(rob_x / resolution),
                              -map_origin.y / resolution + int(rob_y / resolution)])
        robot_position = tuple(robot_pos.astype(int))
        inflation_matrix = get_inflation_matrix(cells_inflation, factor=2)

        # Inflate the map
        local_costmap = inflate_local(occupancy_grid, inflation_matrix, cells_inflation, robot_position,
                                      costmap_size).astype(int)
        # # Flip the local costmap
        # local_costmap = cv2.flip(local_costmap, 0)
        # # Invert the values
        # local_costmap = 100 - local_costmap
        # print(local_costmap)
        # if local_costmap is not None:
        #     cv2.imshow('local_costmap', local_costmap)
        #     if cv2.waitKey(0) & 0xFF == ord('q'):
        #         break
        robot_pos = np.array([rob_x, rob_y])
        costmap_publisher.publish_costmap(local_costmap, costmap_size, robot_pos)
        # cv2.imshow('local_costmap', local_costmap)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    # Get the parent path
    current_path = os.path.dirname(current_path)
    main()

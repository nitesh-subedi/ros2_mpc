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
from matplotlib import pyplot as plt
from nav_msgs.msg import OccupancyGrid


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
                    # self.get_logger().info("Odom to map: x = %f, y = %f" % (transform.transform.translation.x,
                    # transform.transform.translation.y))
                if transform.child_frame_id == "base_footprint" and transform.header.frame_id == "odom":
                    self.odom_to_base_footprint = np.array(
                        [transform.transform.translation.x, transform.transform.translation.y])
                    # self.get_logger().info("Base footprint to odom: x = %f, y = %f" % (
                    # transform.transform.translation.x, transform.transform.translation.y))
                if self.map_to_odom is not None and self.odom_to_base_footprint is not None:
                    self.robot_position = self.map_to_odom + self.odom_to_base_footprint
                    self.get_logger().info(
                        "Robot position on map: x = %f, y = %f" % (self.robot_position[0], self.robot_position[1]))
    
    def get_robot_position(self):
        self.get_logger().info("Waiting for robot position...")
        rclpy.spin_once(self)
        time.sleep(0.1)
        return self.robot_position


class Costmap_publisher(Node):
    def __init__(self):
        super().__init__("costmap_publisher")
        self.publisher = self.create_publisher(OccupancyGrid, "/my_local_costmap", 10)
        self.msg = OccupancyGrid()
    
    def publish_costmap(self, costmap, costmap_size):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.msg.header.frame_id = "map"
        self.msg.info.width = costmap.shape[1]
        self.msg.info.height = costmap.shape[0]
        # self.msg.info.origin.position.x = (-costmap_size / 2) * 0.05
        # self.msg.info.origin.position.y = (-costmap_size / 2) * 0.05
        self.msg.info.resolution = 0.05
        self.msg.data = costmap.flatten().tolist()
        self.publisher.publish(self.msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotPositionSubscriber()
    costmap_publisher = Costmap_publisher()

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
    # Threshold the image into a binary image
    ret, binary = cv2.threshold(img, occupancy_thresh, 255, cv2.THRESH_BINARY)

    # Convert to occupancy grid
    occupancy_grid = np.zeros((img.shape[0], img.shape[1]))
    occupancy_grid[binary == 255] = 100

    # Display a red circle in the center of the map
    map_origin = np.array([origin[0], occupancy_grid.shape[0] - origin[1]])


    while rclpy.ok():
        # time.sleep(0.1)
        # Get the robot position
        robot_position = node.get_robot_position()
        while robot_position is None:    
            robot_position = node.get_robot_position()
            print("Robot position is None!")
            time.sleep(0.1)
        rob_x = robot_position[0]  # in meters
        rob_y = robot_position[1]  # in meters
        # rob_x = 0  # in meters
        # rob_y = 0  # in meters

        
        robot_pos = np.array([map_origin[0] + int(rob_x / resolution),
                            map_origin[1] - int(rob_y / resolution)])
        robot_position = tuple(robot_pos.astype(int))
        inflation_matrix = 1 - ((get_inflation_matrix(cells_inflation, factor=1.5)) / 100)

        # Inflate the map
        local_costmap = inflate_local(occupancy_grid, inflation_matrix, cells_inflation, robot_position, costmap_size).astype(int)
        # print(local_costmap)
        costmap_publisher.publish_costmap(local_costmap, costmap_size)
        # print("Localcostmap published!")
    #     plt.imshow(local_costmap)
           

    # plt.show()
        # cv2.imshow('Local costmap', local_costmap)
        # if cv2.waitKey(0) == ord('q'):
        #     cv2.destroyAllWindows()
        #     break
            


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    # Get the parent path
    current_path = os.path.dirname(current_path)
    main()

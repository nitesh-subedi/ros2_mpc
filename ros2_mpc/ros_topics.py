import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
from ros2_mpc import utils
import time


class MapSubscriber(Node):
    def __init__(self):
        super().__init__('map_subscriber')
        self.map_image = None
        self.map_info = None
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

    def map_callback(self, msg):
        map_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        map_resolution = msg.info.resolution
        self.map_info = {'resolution': map_resolution, 'origin': map_origin}
        map_width = msg.info.width
        map_height = msg.info.height
        self.map_image = np.array(msg.data).reshape(map_height, map_width)
        # Convert the map image to grayscale binary image
        self.map_image[self.map_image < 50] = 1
        self.map_image[self.map_image > 60] = 0
        # Invert the map image
        self.map_image = 1 - self.map_image
        # self.map_image[self.map_image < 1] = 0
        self.map_image = self.map_image.astype(np.uint8) * 255
        # Flip the map image
        self.map_image = np.flipud(self.map_image)
        # self.get_logger().info("Map received!")

    def get_map(self):
        rclpy.spin_once(self)
        return self.map_image, self.map_info


class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pub = Twist()

    def publish_cmd(self, v, w):
        self.pub.linear.x = v
        self.pub.angular.z = w
        self.cmd_publisher.publish(self.pub)
        # self.get_logger().info("cmd published!")


class OdomSubscriber(Node):
    def __init__(self):
        super().__init__('odom_subscriber')
        self.velocities = None
        self.orientation = None
        self.position = None
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, msg):
        self.position = np.array([msg.pose.pose.position.x + 3.0, msg.pose.pose.position.y - 1.0]).round(decimals=2)
        self.orientation = np.array(utils.euler_from_quaternion(
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)).round(decimals=2)
        self.velocities = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z]).round(decimals=2)
        # self.get_logger().info("Odom received!")

    def get_states(self):
        rclpy.spin_once(self)
        time.sleep(0.1)
        return self.position, self.orientation, self.velocities


class LaserSubscriber(Node):
    def __init__(self):
        super().__init__('laser_subscriber')
        self.angles = None
        self.laser_data = None
        self.map_subscriber = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

    def laser_callback(self, msg):
        self.laser_data = np.array(msg.ranges)
        self.angles = np.array([msg.angle_min, msg.angle_max])
        # self.get_logger().info("Data received!")

    def get_scan(self):
        rclpy.spin_once(self)
        time.sleep(0.1)
        return self.laser_data, self.angles

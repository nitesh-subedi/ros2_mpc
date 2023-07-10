import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
from ros2_mpc import utils


class MapSubscriber(Node):
    def __init__(self):
        super().__init__("map_subscriber")
        self.map_image = None
        self.map_info = None
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 10
        )

    def map_callback(self, msg):
        map_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        map_resolution = msg.info.resolution
        self.map_info = {"resolution": map_resolution, "origin": map_origin}
        map_width = msg.info.width
        map_height = msg.info.height
        self.map_image = np.array(msg.data).reshape(map_height, map_width)
        # Convert the map image to grayscale binary image
        self.map_image[self.map_image <= 60] = 1
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
        super().__init__("cmd_vel_publisher")
        self.cmd_publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.pub = Twist()

    def publish_cmd(self, v, w):
        self.pub.linear.x = v
        self.pub.angular.z = w
        self.cmd_publisher.publish(self.pub)
        # self.get_logger().info("cmd published!")


class OdomSubscriber(Node):
    def __init__(self):
        super().__init__("odom_subscriber")
        self.velocities = None
        self.orientation = None
        self.position = None
        self.odom_subscriber = self.create_subscription(
            Odometry, "/robot_position", self.odom_callback, 10
        )

    def odom_callback(self, msg):
        self.position = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y]
        ).round(decimals=2)
        self.orientation = np.array(
            utils.euler_from_quaternion(
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            )
        ).round(decimals=2)
        self.velocities = np.array(
            [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        ).round(decimals=2)
        # self.get_logger().info("Odom received!")

    def get_states(self):
        rclpy.spin_once(self)
        # time.sleep(0.1)
        return self.position, self.orientation


class LaserSubscriber(Node):
    def __init__(self):
        super().__init__("laser_subscriber")
        self.angles = None
        self.laser_data = None
        self.map_subscriber = self.create_subscription(
            LaserScan, "/scan", self.laser_callback, 10
        )

    def laser_callback(self, msg):
        self.laser_data = np.array(msg.ranges)
        self.angles = np.array([msg.angle_min, msg.angle_max])
        # self.get_logger().info("Data received!")

    def get_scan(self):
        rclpy.spin_once(self)
        # time.sleep(0.1)
        return self.laser_data, self.angles


class GoalSubscriber(Node):
    def __init__(self):
        super().__init__("goal_subscriber")
        self.goal = None
        self.goal_subscriber = self.create_subscription(
            PoseStamped, "/goal_pose", self.goal_callback, 10
        )

    def goal_callback(self, msg):
        goal_xy = np.array([msg.pose.position.x, msg.pose.position.y]).round(decimals=2)
        goal_theta = np.array(
            utils.euler_from_quaternion(
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            )
        ).round(decimals=2)
        self.goal = np.concatenate((goal_xy, goal_theta))
        # self.get_logger().info("Goal received!")

    def get_goal(self):
        rclpy.spin_once(self, timeout_sec=0.05)
        return self.goal

    def get_new_goal(self):
        rclpy.spin_once(self)
        return self.goal


class LocalCostmapPublisher(Node):
    def __init__(self):
        super().__init__("costmap_publisher")
        self.publisher = self.create_publisher(OccupancyGrid, "/my_local_costmap", 10)
        self.msg = OccupancyGrid()

    def publish_costmap(self, costmap, costmap_size, robot_pos):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.msg.header.frame_id = "map"
        self.msg.info.width = costmap.shape[1]
        self.msg.info.height = costmap.shape[0]
        self.msg.info.origin.position.x = robot_pos[0] + (-costmap_size / 2)
        self.msg.info.origin.position.y = robot_pos[1] + (-costmap_size / 2)
        self.msg.info.resolution = 0.05
        self.msg.data = costmap.flatten().tolist()
        self.publisher.publish(self.msg)
        # self.get_logger().info("Local Costmap Published!")


class GlobalCostmapPublisher(Node):
    def __init__(self):
        super().__init__("costmap_publisher")
        self.publisher = self.create_publisher(OccupancyGrid, "/my_global_costmap", 10)
        self.msg = OccupancyGrid()

    def publish_costmap(self, costmap, origin):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.msg.header.frame_id = "map"
        self.msg.info.width = costmap.shape[1]
        self.msg.info.height = costmap.shape[0]
        self.msg.info.origin.position.x = origin[0]
        self.msg.info.origin.position.y = origin[1]
        self.msg.info.resolution = 0.05
        self.msg.data = costmap.flatten().tolist()
        self.publisher.publish(self.msg)
        self.get_logger().info("Global Costmap Published!")

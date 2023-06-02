import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from ros2_mpc.planner.global_planner import GlobalPlanner
import numpy as np
from ros2_mpc.ros_topics import OdomSubscriber, MapSubscriber, GoalSubscriber
from ros2_mpc import utils
import cv2
import time
import yaml
from ament_index_python.packages import get_package_share_directory
import os


def get_headings(path_xy, dt):
    # Compute the heading angle
    path_heading = np.arctan2(path_xy[1:, 1] - path_xy[:-1, 1], path_xy[1:, 0] - path_xy[:-1, 0])
    path_heading = np.append(path_heading, path_heading[-1])
    # Compute the angular velocity
    path_omega = (path_heading[1:] - path_heading[:-1]) / 2
    # Compute the velocity
    path_velocity = (np.linalg.norm(path_xy[1:, :] - path_xy[:-1, :], axis=1) / dt) * 2
    path_velocity = np.append(path_velocity, path_velocity[-1])
    return path_heading, path_velocity, path_omega


class PathPublisher(Node):
    def __init__(self):
        super().__init__("goal_publisher")
        self.publisher = self.create_publisher(Path, 'my_path', 10)
        self.pose = PoseStamped()
        self.msg = Path()

    def publish_path(self, path, heading_angles):
        self.msg.header.frame_id = "map"
        self.msg.header.stamp = self.get_clock().now().to_msg()
        for i in range(len(path)):
            self.pose.header.frame_id = "map"
            self.pose.header.stamp = self.get_clock().now().to_msg()
            self.pose.pose.position.x = path[i, 0]
            self.pose.pose.position.y = path[i, 1]
            self.pose.pose.position.z = 0.0
            self.pose.pose.orientation.x = 0.0
            self.pose.pose.orientation.y = 0.0
            self.pose.pose.orientation.z = np.sin(heading_angles[i] / 2)
            self.pose.pose.orientation.w = np.cos(heading_angles[i] / 2)
            self.msg.poses.append(self.pose)
        self.publisher.publish(self.msg)
        # self.get_logger().info("Path Published!")


def dilate_image(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    image = cv2.dilate(image, kernel, iterations=1)
    return image.astype(np.uint8)


def main():
    rclpy.init()
    path_publisher = PathPublisher()
    map_node = MapSubscriber()
    odom_node = OdomSubscriber()
    planner = GlobalPlanner()
    goal_listener = GoalSubscriber()
    project_path = get_package_share_directory('ros2_mpc')
    # get the goal position from the yaml file
    with open(os.path.join(project_path, 'config/params.yaml'), 'r') as file:
        params = yaml.safe_load(file)
    dt = float(params['dt'])
    # goal_xy = np.array([5.0, -0.6])
    map_node.get_logger().info('Waiting for map...')
    goal_listener.get_logger().info("Waiting for goal!")
    # goal_xy = goal_listener.get_goal()[:2]
    path_last = None
    while True:
        try:
            goal_xy = goal_listener.get_goal()[:2]
        except TypeError:
            continue
        map_image, map_info = map_node.get_map()
        pos, ori, velocity = odom_node.get_states()
        # Dilate the map image
        map_image = dilate_image(map_image, 5)
        # Get the current position of the robot
        robot_on_map = utils.world_to_map(pos[0], pos[1], map_image, map_info)
        start = (robot_on_map[1], robot_on_map[0])
        # Get the goal position of the robot
        goal_on_map = utils.world_to_map(goal_xy[0], goal_xy[1], map_image, map_info)
        # Swap the x and y coordinates
        goal = (goal_on_map[1], goal_on_map[0])
        # Check if the path is empty
        if len(planner.get_path(start, goal, map_image)) == 0:
            path_publisher.get_logger().warning("Path empty. Using last path as reference!")
            path = path_last
        else:
            path = planner.get_path(start, goal, map_image)
            path_last = path
        if path_last is None:
            path_publisher.get_logger().error("Goal Unreachable!")
        # Convert the path to world coordinates
        path_xy = utils.map_to_world(path, map_image, map_info)
        # Compute the headings
        try:
            path_heading, _, _ = get_headings(path_xy, dt)
            path_publisher.publish_path(path_xy, path_heading)
        except IndexError:
            path_publisher.get_logger().info("Goal Reached!")
        time.sleep(0.1)
    # path_publisher.destroy_node()
    # rclpy.shutdown()


if __name__ == '__main__':
    main()

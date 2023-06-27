import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from ros2_mpc.planner.global_planner import AstarGlobalPlanner, RRTGlobalPlanner, AStarPlanner2
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
        self.publisher = self.create_publisher(Path, 'smoothed_plan', 10)

    def publish_path(self, smooth_x, smooth_y):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for i in range(len(smooth_x)):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(smooth_x[i])
            pose.pose.position.y = float(smooth_y[i])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.publisher.publish(msg)
        # self.get_logger().info('Smoothed path published')


def erode_image(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    image = cv2.dilate(image, kernel, iterations=2)
    return image.astype(np.uint8)


def main(args=None):
    rclpy.init(args=args)
    path_publisher = PathPublisher()
    map_node = MapSubscriber()
    odom_node = OdomSubscriber()
    planner = AStarPlanner2()
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
        # time.sleep(1.0)
        try:
            goal_xy = goal_listener.get_goal()[:2]
        except TypeError:
            continue
        map_image, map_info = map_node.get_map()
        # planner = RRTGlobalPlanner(map_image)
        pos, ori = odom_node.get_states()
        if pos is None:
            continue
        # Dilate the map image
        map_image = erode_image(map_image, 5)
        cv2.imwrite('/home/nitesh/projects/ros2_ws/src/ros2_mpc/ros2_mpc/scripts/map.png', map_image)
        # Get the current position of the robot
        robot_on_map = utils.world_to_map(pos[0], pos[1], map_image, map_info)
        # # Log the current position of the robot
        # path_publisher.get_logger().info("Robot Position: {}".format(robot_on_map))
        # # Log map info
        # path_publisher.get_logger().info("Map Info: {}".format(map_info))
        # # Log robot position
        # path_publisher.get_logger().info("Robot xy Position: {}".format(pos))
        start = (robot_on_map[1], robot_on_map[0])
        # Get the goal position of the robot
        goal_on_map = utils.world_to_map(goal_xy[0], goal_xy[1], map_image, map_info)
        # Swap the x and y coordinates
        goal = (goal_on_map[1], goal_on_map[0])
        # Check if the path is empty
        # path_publisher.get_logger().info("Start: {}".format(start))
        # path_publisher.get_logger().info("Goal: {}".format(goal))
        path = planner.get_path(start, goal, map_image)
        if len(path) == 0:
            path_publisher.get_logger().warning("Path empty. Using last path as reference!")
            path = path_last
        else:
            # path = planner.get_path(np.array(start), np.array(goal))
            path_last = path
        if path_last is None:
            path_publisher.get_logger().error("Goal Unreachable!")
            continue
        # Convert the path to world coordinates
        path_xy = utils.map_to_world(path, map_image, map_info)
        if path_xy is None:
            path_publisher.get_logger().error("Goal Unreachable!")
            continue
        # Compute the headings
        try:
            # path_heading, _, _ = get_headings(path_xy, dt)
            path_publisher.publish_path(path_xy[:, 0], path_xy[:, 1])
            if len(path_xy) <= 5:
                path_publisher.get_logger().info("Goal Reached!")
                goal_listener.get_logger().info("Waiting for goal!")
                rclpy.spin_once(goal_listener)
        except IndexError:
            path_publisher.get_logger().info("Goal Reached!")
            goal_listener.get_logger().info("Waiting for goal!")
            rclpy.spin_once(goal_listener)
    # path_publisher.destroy_node()
    # rclpy.shutdown()


if __name__ == '__main__':
    main()

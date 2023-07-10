import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from ros2_mpc.planner.global_planner import AStarPlanner2
import numpy as np
from ros2_mpc.core.ros_topics import OdomSubscriber, MapSubscriber, GoalSubscriber
from ros2_mpc import utils
import cv2
from ament_index_python.packages import get_package_share_directory


def get_headings(path_xy):
    """
    The function get_headings takes a path in the world frame and returns the heading of the robot at each
    point in the path, as well as the x and y components of the heading vector.

    :param path_xy: The path in the world frame
    :return: The heading of the robot at each point in the path, as well as the x and y components of the
    heading vector.
    """
    # Compute the heading vector
    path_heading = np.arctan2(np.diff(path_xy[:, 1]), np.diff(path_xy[:, 0]))
    path_heading = np.append(path_heading, path_heading[-1])
    return path_heading


class PathPublisher(Node):
    def __init__(self):
        super().__init__("goal_publisher")
        self.publisher = self.create_publisher(Path, 'smoothed_plan', 10)

    def publish_path(self, path_xy, path_heading):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for i in range(len(path_xy[:, 0])):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(path_xy[:, 0][i])
            pose.pose.position.y = float(path_xy[:, 1][i])
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = np.sin(path_heading[i] / 2)
            pose.pose.orientation.w = np.cos(path_heading[i] / 2)
            msg.poses.append(pose)

        self.publisher.publish(msg)
        # self.get_logger().info('Smoothed path published')


def erode_image(image, kernel_size):
    """
    The function erode_image takes an image and a kernel size as input, and applies erosion to the image
    using the specified kernel size.
    
    :param image: The input image that you want to erode
    :param kernel_size: The kernel_size parameter specifies the size of the kernel used for erosion. It
    determines the extent of the erosion operation on the image. A larger kernel size will result in a
    more significant erosion effect on the image
    :return: the eroded image as a numpy array of type uint8.
    """
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
        map_image = erode_image(map_image, 8)
        # cv2.imwrite('/home/nitesh/projects/ros2_ws/src/ros2_mpc/ros2_mpc/scripts/map.png', map_image)
        # Get the current position of the robot
        robot_on_map = utils.world_to_map(pos[0], pos[1], map_image, map_info)
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
            path_heading = get_headings(path_xy)
            # Publish path with headings
            path_publisher.publish_path(path_xy, path_heading)
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

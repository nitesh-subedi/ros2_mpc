import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
import numpy as np
from geometry_msgs.msg import PoseStamped

from ros2_mpc.planner.local_planner_point_stabilization import Mpc
from ros2_mpc.core.ros_topics import OdomSubscriber, CmdVelPublisher, GoalSubscriber, LaserSubscriber
from ros2_mpc import utils
import time
import yaml
from ament_index_python.packages import get_package_share_directory
import os


def get_goal_for_mpc(path_xy, path_heading, goal, pos, lookahead_dist_=0.5):
    if np.linalg.norm(goal[:2] - pos[:2]) < lookahead_dist_:
        goal_pose = np.array([goal[0], goal[1], goal[4] % (2 * np.pi)])
    else:
        # Find the nearest point on the path that is greater than lookahead distance
        dist = np.linalg.norm(path_xy - pos[:2], axis=1)
        idx = np.where(dist > lookahead_dist_)[0]
        if len(idx) == 0:
            idx = np.argmin(dist)
        else:
            idx = idx[0]
        goal_pose = np.append(path_xy[idx], path_heading[idx] % (2 * np.pi))
        # Add orientation to goal
        # goal_pose = np.append(goal_pose, 0.0)
    return goal_pose


# def orientation_error(goal_pose, pos):
#     # Calculate orientation error
#     goal_heading = goal_pose[2]
#     current_heading = pos[2]
#     orientation_error = goal_heading - current_heading
#     if orientation_error > np.pi:
#         orientation_error = orientation_error - 2 * np.pi
#     elif orientation_error < -np.pi:
#         orientation_error = orientation_error + 2


class GoalPointPublisher(Node):
    def __init__(self):
        super().__init__('goal_point_publisher')
        self.goal_point = None
        self.publisher_ = self.create_publisher(PoseStamped, 'goal_point', 10)

    def publish_goal_point(self, goal_point):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.pose.position.x = goal_point[0]
        msg.pose.position.y = goal_point[1]
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = np.sin(goal_point[2] / 2)
        msg.pose.orientation.w = np.cos(goal_point[2] / 2)
        self.publisher_.publish(msg)


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.path_xy = None
        self.path_heading = None
        self.create_subscription(Path, 'smoothed_plan', self.path_callback, 10)

    def path_callback(self, msg):
        path = np.zeros((len(msg.poses), 2))
        headings = np.zeros((len(msg.poses), 1))
        for i in range(len(msg.poses)):
            path[i, 0] = msg.poses[i].pose.position.x
            path[i, 1] = msg.poses[i].pose.position.y
            headings[i] = utils.euler_from_quaternion(msg.poses[i].pose.orientation.x,
                                                      msg.poses[i].pose.orientation.y,
                                                      msg.poses[i].pose.orientation.z,
                                                      msg.poses[i].pose.orientation.w)[2]
        self.path_xy = path
        self.path_heading = headings

    def get_path(self):
        rclpy.spin_once(self, timeout_sec=0.1)
        return self.path_xy, self.path_heading


def get_obstacles(scan_data, angles, size, resolution, pos, ori, obstacles_x, obstacles_y):
    occ_grid = 1 - (utils.convert_laser_scan_to_occupancy_grid(scan_data, angles, resolution, size * 2) / 100)
    occ_grid = np.rot90(occ_grid, k=2)
    x, y = utils.convert_to_map_coordinates(occ_grid=occ_grid, map_resolution=resolution)
    # print(time.time() - tic)
    obstacles_indices = np.where(occ_grid == 0)
    obs_x, obs_y = x[obstacles_indices], y[obstacles_indices]
    obstacle_array = np.array([obs_x, obs_y])
    rotated_obstacle = utils.rotate_coordinates(obstacle_array, ori[2])
    rotated_obstacle[0, :] += pos[0]
    rotated_obstacle[1, :] += pos[1]

    y_obs = rotated_obstacle[1, :]
    x_obs = rotated_obstacle[0, :]
    try:
        x_obs_array = obstacles_x * x_obs[0]
        # x_obs_array = x_obs_array.ravel()
        x_obs_array[:len(x_obs)] = x_obs
        # x_obs_array = np.reshape(x_obs_array, obstacles_x.shape)
        y_obs_array = obstacles_y * y_obs[0]
        # y_obs_array = y_obs_array.ravel()
        y_obs_array[:len(y_obs)] = y_obs
        y_obs_array = np.reshape(y_obs_array, obstacles_y.shape)
        # print(x_obs_array, y_obs_array)

    except IndexError as e:
        print(e, "No obstacles")
        x_obs_array = obstacles_x * 100
        y_obs_array = obstacles_y * 100

    return x_obs_array, y_obs_array


def main():
    rclpy.init()
    robot_controller = RobotController()
    odom_node = OdomSubscriber()
    cmd_vel_publisher = CmdVelPublisher()
    goal_listener = GoalSubscriber()
    laser_node = LaserSubscriber()
    goal_point_publisher = GoalPointPublisher()
    project_path = get_package_share_directory('ros2_mpc')
    # get the goal position from the yaml file
    with open(os.path.join(project_path, 'config/params.yaml'), 'r') as file:
        params = yaml.safe_load(file)
    dt = params['dt']
    size = params['costmap_size']
    goal_threshold = params['goal_threshold']
    resolution = params['resolution']
    obstacles_y = np.ones(int((size * 2) / resolution) * 2)
    obstacles_x = np.ones(int((size * 2) / resolution) * 2)
    robot_controller.get_logger().info("Obstacles size: {}".format(obstacles_x.shape))
    mpc = Mpc()
    robot_controller.get_logger().info("Waiting for path!")
    odom_node.get_logger().info("Waiting for odom!")
    tic = time.time()
    path_xy = robot_controller.get_path()
    robot_controller.get_logger().info("Time taken to get path: {}".format(time.time() - tic))
    REFRESH_TIME = 1.0
    THINKING_FLAG = True
    goal_listener.get_logger().info("Waiting for goal!")
    GOAL_FLAG = False
    u_last = np.array([0, 0])
    while True:
        try:
            goal = goal_listener.get_goal()
        except TypeError:
            continue
        scan_data, angles = laser_node.get_scan()
        pos, ori = odom_node.get_states()
        if scan_data is None:
            continue
        if pos is None:
            continue
        x_obs_array, y_obs_array = get_obstacles(scan_data, angles, size, resolution, pos, ori, obstacles_x,
                                                 obstacles_y)
        # if time.time() - tic > REFRESH_TIME:
        #     tic = time.time()
        path_xy, path_headings = robot_controller.get_path()
        # robot_controller.get_logger().info("Time taken to get path: {}".format(time.time() - tic))
        if path_xy is None:
            # time.sleep(0.1)
            continue
        # Define initial state
        x0 = np.array([pos[0], pos[1], ori[2] % (2 * np.pi)])
        # Define initial control
        u0 = np.zeros((mpc.n_controls, mpc.N))
        if goal is None or pos is None:
            continue
        # goal = get_goal_for_mpc(path_xy, goal, pos)
        # robot_controller.get_logger().info("Goal: {}".format(goal))
        # goal = np.array([goal[0], goal[1], goal[4]])
        goal_mpc = get_goal_for_mpc(path_xy, path_headings, goal, pos, params['look_ahead_distance'])
        goal_point_publisher.publish_goal_point(goal_mpc)
        # Calculate error in orientation
        # error = goal_mpc[2] - x0[2]
        # robot_controller.get_logger().info("Error: {}".format(error))
        # if abs(error) > np.deg2rad(30):
        #     robot_controller.get_logger().info("Rotating towards goal!")
        #     # Identify the direction of rotation
        #     if error > 0:
        #         cmd_vel_publisher.publish_cmd(0.0, 0.1)
        #     else:
        #         cmd_vel_publisher.publish_cmd(0.0, -0.1)
        #     continue
        # Get the reference trajectory
        u = mpc.perform_mpc(u0, initial_state=x0, final_state=goal_mpc, obstacles_x=x_obs_array,
                            obstacles_y=y_obs_array)
        if GOAL_FLAG:
            cmd_vel_publisher.publish_cmd(0.0, 0.0)
        else:
            if np.linalg.norm(u - u_last) > 0.03:
                robot_controller.get_logger().info("Controlling Acceleration!")
                cmd_vel_publisher.publish_cmd(u_last[0] + 0.03, u_last[1] + 0.03)
                u_last = u
            else:
                cmd_vel_publisher.publish_cmd(u[0], u[1])
                u_last = u
        # cmd_vel_publisher.publish_cmd(0.0, 0.0)
        if x0 is not None and goal is not None:
            if np.linalg.norm(x0[0:2] - goal[0:2]) > goal_threshold:
                if GOAL_FLAG:
                    robot_controller.get_logger().info("New goal received!" + str(goal))
                    # robot_controller.get_logger().info("Thinking!")
                    # time.sleep(2.0)
                GOAL_FLAG = False
                robot_controller.get_logger().info("Passing new path to the controller!")
            else:
                if not GOAL_FLAG:
                    cmd_vel_publisher.publish_cmd(0.0, 0.0)
                    robot_controller.get_logger().info("Goal reached!")
                    # Calculate error in orientation
                    # error = goal_mpc[2] - x0[2]
                    # if abs(error) > np.deg2rad(30):
                    #     robot_controller.get_logger().info("Rotating towards goal!")
                    #     # Identify the direction of rotation
                    #     if error > 0:
                    #         cmd_vel_publisher.publish_cmd(0.0, 0.1)
                    #     else:
                    #         cmd_vel_publisher.publish_cmd(0.0, -0.1)
                    #     continue
                    robot_controller.get_logger().info("Waiting for goal!")
                    cmd_vel_publisher.publish_cmd(0.0, 0.0)
                    GOAL_FLAG = True


if __name__ == '__main__':
    main()

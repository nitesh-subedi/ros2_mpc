import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
import numpy as np
from ros2_mpc.planner.local_planner_tracking import Mpc
from ros2_mpc.ros_topics import OdomSubscriber, CmdVelPublisher, GoalSubscriber, LaserSubscriber
from ros2_mpc import utils
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


def get_reference_trajectory(x0, goal, path_xy, path_heading, path_velocity, path_omega, mpc):
    # Get the nearest point on the path to the robot
    nearest_point = 0  # np.argmin(np.linalg.norm(x0[0:2] - path_xy, axis=1))
    if np.linalg.norm(x0[0:2] - path_xy[-1, :]) < 0.5:
        pxf = np.tile(goal[:3], mpc.N).reshape(-1, 1)
    else:
        # Get the reference trajectory
        pxf = path_xy[nearest_point:nearest_point + mpc.N, :]
        # Add the path_heading to pxf
        pxf = np.column_stack((pxf, path_heading[nearest_point:nearest_point + mpc.N]))
        if nearest_point + mpc.N > len(path_xy):
            # Fill the path_xy with repeated last element
            deficit = mpc.N - len(path_xy[nearest_point:])
            path_xy = np.append(path_xy, np.transpose(np.repeat(path_xy[-1, :], deficit).reshape(2, -1)), axis=0)
            # Fill the path_heading with repeated last element
            deficit = mpc.N - len(path_heading[nearest_point:])
            path_heading = np.append(path_heading, np.repeat(path_heading[-1], deficit))
            pxf = path_xy[nearest_point:nearest_point + mpc.N, :]
            # Add the path_heading to pxf
            pxf = np.column_stack((pxf, path_heading[nearest_point:nearest_point + mpc.N]))
            # pxf = np.row_stack((x0, pxf))

        # Flatten the array
        pxf = pxf.flatten().reshape(-1, 1)

    # Get the reference control

    if len(path_velocity) != len(path_omega):
        deficit = len(path_velocity) - len(path_omega)
        path_omega = np.append(path_omega, np.repeat(path_omega[-1], deficit))

    puf = np.column_stack(
        (path_velocity[nearest_point:nearest_point + mpc.N], path_omega[nearest_point:nearest_point + mpc.N]))
    if nearest_point + mpc.N > len(path_velocity):
        # Fill the path_velocity with repeated last element
        deficit_velocity = mpc.N - len(path_velocity[nearest_point:])
        path_velocity = np.append(path_velocity, np.repeat(path_velocity[-1], deficit_velocity))
        # Fill the path_omega with repeated last element
        deficit_omega = mpc.N - len(path_omega[nearest_point:])
        path_omega = np.append(path_omega, np.repeat(path_omega[-1], deficit_omega))
        puf = np.column_stack((path_velocity[nearest_point:nearest_point + mpc.N],
                               path_omega[nearest_point:nearest_point + mpc.N]))

    puf = puf.flatten().reshape(-1, 1) * 0.0 + 0.05
    return pxf, puf


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.path_xy = None
        self.path_heading = None
        self.create_subscription(Path, 'my_path', self.path_callback, 10)

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
        rclpy.spin_once(self)
        return self.path_xy, self.path_heading


def get_obstacles(scan_data, angles, size, resolution, pos, ori, obstacles_x, obstacles_y):
    occ_grid = 1 - utils.convert_laser_scan_to_occupancy_grid(scan_data, angles, resolution, size * 2)
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
    project_path = get_package_share_directory('ros2_mpc')
    # get the goal position from the yaml file
    with open(os.path.join(project_path, 'config/params.yaml'), 'r') as file:
        params = yaml.safe_load(file)
    dt = params['dt']
    size = params['costmap_size']
    resolution = params['resolution']
    obstacles_y = np.ones(int((size * 2) / resolution) * 2)
    obstacles_x = np.ones(int((size * 2) / resolution) * 2)
    robot_controller.get_logger().info("Obstacles size: {}".format(obstacles_x.shape))
    mpc = Mpc()
    robot_controller.get_logger().info("Waiting for path!")
    odom_node.get_logger().info("Waiting for odom!")
    tic = time.time()
    path_xy, path_heading = robot_controller.get_path()
    robot_controller.get_logger().info("Time taken to get path: {}".format(time.time() - tic))
    REFRESH_TIME = 2.0
    goal_listener.get_logger().info("Waiting for goal!")
    GOAL_FLAG = False
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
        # x_obs_array = x_obs_array * -1
        # y_obs_array = y_obs_array * -1
        # robot_controller.get_logger().info("Obstacles_x: {}".format(x_obs_array))
        if time.time() - tic > REFRESH_TIME:
            tic = time.time()
            path_xy, path_heading = robot_controller.get_path()
            # robot_controller.get_logger().info("Time taken to get path: {}".format(time.time() - tic))
        _, path_velocity, path_omega = get_headings(path_xy, dt)
        # Define initial state
        x0 = np.array([pos[0], pos[1], ori[2]])
        # Define initial control
        u0 = np.zeros((mpc.n_controls, mpc.N))
        # Get the reference trajectory
        pxf, puf = get_reference_trajectory(x0, goal, path_xy, path_heading, path_velocity, path_omega, mpc)
        x, u = mpc.perform_mpc(u0, x0, pxf, puf, obstacles_x=x_obs_array,
                               obstacles_y=y_obs_array)
        # robot_controller.get_logger().info("Obstacle cost: {}".format(obstacle_cost))
        # robot_controller.get_logger().info("Position cost: {}".format(position_cost))
        # Publish the control
        cmd_vel_publisher.publish_cmd(u[0], u[1])
        if GOAL_FLAG:
            cmd_vel_publisher.publish_cmd(0.0, 0.0)
        # cmd_vel_publisher.publish_cmd(0.0, 0.0)
        if x0 is not None and goal is not None:
            if np.linalg.norm(x0[0:2] - goal[0:2]) > 0.15:
                if GOAL_FLAG:
                    robot_controller.get_logger().info("New goal received!" + str(goal))
                GOAL_FLAG = False
                robot_controller.get_logger().info("Passing new path to the controller!")
            else:
                if not GOAL_FLAG:
                    cmd_vel_publisher.publish_cmd(0.0, 0.0)
                    robot_controller.get_logger().info("Goal reached!")
                    cmd_vel_publisher.publish_cmd(0.0, 0.0)
                    GOAL_FLAG = True


if __name__ == '__main__':
    main()

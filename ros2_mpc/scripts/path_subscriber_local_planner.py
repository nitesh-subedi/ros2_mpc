import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
import numpy as np
from ros2_mpc.planner.local_planner_tracking import Mpc
from ros2_mpc.ros_topics import OdomSubscriber, CmdVelPublisher
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


def get_reference_trajectory(x0, goal_xy, path_xy, path_heading, path_velocity, path_omega, mpc, robot_controller):
    # Get the nearest point on the path to the robot
    nearest_point = np.argmin(np.linalg.norm(x0[0:2] - path_xy, axis=1))
    if np.linalg.norm(x0[0:2] - path_xy[-1, :]) < 0.5:
        # Put all points of path to be the goal
        goal_new = np.append(np.array(goal_xy), 0)
        pxf = np.tile(goal_new, mpc.N).reshape(-1, 1)
        robot_controller.info("Inside Circle!")
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

    puf = puf.flatten().reshape(-1, 1)
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


def main():
    rclpy.init()
    robot_controller = RobotController()
    odom_node = OdomSubscriber()
    cmd_vel_publisher = CmdVelPublisher()
    project_path = get_package_share_directory('ros2_mpc')
    # get the goal position from the yaml file
    with open(os.path.join(project_path, 'config/params.yaml'), 'r') as file:
        params = yaml.safe_load(file)
    dt = params['dt']
    N = params['N']
    goal_xy = np.array(params['goal_pose'])
    mpc = Mpc(dt, N)
    robot_controller.get_logger().info("Waiting for path!")
    odom_node.get_logger().info("Waiting for odom!")
    tic = time.time()
    path_xy, path_heading = robot_controller.get_path()
    robot_controller.get_logger().info("Time taken to get path: {}".format(time.time() - tic))
    REFRESH_TIME = 2.0
    tic = time.time()
    while True:
        pos, ori, velocity = odom_node.get_states()
        if time.time() - tic > REFRESH_TIME:
            tic = time.time()
            path_xy, path_heading = robot_controller.get_path()
            robot_controller.get_logger().info("Time taken to get path: {}".format(time.time() - tic))
        _, path_velocity, path_omega = get_headings(path_xy, dt)
        # Define initial state
        x0 = np.array([pos[0], pos[1], ori[2]])
        # Define initial control
        u0 = np.zeros((mpc.n_controls, mpc.N))
        # Get the reference trajectory
        pxf, puf = get_reference_trajectory(x0, goal_xy, path_xy, path_heading, path_velocity, path_omega, mpc,
                                            robot_controller.get_logger())
        # noinspection PyUnboundLocalVariable
        x, u = mpc.perform_mpc(u0, x0, pxf, puf)
        # Publish the control
        cmd_vel_publisher.publish_cmd(u[0], u[1])
        robot_controller.get_logger().info("Passing new path to the controller!")
        if np.linalg.norm(x0[0:2] - goal_xy) < 0.15:
            break
    cmd_vel_publisher.publish_cmd(0.0, 0.0)
    odom_node.destroy_node()
    robot_controller.destroy_node()
    cmd_vel_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
import numpy as np
from ros2_mpc.planner.local_planner_tracking import Mpc
from ros2_mpc.ros_topics import OdomSubscriber, CmdVelPublisher
from ros2_mpc import utils
import time


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


def get_reference_trajectory(x0, goal_xy, path_xy, path_heading, path_velocity, path_omega, mpc):
    # Get the nearest point on the path to the robot
    nearest_point = np.argmin(np.linalg.norm(x0[0:2] - path_xy, axis=1))
    if np.linalg.norm(x0[0:2] - path_xy[-1, :]) < 0.5:
        # Put all points of path to be the goal
        goal_new = np.append(np.array(goal_xy), 0)
        pxf = np.tile(goal_new, mpc.N).reshape(-1, 1)
        print("Inside the circle")
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


class PathSubscriber(Node):
    def __init__(self):
        super().__init__('path_subscriber')
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
        self.get_logger().info("Path Received!")
        self.path_xy = path
        self.path_heading = headings

    def get_path(self):
        rclpy.spin_once(self)
        return self.path_xy, self.path_heading


def main():
    rclpy.init()
    path_subscriber = PathSubscriber()
    odom_node = OdomSubscriber()
    cmd_vel_publisher = CmdVelPublisher()
    dt = 0.2
    N = 20
    goal_xy = np.array([5.0, -0.6])
    mpc = Mpc(dt, N)
    while True:
        pos, ori, velocity = odom_node.get_states()
        path_xy, path_heading = path_subscriber.get_path()
        if path_xy is None:
            continue
        _, path_velocity, path_omega = get_headings(path_xy, dt)
        # Define initial state
        x0 = np.array([pos[0], pos[1], ori[2]])
        # Define initial control
        u0 = np.zeros((mpc.n_controls, mpc.N))
        # Get the reference trajectory
        pxf, puf = get_reference_trajectory(x0, goal_xy, path_xy, path_heading, path_velocity, path_omega, mpc)
        tictic = time.time()
        # noinspection PyUnboundLocalVariable
        x, u = mpc.perform_mpc(u0, x0, pxf, puf)
        print(time.time() - tictic)
        # Publish the control
        cmd_vel_publisher.publish_cmd(u[0], u[1])
        if np.linalg.norm(x0[0:2] - goal_xy) < 0.15:
            break
    cmd_vel_publisher.publish_cmd(0.0, 0.0)


if __name__ == '__main__':
    main()

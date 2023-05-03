import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ros2_mpc.utils import euler_from_quaternion
import time
import casadi
import numpy as np
import matplotlib.pyplot as plt
from ros2_mpc.mpc_trajectory import Mpc
from global_planner import get_path
import cv2


class OdomSubscriber(Node):
    def __init__(self):
        super().__init__('odom_subscriber')
        self.velocities = None
        self.orientation = None
        self.position = None
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, msg):
        self.position = np.array([msg.pose.pose.position.x + 3.0, msg.pose.pose.position.y - 1.0]).round(decimals=2)
        self.orientation = np.array(euler_from_quaternion(
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)).round(decimals=2)
        self.velocities = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z]).round(decimals=2)
        self.get_logger().info("Data received!")

    def get_states(self):
        rclpy.spin_once(self)
        time.sleep(0.1)
        return self.position, self.orientation, self.velocities


def main():
    rclpy.init()
    map_image = cv2.imread("/home/nitesh/workspaces/ros2_mpc_ws/src/ros2_mpc/maps/map_carto.pgm", cv2.IMREAD_GRAYSCALE)
    map_image[map_image == 0] = 1
    map_image[map_image > 1] = 0
    # Dilate the image by 10 pixels
    kernel = np.ones((10, 10), np.uint8)
    map_image = cv2.dilate(map_image, kernel, iterations=1)
    map_image = map_image.astype(np.uint8)
    resolution = 0.05
    origin = np.array([-4.84, -6.61])
    odom_node = OdomSubscriber()
    # Get the current position of the robot
    pos, ori, velocity = odom_node.get_states()
    robot_on_map = (np.array([pos[0], pos[1]]) - origin) / resolution
    robot_on_map = tuple(robot_on_map.astype(int))
    # Get the goal position
    start_position = (robot_on_map[0], robot_on_map[1])
    goal_position = (50, 175)
    # Get the path
    path = get_path(start_position, goal_position, map_image)
    # Convert the path to world coordinates
    path = np.array(path) * resolution + np.array([origin[1], origin[0]])
    print(path)
    # dt = 0.1
    # N = 20
    # mpc = Mpc(dt, N)
    # # Define initial state
    # x0 = np.array([0, 0, 0])
    # # Define final state
    # xf = np.array([1, 1, 0])
    # # Define initial control
    # u0 = np.zeros((mpc.n_controls, mpc.N))
    # count = 0
    # x_pos = []
    # while count <= 300:
    #     current_time = count * dt
    #     pxf = np.array([])
    #     puf = np.array([])
    #     for k in range(mpc.N):
    #         t_predict = current_time + k * dt
    #         x_ref = 0.5 * t_predict * casadi.cos(np.deg2rad(45))
    #         y_ref = 0.5 * t_predict * casadi.sin(np.deg2rad(45))
    #         theta_ref = np.deg2rad(45)
    #         u_ref = 0.25
    #         omega_ref = 0
    #         if np.linalg.norm(x0[0:2] - xf[0:2]) < 0.1:
    #             x_ref = xf[0]
    #             y_ref = xf[1]
    #             u_ref = 0
    #             omega_ref = 0
    #         if k == 0:
    #             pxf = casadi.vertcat(x_ref, y_ref, theta_ref)
    #             puf = casadi.vertcat(u_ref, omega_ref)
    #         else:
    #             pxf = casadi.vertcat(pxf, casadi.vertcat(x_ref, y_ref, theta_ref))
    #             puf = casadi.vertcat(puf, casadi.vertcat(u_ref, omega_ref))
    #
    #     x, u = mpc.perform_mpc(u0, x0, pxf, puf)
    #     x0 = x
    #     x_pos.append(x0)
    #     count += 1
    #     print(x0, xf)
    #     print('u = ', u)
    #     pass
    #
    # x_pos = np.array(x_pos)
    # plt.plot(x_pos[:, 0], x_pos[:, 1])
    # # Plot theta vs time
    # # plt.plot(x_pos[:, 2])
    # plt.show()


if __name__ == '__main__':
    main()

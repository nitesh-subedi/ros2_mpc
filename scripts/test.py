import casadi
import numpy as np
import matplotlib.pyplot as plt
from ros2_mpc.mpc import Mpc
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import rclpy
import time
from ros2_mpc.utils import *


class LaserSubscriber(Node):
    def __init__(self):
        super().__init__('laser_subscriber')
        self.angles = None
        self.laser_data = None
        self.map_subscriber = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

    def laser_callback(self, msg):
        self.laser_data = np.array(msg.ranges)
        self.angles = np.array([msg.angle_min, msg.angle_max])
        self.get_logger().info("Data received!")

    def get_scan(self):
        rclpy.spin_once(self)
        time.sleep(0.1)
        return self.laser_data, self.angles


class OdomSubscriber(Node):
    def __init__(self):
        super().__init__('odom_subscriber')
        self.velocities = None
        self.orientation = None
        self.position = None
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, msg):
        self.position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y]).round(decimals=2)
        self.orientation = np.array(euler_from_quaternion(
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)).round(decimals=2)
        self.velocities = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z]).round(decimals=2)
        self.get_logger().info("Data received!")

    def get_states(self):
        rclpy.spin_once(self)
        time.sleep(0.1)
        return self.position, self.orientation, self.velocities


class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pub = Twist()

    def publish_cmd(self, v, w):
        self.pub.linear.x = v
        self.pub.angular.z = w
        self.cmd_publisher.publish(self.pub)
        self.get_logger().info("cmd published!")


def main():
    rclpy.init()
    laser_node = LaserSubscriber()
    odom_node = OdomSubscriber()
    cmd_publisher = CmdVelPublisher()
    # Define time step
    dt = 0.2
    # Define prediction horizon
    N = 20
    # Define costmap size and resolution
    costmap_size = 2
    resolution = 0.5
    # Define initial state
    x0 = np.array([0, 0, 0])
    # Define final state
    xf = np.array([5, 5, 0])
    # Create an instance of the MPC class
    mpc_planner = Mpc(dt, N, cost_factor=1.0, costmap_size=costmap_size, resolution=resolution)
    x_pos = []
    u0 = np.zeros((mpc_planner.n_controls, mpc_planner.N))
    obstacles_x = np.ones(int((costmap_size * 2) / resolution) * 2) * 100
    obstacles_y = np.ones(int((costmap_size * 2) / resolution) * 2) * 100
    count = 0
    while np.linalg.norm(x0 - xf) > 0.2 and count < 1000:
        scan_data, angles = laser_node.get_scan()
        pos, ori, velocity = odom_node.get_states()
        if scan_data is None:
            continue

        occ_grid = 1 - convert_laser_scan_to_occupancy_grid(scan_data, angles, resolution, costmap_size * 2)
        occ_grid = np.rot90(occ_grid, k=2)
        x, y = convert_to_map_coordinates(occ_grid=occ_grid, map_resolution=resolution)
        obstacles_indices = np.where(occ_grid == 0)
        obs_x, obs_y = x[obstacles_indices], y[obstacles_indices]
        obstacle_array = np.array([obs_x, obs_y])
        rotated_obstacle = rotate_coordinates(obstacle_array, ori[2])
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
            # y_obs_array = np.reshape(y_obs_array, obstacles_y.shape)
        except IndexError as e:
            print(e, "No obstacles")
            x_obs_array = obstacles_x
            y_obs_array = obstacles_y
        x0 = casadi.vertcat(pos[0], pos[1], ori[2])
        x, u = mpc_planner.perform_mpc(u0=u0, initial_state=x0, final_state=xf, obstacles_x=x_obs_array,
                                       obstacles_y=y_obs_array)
        # x0 = x[:, 1]
        u0 = np.concatenate((u[:, 1:], u[:, -1].reshape(2, 1)), axis=1)
        count += 1
        x_pos.append(x0)
        print(x0)
        cmd_publisher.publish_cmd(u[0, 0], u[1, 0])
    cmd_publisher.publish_cmd(0.0, 0.0)

    x_pos = np.array(x_pos)
    plt.plot(x_pos[:, 0], x_pos[:, 1])
    plt.show()


if __name__ == '__main__':
    main()

import casadi
import numpy as np
import rclpy
from matplotlib import pyplot as plt
from rclpy.node import Node
from numba import njit
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import time
import math


def convert_laser_scan_to_occupancy_grid(laser_scan_data, angles, map_resolution, map_size):
    # Calculate the size of each cell in the occupancy grid
    cell_size = map_resolution
    angle_min = angles[0]
    angle_max = angles[1]

    # Calculate the number of cells in the occupancy grid
    num_cells = int(map_size / cell_size)

    # Create an empty occupancy grid
    occupancy_grid = np.zeros((num_cells, num_cells))

    # Convert laser scan data to Cartesian coordinates
    angles = np.linspace(angle_min, angle_max, len(laser_scan_data))
    x_coords = laser_scan_data * np.cos(angles)
    y_coords = laser_scan_data * np.sin(angles)

    # Convert Cartesian coordinates to occupancy grid indices
    x_indices = np.array((x_coords + (map_size / 2)) / cell_size, dtype=int)
    y_indices = np.array((y_coords + (map_size / 2)) / cell_size, dtype=int)

    # Set occupied cells in the occupancy grid
    for x, y in zip(x_indices, y_indices):
        if 0 <= x < num_cells and 0 <= y < num_cells:
            occupancy_grid[x, y] = 1

    return occupancy_grid


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


@njit
def convert_to_map_coordinates(occ_grid, map_resolution=0.8):
    map_origin = np.array([occ_grid.shape[0] // 2, occ_grid.shape[1] // 2]) * map_resolution
    meter_x = occ_grid.copy()
    meter_y = occ_grid.copy()
    for i in range(occ_grid.shape[0]):
        for j in range(occ_grid.shape[1]):
            meter_x[i, j] = - j * map_resolution + map_origin[1]
            meter_y[i, j] = - i * map_resolution + map_origin[0]

    return meter_y, meter_x


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


def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


class OptiPlanner:
    def __init__(self, costmap_size, resolution):
        self.opti = casadi.Opti()
        self.N = 20
        self.dt = 0.1
        self.costmap_size = costmap_size
        self.resolution = resolution

        # Define state variables
        self.x = self.opti.variable()
        self.y = self.opti.variable()
        self.th = self.opti.variable()
        self.states = casadi.vertcat(self.x, self.y, self.th)
        self.n_states = self.states.shape[0]

        # Define control variables
        self.v = self.opti.variable()
        self.w = self.opti.variable()
        self.controls = casadi.vertcat(self.v, self.w)
        self.n_controls = self.controls.shape[0]

        self.dx = self.v * casadi.cos(self.th)
        self.dy = self.v * casadi.sin(self.th)
        self.dth = self.w

        self.rhs = casadi.vertcat(self.dx, self.dy, self.dth)

        # Define the mapping function
        self.f = casadi.Function('f', [self.states, self.controls], [self.rhs])  # Non-linear mapping function

        # Define state, control and parameter matrices
        self.X = self.opti.variable(self.n_states, self.N + 1)
        self.U = self.opti.variable(self.n_controls, self.N)
        self.P = self.opti.parameter(2 * self.n_states)
        # States Integration
        self.X[:, 0] = self.P[0:self.n_states]  # Initial State
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            f_value = self.f(st, con)
            st_next = st + self.dt * f_value
            self.X[:, k + 1] = st_next
            # self.opti.subject_to(X[:, k + 1] == st_next)

        obj = 0

        # Defining weighing matrices
        Q = np.eye(self.n_states, dtype=float)
        Q = Q * 0
        Q[0, 0] = 0.5
        Q[1, 1] = 2

        R = np.eye(self.n_controls, dtype=float)
        R = R * 0.5

        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            obj = obj + casadi.mtimes(casadi.mtimes((st - self.P[self.n_states:2 * self.n_states]).T, Q),
                                      (st - self.P[self.n_states:2 * self.n_states])) + casadi.mtimes(
                casadi.mtimes(con.T, R), con)

        self.opti.minimize(obj)

        max_vel = 0.3
        self.opti.subject_to(self.U[0, :] <= max_vel)
        self.opti.subject_to(self.U[0, :] >= -max_vel)

        max_ang = 0.1
        self.opti.subject_to(self.U[1, :] <= max_ang)
        self.opti.subject_to(self.U[1, :] >= -max_ang)

        # Define obstacles
        # self.obstacles_x = self.opti.parameter(int((costmap_size * 2) / resolution) * 2)
        # self.obstacles_y = self.opti.parameter(int((costmap_size * 2) / resolution) * 2)
        #
        # for i in range(self.obstacles_x.shape[0]):
        #     for k in range(self.N + 1):
        #         distance = np.sqrt(
        #             (self.X[0, k] - self.obstacles_x[i]) ** 2 + (self.X[1, k] - self.obstacles_y[i]) ** 2)
        #         self.opti.subject_to(distance > 0.3)
        for k in range(self.N + 1):
            # distance = np.sqrt((self.X[0, k] - 1.6) ** 2 + (self.X[1, k] + 0.8) ** 2)
            distance_2 = casadi.sqrt((self.X[0, k] - 2.5) ** 2 + (self.X[1, k] - 2.5) ** 2)
            # self.opti.subject_to(distance >= 0.2)
            self.opti.subject_to(distance_2 >= 0.5)
        # opts = {}
        # opts['ipopt.print_level'] = 0
        self.opti.solver('ipopt')

    def run(self, final_state, robot_pos, robot_ori):
        u0 = np.zeros((self.n_controls, self.N))
        pos = casadi.vertcat(robot_pos, robot_ori)
        # print(pos)
        # print(final_state)
        self.opti.set_value(self.P, casadi.vertcat(pos, final_state))
        self.opti.set_initial(self.U, u0)
        # self.opti.set_value(self.obstacles_x, obstacles_x)
        # self.opti.set_value(self.obstacles_y, obstacles_y)
        self.opti.solve()
        return self.opti.value(self.U[:, 0])


@njit
def rotate_coordinates(coordinates, rotation):
    rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                           [np.sin(rotation), np.cos(rotation)]])

    rotated = np.dot(rot_matrix, coordinates)

    return rotated


def main(args=None):
    rclpy.init(args=args)
    laser_node = LaserSubscriber()
    resolution = 0.5
    size = 2.5
    planner = OptiPlanner(size, resolution)
    odom_node = OdomSubscriber()
    cmd_publisher = CmdVelPublisher()
    final_pose = casadi.vertcat(0, 0, 0)
    # u0 = np.zeros((planner.n_controls, planner.N))
    obstacles_y = np.ones(int((size * 2) / resolution) * 2)
    obstacles_x = np.ones(int((size * 2) / resolution) * 2)
    max_linear_accel = 0.05
    max_angular_accel = 0.005
    last_cmd = np.array([0, 0])
    while rclpy.ok():
        scan_data, angles = laser_node.get_scan()
        pos, ori, velocity = odom_node.get_states()
        if scan_data is None:
            continue
        occ_grid = 1 - convert_laser_scan_to_occupancy_grid(scan_data, angles, resolution, size * 2)
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
        # plt.scatter(x_obs, y_obs)
        # plt.show()
        # break

        # x_obs_array = obstacles_x * x_obs[0]
        # # x_obs_array = x_obs_array.ravel()
        # x_obs_array[:len(x_obs)] = x_obs
        # # x_obs_array = np.reshape(x_obs_array, obstacles_x.shape)
        # y_obs_array = obstacles_y * y_obs[0]
        # # y_obs_array = y_obs_array.ravel()
        # y_obs_array[:len(y_obs)] = y_obs
        # y_obs_array = np.reshape(y_obs_array, obstacles_y.shape)
        # print(x_obs_array, y_obs_array)
        #
        cmd = planner.run(final_pose, pos, ori[2])
        # if np.abs(np.abs(cmd[0]) - np.abs(last_cmd[0])) > max_linear_accel:
        #     cmd[0] = np.minimum(np.abs(last_cmd[0] + max_linear_accel * (cmd[0] / np.abs(cmd[0]))), 0.1) * (
        #             cmd[1] / np.abs(cmd[1]))
        # if np.abs(np.abs(cmd[1]) - np.abs(last_cmd[1])) > max_angular_accel:
        #     cmd[1] = np.minimum(np.abs(last_cmd[1] + max_angular_accel * (cmd[1] / np.abs(cmd[1]))), 0.05) * (
        #             cmd[1] / np.abs(cmd[1]))

        print(cmd[0], cmd[1])
        print(pos, ori[2])
        cmd_publisher.publish_cmd(cmd[0], cmd[1])
        last_cmd = cmd
        # time.sleep(1)
        # cmd_publisher.publish_cmd(0.0, 0.0)


if __name__ == '__main__':
    main()

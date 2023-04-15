import casadi
import numpy as np
import rclpy
from rclpy.node import Node
from numba import njit
from sensor_msgs.msg import LaserScan
import time


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


class OptiPlanner:
    def __init__(self, costmap_size, resolution):
        self.opti = casadi.Opti()
        self.N = 20
        self.dt = 0.2
        self.costmap_size = 2
        self.resolution = 0.5

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
        Q = Q * 0.5

        R = np.eye(self.n_controls, dtype=float)
        R = R * 0.5

        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            obj = obj + casadi.mtimes(casadi.mtimes((st - self.P[self.n_states:2 * self.n_states]).T, Q),
                                      (st - self.P[self.n_states:2 * self.n_states])) + casadi.mtimes(
                casadi.mtimes(con.T, R), con)

        self.opti.minimize(obj)

        max_vel = 0.5
        self.opti.subject_to(self.U[0, :] <= max_vel)
        self.opti.subject_to(self.U[0, :] >= -max_vel)

        max_ang = 0.2
        self.opti.subject_to(self.U[1, :] <= max_ang)
        self.opti.subject_to(self.U[1, :] >= -max_ang)

        # Define obstacles
        obstacles_x = self.opti.parameter(int((costmap_size * 2) / resolution), int((costmap_size * 2) / resolution))
        obstacles_y = self.opti.parameter(int((costmap_size * 2) / resolution), int((costmap_size * 2) / resolution))

        for i in range(obstacles_x.shape[0]):
            for j in range(obstacles_x.shape[1]):
                for k in range(self.N + 1):
                    distance = np.sqrt((self.X[0, k] - obstacles_x[i, j]) ** 2 + (self.X[1, k] - obstacles_y[i, j]))
                    self.opti.subject_to(distance > 0.3)

        self.opti.solver('ipopt')
        self.initial_state = casadi.vertcat(0, 0, 0)
        self.final_state = casadi.vertcat(2, 0, 0)


def main(args=None):
    rclpy.init(args=args)
    laser_node = LaserSubscriber()
    resolution = 0.8
    size = 2
    planner = OptiPlanner(size, resolution)
    while rclpy.ok():
        scan_data, angles = laser_node.get_scan()
        if scan_data is None:
            continue
        occ_grid = 1 - convert_laser_scan_to_occupancy_grid(scan_data, angles, resolution, size * 2)
        occ_grid = cv2.rotate(occ_grid, cv2.ROTATE_180)
        tic = time.time()
        x, y = convert_to_map_coordinates(occ_grid=occ_grid, map_resolution=resolution)
        print(time.time() - tic)
        obstacles_indices = np.where(occ_grid == 0)
        obs_x, obs_y = x[obstacles_indices], y[obstacles_indices]

    # y_obs = np.array([-0.8, -1.6, 1.6, 0.8, -0.8, -0.8])
    # x_obs = np.array([1.6, 1.6, 0.8, 0.8, 0.8, 0.0])
    #
    # x_obs_array = np.ones(obstacles_x.shape) * x_obs[0]
    # x_obs_array = x_obs_array.ravel()
    # x_obs_array[:len(x_obs)] = x_obs
    # x_obs_array = np.reshape(x_obs_array, obstacles_x.shape)
    # y_obs_array = np.ones(obstacles_y.shape) * y_obs[0]
    # y_obs_array = y_obs_array.ravel()
    # y_obs_array[:len(y_obs)] = y_obs
    # y_obs_array = np.reshape(y_obs_array, obstacles_y.shape)


if __name__ == '__main__':
    main()

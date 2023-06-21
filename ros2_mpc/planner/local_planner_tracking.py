import casadi
import numpy as np
from ament_index_python.packages import get_package_share_directory
import os
import yaml


# import matplotlib.pyplot as plt


class Mpc:
    def __init__(self):
        self.position_cost = None
        self.hxy = None
        project_path = get_package_share_directory('ros2_mpc')
        # get the goal position from the yaml file
        with open(os.path.join(project_path, 'config/params.yaml'), 'r') as file:
            params = yaml.safe_load(file)
        self.dt = params['dt']
        self.N = params['N']
        # params['inflation_radius = inflation_radius
        self.opti = casadi.Opti()

        # Get system function 'f'
        self.f, self.n_states, self.n_controls = self.get_system_function()

        # Get decision variables
        self.X, self.U, self.P_X, self.P_U = self.get_decision_variables()

        # Perform integration using RK4
        # self.rk4()

        # Perform integration using Euler
        self.euler_integration()

        self.obstacles_x = self.opti.parameter(int((params['costmap_size'] * 2) / params['resolution']) * 2)
        self.obstacles_y = self.opti.parameter(int((params['costmap_size'] * 2) / params['resolution']) * 2)

        obstacles_cost = self.define_obstacles_cost_function(params)
        # Define cost function
        self.define_cost_function(params, obstacles_cost=0)

        # Define constraints
        self.constraints()

        # # Define obstacle avoidance constraints
        # self.obstacle_avoidance_constraints()

        # Define opts for solver
        opts = {"ipopt.print_level": 0, "print_time": 0}

        # Define solver
        self.opti.solver('ipopt', opts)

    def define_obstacles_cost_function(self, params):
        obj = 0
        for k in range(self.N + 1):
            for i in range(self.obstacles_x.shape[0]):
                self.hxy = casadi.log(((self.X[0, k] - self.obstacles_x[i]) / params['inflation_radius']) ** 2 + (
                        (self.X[1, k] - self.obstacles_y[i]) / params['inflation_radius']) ** 2)
                obj = obj + casadi.exp(casadi.exp(-self.hxy) * 3.0) * params['cost_factor']
        # self.hxy = obj
        return obj

    def perform_mpc(self, u0, x0, pf, puf, obstacles_x=None, obstacles_y=None):
        # Set initial state and parameter value
        self.opti.set_initial(self.U, u0)
        self.opti.set_value(self.P_X, casadi.vertcat(x0, pf))
        self.opti.set_value(self.P_U, puf)
        if obstacles_x is not None and obstacles_y is not None:
            self.opti.set_value(self.obstacles_x, obstacles_x)
            self.opti.set_value(self.obstacles_y, obstacles_y)
        # Solve the optimization problem
        sol = self.opti.solve()
        # Extract optimal control
        u_opt = sol.value(self.U)
        x_opt = sol.value(self.X)
        # obstacle_cost = sol.value(self.hxy)
        # position_cost = sol.value(self.position_cost)
        return x_opt, u_opt[:, 0]  # , obstacle_cost, position_cost

    # def obstacle_avoidance_constraints(self):
    #     # Define obstacle at (x,y) = (5,3)
    #     x_obs = 4
    #     y_obs = 7
    #     # Define radius of obstacle
    #     r_obs = 1
    #     # Define obstacle avoidance constraint
    #     for k in range(self.N):
    #         self.opti.subject_to((self.X[0, k] - x_obs) ** 2 + (self.X[1, k] - y_obs) ** 2 >= r_obs ** 2)

    def constraints(self):
        # Define constraints
        self.opti.subject_to(self.opti.bounded(-0.1, self.U[0, :], 0.2))
        self.opti.subject_to(self.opti.bounded(-0.2, self.U[1, :], 0.2))

    def define_cost_function(self, params, obstacles_cost):
        # Define cost function
        Q = np.eye(self.n_states, dtype=float)
        Q[0, 0] = params['Q'][0]
        Q[1, 1] = params['Q'][1]
        Q[2, 2] = params['Q'][2]
        obj = 0
        R = np.eye(self.n_controls, dtype=float)
        R[0, 0] = params['R'][0]
        R[1, 1] = params['R'][1]
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            obj = obj + casadi.mtimes(
                casadi.mtimes((st - self.P_X[self.n_states * (k + 1):self.n_states * (k + 1) + self.n_states]).T, Q),
                (st - self.P_X[self.n_states * (k + 1):self.n_states * (k + 1) + self.n_states])) + casadi.mtimes(
                casadi.mtimes((con - self.P_U[self.n_controls * k:self.n_controls * k + self.n_controls]).T, R),
                (con - self.P_U[self.n_controls * k:self.n_controls * k + self.n_controls]))
            obj = obj + (1 / casadi.exp(con[0])) ** params['reverse_factor']
        # self.position_cost = obj
        # obj = obj + obstacles_cost
        self.opti.minimize(obj)

    def euler_integration(self):
        self.X[:, 0] = self.P_X[0:self.n_states]
        for k in range(self.N):
            x_next = self.X[:, k] + self.dt * self.f(self.X[:, k], self.U[:, k])
            # self.X[:, k + 1] = x_next
            self.opti.subject_to(self.X[:, k + 1] == x_next)

    def rk4(self):
        self.X[:, 0] = self.P_X[0:self.n_states]  # Initial State
        for k in range(self.N):
            # Define RK4 constants
            k1 = self.f(self.X[:, k], self.U[:, k])
            k2 = self.f(self.X[:, k] + self.dt / 2 * k1, self.U[:, k])
            k3 = self.f(self.X[:, k] + self.dt / 2 * k2, self.U[:, k])
            k4 = self.f(self.X[:, k] + self.dt * k3, self.U[:, k])

            # Perform integration
            x_next = self.X[:, k] + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            # self.X[:, k + 1] = x_next
            self.opti.subject_to(self.X[:, k + 1] == x_next)

    def get_decision_variables(self):
        # Define decision variables
        x = self.opti.variable(self.n_states, self.N + 1)
        u = self.opti.variable(self.n_controls, self.N)
        p_x = self.opti.parameter(self.n_states + self.N * self.n_states)
        p_u = self.opti.parameter(self.N * self.n_controls)
        return x, u, p_x, p_u

    def get_system_function(self):
        # Define number of states
        x = self.opti.variable()
        y = self.opti.variable()
        theta = self.opti.variable()
        states = casadi.vertcat(x, y, theta)
        n_states = states.shape[0]

        # Define controls
        v = self.opti.variable()
        w = self.opti.variable()
        controls = casadi.vertcat(v, w)
        n_controls = controls.shape[0]

        # Define dynamic constraints
        rhs = casadi.vertcat(v * casadi.cos(theta), v * casadi.sin(theta), w)
        f = casadi.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])
        return f, n_states, n_controls

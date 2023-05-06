import casadi
import numpy as np


# import matplotlib.pyplot as plt


class Mpc:
    def __init__(self, dt, N):
        self.dt = dt
        self.N = N
        # self.inflation_radius = inflation_radius
        self.opti = casadi.Opti()

        # Get system function 'f'
        self.f, self.n_states, self.n_controls = self.get_system_function()

        # Get decision variables
        self.X, self.U, self.P_X, self.P_U = self.get_decision_variables()

        # Perform integration using RK4
        self.rk4()

        # Perform integration using Euler
        # self.euler_integration()

        # obstacles_cost = self.define_obstacles_cost_function()
        # Define cost function
        self.define_cost_function()

        # Define constraints
        self.constraints()

        # # Define obstacle avoidance constraints
        # self.obstacle_avoidance_constraints()

        # Define solver
        self.opti.solver('ipopt')

    # def define_obstacles_cost_function(self, cost_factor):
    #     obj = 0
    #     for k in range(self.N + 1):
    #         for i in range(self.obstacles_x.shape[0]):
    #             hxy = casadi.log(((self.X[0, k] - self.obstacles_x[i]) / self.inflation_radius) ** 2 + (
    #                     (self.X[1, k] - self.obstacles_y[i]) / self.inflation_radius) ** 2)
    #             obj = obj + casadi.exp(cost_factor * casadi.exp(-hxy))
    #     return obj

    def perform_mpc(self, u0, x0, pf, puf):
        # Set initial state and parameter value
        self.opti.set_initial(self.U, u0)
        self.opti.set_value(self.P_X, casadi.vertcat(x0, pf))
        self.opti.set_value(self.P_U, puf)
        # Solve the optimization problem
        sol = self.opti.solve()
        # Extract optimal control
        u_opt = sol.value(self.U)
        x_opt = sol.value(self.X)
        return x_opt, u_opt[:, 0]

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
        self.opti.subject_to(self.opti.bounded(-0.1, self.U[0, :], 0.1))
        self.opti.subject_to(self.opti.bounded(-0.1, self.U[1, :], 0.1))

    def define_cost_function(self):
        # Define cost function
        Q = np.eye(self.n_states, dtype=float)
        Q[0, 0] = 0.5
        Q[1, 1] = 0.5
        Q[2, 2] = 0.0005
        obj = 0
        R = np.eye(self.n_controls, dtype=float)
        R = R * 0.1
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            obj = obj + casadi.mtimes(
                casadi.mtimes((st - self.P_X[self.n_states * (k + 1):self.n_states * (k + 1) + self.n_states]).T, Q),
                (st - self.P_X[self.n_states * (k + 1):self.n_states * (k + 1) + self.n_states])) + casadi.mtimes(
                casadi.mtimes((con - self.P_U[self.n_controls * k:self.n_controls * k + self.n_controls]).T, R),
                (con - self.P_U[self.n_controls * k:self.n_controls * k + self.n_controls]))
        self.opti.minimize(obj)

    def euler_integration(self):
        self.X[:, 0] = self.P_X[0:self.n_states]
        for k in range(self.N):
            x_next = self.X[:, k] + self.dt * self.f(self.X[:, k], self.U[:, k])
            self.X[:, k + 1] = x_next
            # self.opti.subject_to(self.X[:, k + 1] == x_next)

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

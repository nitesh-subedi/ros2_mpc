import casadi
import numpy as np

opti = casadi.Opti()

N = 20
dt = 0.2
costmap_size = 2
resolution = 0.8

# Define state variables
x = opti.variable()
y = opti.variable()
th = opti.variable()
states = casadi.vertcat(x, y, th)
n_states = states.shape[0]

# Define control variables
v = opti.variable()
w = opti.variable()
controls = casadi.vertcat(v, w)
n_controls = controls.shape[0]

dx = v * casadi.cos(th)
dy = v * casadi.sin(th)
dth = w

rhs = casadi.vertcat(dx, dy, dth)

# Define the mapping function
f = casadi.Function('f', [states, controls], [rhs])  # Non-linear mapping function

# Define state, control and parameter matrices
X = opti.variable(n_states, N + 1)
U = opti.variable(n_controls, N)
P = opti.parameter(2 * n_states)
# States Integration
X[:, 0] = P[0:n_states]  # Initial State
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    f_value = f(st, con)
    st_next = st + dt * f_value
    X[:, k + 1] = st_next
    # opti.subject_to(X[:, k + 1] == st_next)

obj = 0

# Defining weighing matrices
Q = np.eye(n_states, dtype=float)
Q = Q * 0.5

R = np.eye(n_controls, dtype=float)
R = R * 0.5

for k in range(N):
    st = X[:, k]
    con = U[:, k]
    obj = obj + casadi.mtimes(casadi.mtimes((st - P[n_states:2 * n_states]).T, Q),
                              (st - P[n_states:2 * n_states])) + casadi.mtimes(casadi.mtimes(con.T, R), con)

opti.minimize(obj)

max_vel = 0.5
opti.subject_to(U[0, :] <= max_vel)
opti.subject_to(U[0, :] >= -max_vel)

max_ang = 0.2
opti.subject_to(U[1, :] <= max_ang)
opti.subject_to(U[1, :] >= -max_ang)

# Define obstacles
obstacles = opti.parameter(int((costmap_size * 2) / resolution), int((costmap_size * 2) / resolution))

print(obstacles.shape)

import casadi
import numpy as np
import matplotlib.pyplot as plt
from ros2_mpc import mpc


def main():
    # Define time step
    dt = 0.2
    # Define prediction horizon
    N = 20
    # Define initial state
    x0 = np.array([0, 0, 0])
    # Define final state
    xf = np.array([10, 10, 0])
    # Create an instance of the MPC class
    mpc_planner = mpc.Mpc(dt, N)
    count = 0
    x_pos = []
    u0 = np.zeros((mpc_planner.n_controls, mpc_planner.N))
    while np.linalg.norm(x0 - xf) > 0.2 and count < 500:
        x, u = mpc_planner.perform_mpc(u0=u0, initial_state=x0, final_state=xf)
        x0 = x[:, 1]
        u0 = np.concatenate((u[:, 1:], u[:, -1].reshape(2, 1)), axis=1)
        count += 1
        x_pos.append(x0)
        print(x0)

    x_pos = np.array(x_pos)
    plt.plot(x_pos[:, 0], x_pos[:, 1])
    plt.show()


if __name__ == '__main__':
    main()

import casadi
import numpy as np
import matplotlib.pyplot as plt
from ros2_mpc.mpc_trajectory import Mpc


def main():
    dt = 0.1
    N = 20
    mpc = Mpc(dt, N)
    # Define initial state
    x0 = np.array([0, 0, 0])
    # Define final state
    xf = np.array([1, 1, 0])
    # Define initial control
    u0 = np.zeros((mpc.n_controls, mpc.N))
    count = 0
    x_pos = []
    theta_pos = []
    while count <= 300:
        current_time = count * dt
        pxf = np.array([])
        puf = np.array([])
        for k in range(mpc.N):
            t_predict = current_time + k * dt
            x_ref = 0.5 * t_predict * casadi.cos(np.deg2rad(45))
            y_ref = 0.5 * t_predict * casadi.sin(np.deg2rad(45))
            theta_ref = np.deg2rad(45)
            u_ref = 0.25
            omega_ref = 0
            if np.linalg.norm(x0[0:2] - xf[0:2]) < 0.1:
                x_ref = xf[0]
                y_ref = xf[1]
                u_ref = 0
                omega_ref = 0
            if k == 0:
                pxf = casadi.vertcat(x_ref, y_ref, theta_ref)
                puf = casadi.vertcat(u_ref, omega_ref)
            else:
                pxf = casadi.vertcat(pxf, casadi.vertcat(x_ref, y_ref, theta_ref))
                puf = casadi.vertcat(puf, casadi.vertcat(u_ref, omega_ref))

        x, u = mpc.perform_mpc(u0, x0, pxf, puf)
        x0 = x
        x_pos.append(x0)
        count += 1
        print(x0, xf)
        print('u = ', u)
        pass

    x_pos = np.array(x_pos)
    plt.plot(x_pos[:, 0], x_pos[:, 1])
    # Plot theta vs time
    # plt.plot(x_pos[:, 2])
    plt.show()


if __name__ == '__main__':
    main()

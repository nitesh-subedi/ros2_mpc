import rclpy
from ros2_mpc.ros_topics import OdomSubscriber, CmdVelPublisher
import numpy as np
from ros2_mpc.planner.local_planner_tracking import Mpc
from ros2_mpc.planner.global_planner import GlobalPlanner
import cv2


def main():
    rclpy.init()
    dt = 0.2
    N = 20
    map_image = cv2.imread("/home/nitesh/workspaces/ros2_mpc_ws/src/ros2_mpc/maps/map_carto.pgm", cv2.IMREAD_GRAYSCALE)
    map_image[map_image == 0] = 1
    map_image[map_image > 1] = 0
    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    map_image = cv2.dilate(map_image, kernel, iterations=1)
    map_image = map_image.astype(np.uint8)
    resolution = 0.05
    origin = np.array([-4.84, -6.61])
    odom_node = OdomSubscriber()
    cmd_vel_publisher = CmdVelPublisher()
    # Get the current position of the robot
    pos, ori, velocity = odom_node.get_states()
    robot_on_map = ((np.array([pos[0], pos[1]]) - origin) / resolution).astype(np.int32)
    # Change the origin from bottom left to top left
    robot_on_map[1] = map_image.shape[0] - robot_on_map[1]
    start = (robot_on_map[1], robot_on_map[0])
    goal_xy = (3.0, -0.5)  # World coordinates
    # Convert goal to map coordinates
    goal = ((goal_xy - origin) / resolution).astype(np.int32)
    # Change the origin from bottom left to top left
    goal[1] = map_image.shape[0] - goal[1]
    # Swap the x and y coordinates
    goal = (goal[1], goal[0])
    planner = GlobalPlanner(map_image)
    path = planner.get_path(start, goal)
    # Convert back to bottom left origin
    path = np.array(path)
    path = np.column_stack((path[:, 1], map_image.shape[0] - path[:, 0]))
    # Convert back to world coordinates
    path_xy = path * resolution + origin
    # Compute the heading angle
    path_heading = np.arctan2(path_xy[1:, 1] - path_xy[:-1, 1], path_xy[1:, 0] - path_xy[:-1, 0])
    path_heading = np.append(path_heading, path_heading[-1])
    # Compute the angular velocity
    path_omega = (path_heading[1:] - path_heading[:-1]) / 2
    # Compute the velocity
    path_velocity = (np.linalg.norm(path_xy[1:, :] - path_xy[:-1, :], axis=1) / dt) * 2
    path_velocity = np.append(path_velocity, path_velocity[-1])
    mpc = Mpc(dt, N)
    # Define initial state
    x0 = np.array([pos[0], pos[1], ori[2]])
    # Define initial control
    u0 = np.zeros((mpc.n_controls, mpc.N))
    count = 0
    x_pos = []
    while np.linalg.norm(x0[0:2] - path_xy[-1, :]) > 0.25:
        x_pos.append(x0)
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
        # Solve the optimization problem
        if pxf.shape[0] + 3 != mpc.P_X.shape[0]:
            print("pxf shape: ", pxf.shape)
            print("mpc.P_X shape: ", mpc.P_X.shape)
            print("puf shape: ", puf.shape)
            print("mpc.P_U shape: ", mpc.P_U.shape)
            break
        x, u = mpc.perform_mpc(u0, x0, pxf, puf)
        print("controls: ", u)
        print("next state: ", pxf[0:3])
        # Publish the control
        cmd_vel_publisher.publish_cmd(u[0], u[1])
        # cmd_vel_publisher.publish_cmd(0.0, 0.0)
        count += 1

        pos, ori, velocity = odom_node.get_states()
        x0 = np.array([pos[0], pos[1], ori[2]])

    cmd_vel_publisher.publish_cmd(0.0, 0.0)

    # x_pos = np.array(x_pos)
    # plt.plot(x_pos[:, 0], x_pos[:, 1])
    # # Plot theta vs time
    # # plt.plot(x_pos[:, 2])
    # plt.show()


if __name__ == '__main__':
    main()

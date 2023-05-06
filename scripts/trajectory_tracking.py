import rclpy
from ros2_mpc.ros_topics import OdomSubscriber, CmdVelPublisher, MapSubscriber
import numpy as np
from ros2_mpc.planner.local_planner_tracking import Mpc
from ros2_mpc.planner.global_planner import GlobalPlanner
import cv2
from ros2_mpc import utils


def dilate_image(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    image = cv2.dilate(image, kernel, iterations=1)
    return image.astype(np.uint8)


def get_headings(path_xy, dt):
    # Compute the heading angle
    path_heading = np.arctan2(path_xy[1:, 1] - path_xy[:-1, 1], path_xy[1:, 0] - path_xy[:-1, 0])
    path_heading = np.append(path_heading, path_heading[-1])
    # Compute the angular velocity
    path_omega = (path_heading[1:] - path_heading[:-1]) / 2
    # Compute the velocity
    path_velocity = (np.linalg.norm(path_xy[1:, :] - path_xy[:-1, :], axis=1) / dt) * 2
    path_velocity = np.append(path_velocity, path_velocity[-1])
    return path_heading, path_velocity, path_omega


def get_reference_trajectory(x0, goal_xy, path_xy, path_heading, path_velocity, path_omega, mpc):
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
    return pxf, puf


def main():
    rclpy.init()
    dt = 0.2
    N = 20
    map_node = MapSubscriber()
    odom_node = OdomSubscriber()
    cmd_vel_publisher = CmdVelPublisher()
    planner = GlobalPlanner()
    mpc = Mpc(dt, N)

    map_image, map_info = map_node.get_map()
    pos, ori, velocity = odom_node.get_states()
    # Dilate the map image
    map_image = dilate_image(map_image, 10)
    # Get the current position of the robot
    robot_on_map = utils.world_to_map(pos[0], pos[1], map_image, map_info)
    start = (robot_on_map[1], robot_on_map[0])
    # Get the goal position of the robot
    goal_xy = np.array([4.0, 2.0])
    goal_on_map = utils.world_to_map(goal_xy[0], goal_xy[1], map_image, map_info)
    # Swap the x and y coordinates
    goal = (goal_on_map[1], goal_on_map[0])
    path = planner.get_path(start, goal, map_image)
    # Convert the path to world coordinates
    path_xy = utils.map_to_world(path, map_image, map_info)
    # Compute the headings
    path_heading, path_velocity, path_omega = get_headings(path_xy, dt)
    # Define initial state
    x0 = np.array([pos[0], pos[1], ori[2]])
    # Define initial control
    u0 = np.zeros((mpc.n_controls, mpc.N))
    count = 0
    x_pos = []
    while np.linalg.norm(x0[0:2] - path_xy[-1, :]) > 0.15 and count <= 500:
        x_pos.append(x0)
        # Get the reference trajectory
        pxf, puf = get_reference_trajectory(x0, goal_xy, path_xy, path_heading, path_velocity, path_omega, mpc)
        x, u = mpc.perform_mpc(u0, x0, pxf, puf)
        # Publish the control
        cmd_vel_publisher.publish_cmd(u[0], u[1])
        count += 1
        pos, ori, velocity = odom_node.get_states()
        x0 = np.array([pos[0], pos[1], ori[2]])

    cmd_vel_publisher.publish_cmd(0.0, 0.0)
    print("Goal Reached!")


if __name__ == '__main__':
    main()

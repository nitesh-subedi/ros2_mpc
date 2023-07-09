import time
import numpy as np
from ros2_mpc.utils import convert_laser_scan_to_occupancy_grid
from ros2_mpc.ros_topics import OdomSubscriber, LocalCostmapPublisher, LaserSubscriber
import cv2
import rclpy
import os
import yaml
from ament_index_python.packages import get_package_share_directory


def main():
    rclpy.init()
    scan_subscriber = LaserSubscriber()
    odom_subscriber = OdomSubscriber()
    costmap_publisher = LocalCostmapPublisher()
    project_path = get_package_share_directory('ros2_mpc')
    with open(os.path.join(project_path, 'config/params.yaml'), 'r') as file:
        params = yaml.safe_load(file)
    costmap_size = params['costmap_size']
    resolution = params['resolution']
    while True:
        scan, angles = scan_subscriber.get_scan()
        if scan is None:
            continue
        position, orientation = odom_subscriber.get_states()
        if position is None:
            continue
        # Convert the laser scan to occupancy grid
        occupancy_grid = convert_laser_scan_to_occupancy_grid(scan, angles, map_resolution=resolution,
                                                              map_size=costmap_size * 2, rotation=orientation[2])
        # occupancy_grid = np.flipud(occupancy_grid)
        # Inflate the occupancy grid by 5 cells
        inflation_matrix = np.ones((10, 10))
        inflated_grid = cv2.dilate(occupancy_grid, inflation_matrix, iterations=1).astype(np.uint8)
        # Rotate the inflated grid according to the orientation of the robot
        costmap_publisher.publish_costmap(inflated_grid, costmap_size * 2, position)
        time.sleep(0.1)


if __name__ == "__main__":
    main()

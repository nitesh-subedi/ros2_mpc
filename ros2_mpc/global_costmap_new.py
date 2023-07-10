import time
import numpy as np
from ros2_mpc import utils
from ros2_mpc.ros_topics import OdomSubscriber, GlobalCostmapPublisher, LaserSubscriber, MapSubscriber
import cv2
import rclpy


def main():
    rclpy.init()
    scan_subscriber = LaserSubscriber()
    odom_subscriber = OdomSubscriber()
    costmap_publisher = GlobalCostmapPublisher()
    map_subscriber = MapSubscriber()
    while True:
        scan, angles = scan_subscriber.get_scan()
        if scan is None:
            continue
        position, orientation = odom_subscriber.get_states()
        if position is None:
            continue
        global_map_image, map_info = map_subscriber.get_map()
        if global_map_image is None:
            continue
        # Convert scan data to xy coordinates
        x_scan, y_scan = utils.convert_laser_scan_to_xy_coordinates(scan, angles, rotation=orientation[2])
        # Add the robot position to the xy coordinates
        x_scan += position[0]
        y_scan += position[1]
        # Convert global map to xy coordinates
        x_map, y_map = utils.convert_occ_grid_to_xy_coordinates(global_map_image, map_info['resolution'],
                                                                map_info['origin'])
        # Merge the xy coordinates of the map and the scan
        x = np.concatenate((x_scan, x_map))
        y = np.concatenate((y_scan, y_map))
        # # Convert the xy coordinates to occupancy grid
        occupancy_grid = utils.convert_xy_coordinates_to_occ_grid(x, y, map_size=np.array(global_map_image.shape),
                                                                  map_resolution=map_info['resolution'],
                                                                  map_origin=map_info['origin'])
        # Inflate the occupancy grid by 5 cells
        inflation_matrix = np.ones((10, 10))
        inflated_grid = cv2.dilate(occupancy_grid, inflation_matrix, iterations=1).astype(np.uint8)
        costmap_publisher.publish_costmap(inflated_grid, map_info['origin'])


if __name__ == "__main__":
    main()

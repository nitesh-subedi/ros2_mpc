import numpy as np
from ros2_mpc import utils
from ros2_mpc.core.ros_topics import OdomSubscriber, GlobalCostmapPublisher, LaserSubscriber
import cv2
import rclpy
import os
from ament_index_python.packages import get_package_share_directory
import yaml


def main():
    rclpy.init()
    scan_subscriber = LaserSubscriber()
    odom_subscriber = OdomSubscriber()
    costmap_publisher = GlobalCostmapPublisher()
    while True:
        scan, angles = scan_subscriber.get_scan()
        if scan is None:
            continue
        position, orientation = odom_subscriber.get_states()
        if position is None:
            continue
        with open(os.path.join(get_package_share_directory('ros2_mpc'), 'maps', 'map_carto.yaml'), 'r') as f:
            map_yaml = yaml.safe_load(f)
        map_info = {'origin': [map_yaml['origin'][0], map_yaml['origin'][1]], 'resolution': map_yaml['resolution']}
        map_name = map_yaml['image']
        map_image = cv2.imread(os.path.join(get_package_share_directory('ros2_mpc'), 'maps', map_name))
        map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)
        map_image[map_image == 0] = 255
        map_image[map_image == 254] = 0
        map_image[map_image == 205] = 0
        map_image = np.array(map_image).astype(np.int8)
        global_map_image = np.flipud(map_image)
        # global_map_image, map_info = map_subscriber.get_map()
        # if global_map_image is None:
        #     continue
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

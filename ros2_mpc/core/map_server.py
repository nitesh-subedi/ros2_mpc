import time
import numpy as np
import rclpy
from ros2_mpc.core.ros_topics import MapServer
from ament_index_python.packages import get_package_share_directory
import cv2
import os
import yaml


def main():
    rclpy.init()
    map_server = MapServer()
    map_image = cv2.imread(os.path.join(get_package_share_directory('ros2_mpc'), 'maps', 'map_carto.pgm'))
    map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)
    map_image[map_image == 0] = 100
    map_image[map_image == 254] = 0
    map_image[map_image == 205] = 0
    map_image = np.array(map_image).astype(np.int8)
    map_image = np.flipud(map_image)
    with open(os.path.join(get_package_share_directory('ros2_mpc'), 'maps', 'map_carto.yaml'), 'r') as f:
        map_yaml = yaml.safe_load(f)
    map_info = {'origin': [map_yaml['origin'][0], map_yaml['origin'][1]], 'resolution': map_yaml['resolution']}
    last_count = str("0")
    while rclpy.ok():
        time.sleep(0.2)
        if str(map_server.publisher.get_subscription_count()) != last_count \
                and map_server.publisher.get_subscription_count() > 0:
            map_server.get_logger().info("Subscriptions to map:" + str(map_server.publisher.get_subscription_count()))
            map_server.publish_map(map_image, map_info)
            last_count = str(map_server.publisher.get_subscription_count())


if __name__ == "__main__":
    main()

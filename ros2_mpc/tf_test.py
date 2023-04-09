#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_msgs.msg import TFMessage
import numpy as np


class RobotPositionSubscriber(Node):
    def __init__(self):
        super().__init__("robot_position_subscriber")
        self.map_to_odom = None
        self.odom_to_base_footprint = None
        self.robot_position = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.subscription = self.create_subscription(TFMessage, "/tf", self.listener_callback, 10)

    def listener_callback(self, data):
        if not data.transforms:
            pass
        else:
            for transform in data.transforms:
                if transform.child_frame_id == "odom" and transform.header.frame_id == "map":
                    self.map_to_odom = np.array([transform.transform.translation.x, transform.transform.translation.y])
                    # self.get_logger().info("Odom to map: x = %f, y = %f" % (transform.transform.translation.x,
                    # transform.transform.translation.y))
                if transform.child_frame_id == "base_footprint" and transform.header.frame_id == "odom":
                    self.odom_to_base_footprint = np.array(
                        [transform.transform.translation.x, transform.transform.translation.y])
                    # self.get_logger().info("Base footprint to odom: x = %f, y = %f" % (
                    # transform.transform.translation.x, transform.transform.translation.y))
                if self.map_to_odom is not None and self.odom_to_base_footprint is not None:
                    self.robot_position = self.map_to_odom + self.odom_to_base_footprint
                    self.get_logger().info(
                        "Robot position on map: x = %f, y = %f" % (self.robot_position[0], self.robot_position[1]))


def main(args=None):
    rclpy.init(args=args)
    node = RobotPositionSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

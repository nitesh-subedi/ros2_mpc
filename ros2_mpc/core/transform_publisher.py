#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomSubscriber(Node):
    def __init__(self):
        super().__init__('map_odom_transform')
        self.t = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 50)
        self.map_broadcaster = TransformBroadcaster(self)

    def odom_callback(self, msg):
        self.t = TransformStamped()
        self.t.header.stamp = self.get_clock().now().to_msg()
        self.t.header.frame_id = "map"
        self.t.child_frame_id = "odom"
        self.t.transform.translation.x = -msg.pose.pose.position.x
        self.t.transform.translation.y = -msg.pose.pose.position.y
        self.t.transform.translation.z = 0.0
        self.t.transform.rotation = msg.pose.pose.orientation
        self.map_broadcaster.sendTransform(self.t)


def main(args=None):
    rclpy.init(args=args)
    map_odom_transform = OdomSubscriber()
    rclpy.spin(map_odom_transform)


if __name__ == "__main__":
    main()

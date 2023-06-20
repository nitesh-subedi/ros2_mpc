#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
import math


class SubscriberLaserPose(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.position_publisher = self.create_publisher(Odometry, '/robot_position', 50)
        timer_period = 0.05  # seconds
        self.robot_position = Odometry()
        self.timer = self.create_timer(timer_period, self.take_action)

    def get_transform(self):
        try:
            # noinspection PyUnresolvedReferences
            trans = self.tf_buffer.lookup_transform(
                "map",  # target_frame
                "base_footprint",  # src frame
                rclpy.time.Time())
            return trans
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform base_footprint to map: {ex}')
            return None

    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    def take_action(self):
        trans = self.get_transform()
        if trans is None:
            return
        else:
            self.robot_position.pose.pose.position.x = trans.transform.translation.x
            self.robot_position.pose.pose.position.y = trans.transform.translation.y
            self.robot_position.pose.pose.orientation = trans.transform.rotation
            self.position_publisher.publish(self.robot_position)
            # self.get_logger().info("Robot position: " + str(self.robot_position.pose.pose.position.x) + ", " + str(
            #     self.robot_position.pose.pose.position.y))


def main(args=None):
    rclpy.init(args=args)
    laser_pose_subscriber = SubscriberLaserPose()
    rclpy.spin(laser_pose_subscriber)
    laser_pose_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

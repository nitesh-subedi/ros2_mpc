# Launch parameters for all nodes
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Get the launch directory
    # Specify the actions
    path_subscriber_local_planner_node = Node(
        package='ros2_mpc',
        executable='local_planner',
        output='screen',
        remappings=[('/cmd_vel', '/cmd_vel'), ('/odom', '/odom'), ('/path', '/path')],
        arguments=['--ros-args', '--log-level', 'INFO']
    )
    path_publisher_node = Node(
        package='ros2_mpc',
        executable='path_publisher',
        output='screen',
        arguments=['--ros-args', '--log-level', 'INFO']
    )

    robot_state_publisher_node = Node(
        package='ros2_mpc',
        executable='robot_state_publisher',
        output='screen',
        arguments=['--ros-args', '--log-level', 'INFO']
    )

    # Create the launch description and populate
    ld = LaunchDescription()
    # Declare the launch options
    ld.add_action(path_subscriber_local_planner_node)
    ld.add_action(path_publisher_node)
    ld.add_action(robot_state_publisher_node)
    return ld

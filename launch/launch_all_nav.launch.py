# Launch parameters for all nodes
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rviz_config_dir = os.path.join(get_package_share_directory('ros2_mpc'), 'config', 'rviz_config.rviz')
    # Get the launch directory
    # Specify the actions
    path_subscriber_local_planner_node = Node(
        package='ros2_mpc',
        executable='local_point_follower',
        output='screen',
        remappings=[('/cmd_vel', '/cmd_vel'), ('/odom', '/odom'), ('/path', '/path')],
        arguments=['--ros-args', '--log-level', 'INFO']
    )
    path_publisher_node = Node(
        package='ros2_mpc',
        executable='path_publisher',
        output='screen',
        arguments=['--ros-args', '--log-level', 'INFO']  # ,
        # parameters=[{'use_sim_time': True}]
    )

    robot_state_publisher_node = Node(
        package='ros2_mpc',
        executable='robot_state_publisher',
        output='screen',
        arguments=['--ros-args', '--log-level', 'INFO']
    )

    global_costmap_publisher_node = Node(
        package='ros2_mpc',
        executable='global_costmap_publisher',
        output='screen',
        arguments=['--ros-args', '--log-level', 'INFO']
    )

    local_costmap_publisher_node = Node(
        package='ros2_mpc',
        executable='local_costmap_publisher',
        output='screen',
        arguments=['--ros-args', '--log-level', 'INFO']
    )

    map_server_node = Node(
        package='ros2_mpc',
        executable='map_server',
        output='screen',
        arguments=['--ros-args', '--log-level', 'INFO']
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir]
    )

    # Create the launch description and populate
    ld = LaunchDescription()
    # Declare the launch options
    ld.add_action(path_subscriber_local_planner_node)
    ld.add_action(path_publisher_node)
    ld.add_action(robot_state_publisher_node)
    ld.add_action(global_costmap_publisher_node)
    ld.add_action(local_costmap_publisher_node)
    # ld.add_action(map_server_node)
    ld.add_action(rviz_node)
    return ld

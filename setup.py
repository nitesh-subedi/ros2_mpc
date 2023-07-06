from setuptools import setup
import os
from glob import glob

package_name = 'ros2_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/params.yaml']),
        ('share/' + package_name + '/config', ['config/rviz_config.rviz']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nitesh22',
    maintainer_email='subedinitesh43@gmail.com',
    description='MPC local planner for ROS 2 Navigation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'path_publisher = ros2_mpc.scripts.global_path_publisher:main',
            'local_planner = ros2_mpc.scripts.path_follower_local_planner:main',
            'local_point_follower = ros2_mpc.scripts.point_follower_local_planner:main',
            'robot_state_publisher = ros2_mpc.core.robot_state_publisher:main',
        ],
    },
)

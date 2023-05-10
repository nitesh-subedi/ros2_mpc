from setuptools import setup

package_name = 'ros2_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nitesh22',
    maintainer_email='subedinitesh43@gmail.com',
    description='MPC local planner for ROS 2 Navigation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'path_publisher = ros2_mpc.scripts.path_publisher:main',
        ],
    },
)

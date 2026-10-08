"""
Starts YOUR leader and follower nodes.

Evaluation (simulation + Nav2 already started by the organisers):
    ros2 launch clue_hunt_solver hunt.launch.py

Self-contained (bring up sim + AMCL + Nav2 + RViz from this launch too -
Phase 1 verification uses this):
    ros2 launch clue_hunt_solver hunt.launch.py start_nav:=true

Arguments:
    start_nav    false (default): only our two nodes, Nav2 assumed running.
                 true: also include clue_hunt_navigation/navigation.launch.py
                       (sim + map_server + AMCL + Nav2 + RViz).
    map          map yaml (default: cleaned arena map inside this package);
                 forwarded to navigation.launch.py and to hunt_node
                 (parameter map_yaml - used from Phase 3 for pillar extraction).
    params_file  Nav2 params (default: our config/nav2_params.yaml copy).
    sim, rviz    forwarded to navigation.launch.py when start_nav:=true.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    solver_share = get_package_share_directory('clue_hunt_solver')
    nav_share = get_package_share_directory('clue_hunt_navigation')

    default_map = os.path.join(solver_share, 'maps', 'arena.yaml')
    default_params = os.path.join(solver_share, 'config', 'nav2_params.yaml')

    declare_args = [
        DeclareLaunchArgument(
            'start_nav', default_value='false',
            description='true: also start sim + AMCL + Nav2 + RViz'),
        DeclareLaunchArgument(
            'map', default_value=default_map,
            description='Map yaml (navigation.launch + hunt_node map_yaml)'),
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='Nav2 params file (our tuned copy)'),
        DeclareLaunchArgument(
            'sim', default_value='true',
            description='When start_nav:=true, also start Gazebo'),
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='When start_nav:=true, also start RViz'),
    ]

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav_share, 'launch', 'navigation.launch.py')),
        launch_arguments={
            'map': LaunchConfiguration('map'),
            'params_file': LaunchConfiguration('params_file'),
            'sim': LaunchConfiguration('sim'),
            'rviz': LaunchConfiguration('rviz'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('start_nav')),
    )

    hunt_node = Node(
        package='clue_hunt_solver', executable='hunt_node', output='screen',
        parameters=[{
            'use_sim_time': True,
            'map_yaml': LaunchConfiguration('map'),
        }])

    follower_node = Node(
        package='clue_hunt_solver', executable='follower_node', output='screen',
        parameters=[{'use_sim_time': True}])

    return LaunchDescription(declare_args + [navigation, hunt_node, follower_node])

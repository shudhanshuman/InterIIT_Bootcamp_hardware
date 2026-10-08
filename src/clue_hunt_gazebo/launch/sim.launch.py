"""
Start Gazebo Fortress with a Clue Chain Hunt world, spawn the LEADER (and the FOLLOWER)
and bridge their topics to ROS 2.

  ros2 launch clue_hunt_gazebo sim.launch.py                        # practice world, 2 robots
  ros2 launch clue_hunt_gazebo sim.launch.py follower:=false        # leader only (e.g. mapping)
  ros2 launch clue_hunt_gazebo sim.launch.py gui:=false             # headless
  ros2 launch clue_hunt_gazebo sim.launch.py world:=<name> world_pkg:=<package with worlds/>

Leader   : model "hunter",   topics /scan /camera/... /cmd_vel /odom,        frames odom, base_link, ...
Follower : model "follower", topics /follower/camera/... /follower/cmd_vel
                             /follower/odom,                                 frames follower/odom, ...
Both robots publish their TF on the shared /tf topic.
"""
import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ROS topic @ ROS type [ = Gazebo -> ROS,  ] = ROS -> Gazebo
LEADER_TOPICS = [
    '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
    '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
    '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
    '/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
    '/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model',
    '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
    '/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU',
    '/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
    '/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
]
FOLLOWER_TOPICS = [
    '/follower/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
    '/follower/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
    '/follower/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
    '/follower/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model',
    '/follower/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
    '/follower/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
]
FOLLOWER_XACRO_ARGS = {
    'prefix': 'follower/', 'topic_ns': '/follower/', 'lidar': 'false', 'imu': 'false',
    'rear_tag': 'false', 'camera_pitch': '-0.14', 'body': 'blue',
}


def launch_setup(context):
    world = LaunchConfiguration('world').perform(context)
    world_pkg = LaunchConfiguration('world_pkg').perform(context)
    gui = LaunchConfiguration('gui').perform(context).lower() in ('true', '1', 'yes')
    with_follower = LaunchConfiguration('follower').perform(context).lower() in ('true', '1', 'yes')

    world_share = get_package_share_directory(world_pkg)
    world_file = world if world.endswith('.sdf') else os.path.join(world_share, 'worlds', world + '.sdf')
    if not os.path.isfile(world_file):
        raise RuntimeError(f'World file not found: {world_file}')

    # Board + banner models are found through model://...
    model_paths = [os.path.join(world_share, 'models'),
                   os.path.join(get_package_share_directory('clue_hunt_gazebo'), 'models')]
    resource_path = os.pathsep.join(model_paths + [os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')])

    xacro_file = os.path.join(get_package_share_directory('clue_hunt_description'), 'urdf', 'robot.urdf.xacro')
    leader_urdf = xacro.process_file(xacro_file).toxml()

    gz_cmd = ['ign', 'gazebo', '-r', '-v', '2', world_file]
    if not gui:
        gz_cmd.insert(2, '-s')

    actions = [
        SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', resource_path),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_path),
        ExecuteProcess(cmd=gz_cmd, output='screen'),

        # ---------------- leader ----------------
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': leader_urdf, 'use_sim_time': True}],
        ),
        Node(
            package='ros_gz_sim',
            executable='create',
            output='screen',
            arguments=['-name', 'hunter', '-topic', 'robot_description',
                       '-x', '0.0', '-y', '0.0', '-z', '0.02', '-Y', '0.0'],
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='leader_bridge',
            output='screen',
            arguments=LEADER_TOPICS,
            parameters=[{'use_sim_time': True}],
        ),
    ]

    if with_follower:
        follower_urdf = xacro.process_file(xacro_file, mappings=FOLLOWER_XACRO_ARGS).toxml()
        actions += [
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                namespace='follower',
                output='screen',
                parameters=[{'robot_description': follower_urdf, 'use_sim_time': True,
                             'frame_prefix': 'follower/'}],
            ),
            Node(
                package='ros_gz_sim',
                executable='create',
                output='screen',
                arguments=['-name', 'follower', '-topic', '/follower/robot_description',
                           '-x', '-1.0', '-y', '0.0', '-z', '0.02', '-Y', '0.0'],
            ),
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name='follower_bridge',
                output='screen',
                arguments=FOLLOWER_TOPICS,
                remappings=[('/follower/tf', '/tf')],
                parameters=[{'use_sim_time': True}],
            ),
        ]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='practice',
                              description='World name (file in <world_pkg>/worlds) or absolute .sdf path'),
        DeclareLaunchArgument('world_pkg', default_value='clue_hunt_gazebo',
                              description='Package that contains the world and its board models'),
        DeclareLaunchArgument('gui', default_value='true', description='Show the Gazebo GUI'),
        DeclareLaunchArgument('follower', default_value='true', description='Also spawn the follower robot'),
        OpaqueFunction(function=launch_setup),
    ])

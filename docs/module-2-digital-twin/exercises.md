---
sidebar_label: 'Digital Twin Exercises'
sidebar_position: 3
---

# Digital Twin Exercises: Hands-On Simulation for Humanoid Robotics

## Exercise 1: Setting Up Your First Humanoid Robot Simulation

### Objective
Create a basic humanoid robot model in Gazebo and establish ROS 2 communication.

### Prerequisites
- ROS 2 Humble installed
- Gazebo Garden installed
- Completed Module 1 (ROS 2 basics)

### Steps

#### Step 1: Create a Robot Description Package
```bash
cd ~/humanoid_ws/src
ros2 pkg create --build-type ament_cmake my_humanoid_description --dependencies urdf xacro
```

#### Step 2: Create a Simple Humanoid URDF
Create `my_humanoid_description/urdf/simple_humanoid.urdf`:
```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="head_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_thigh"/>
    <origin xyz="-0.1 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="left_thigh">
    <visual>
      <geometry>
        <box size="0.08 0.08 0.3"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.08 0.08 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="2.35" effort="100" velocity="1"/>
  </joint>

  <link name="left_shin">
    <visual>
      <geometry>
        <box size="0.08 0.08 0.3"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.08 0.08 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Right Leg (similar to left) -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_thigh"/>
    <origin xyz="0.1 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="right_thigh">
    <visual>
      <geometry>
        <box size="0.08 0.08 0.3"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.08 0.08 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="2.35" effort="100" velocity="1"/>
  </joint>

  <link name="right_shin">
    <visual>
      <geometry>
        <box size="0.08 0.08 0.3"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.08 0.08 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.2"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.15"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Right Arm (similar to left) -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.15 -0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.2"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.15"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>
</robot>
```

#### Step 3: Create a Gazebo World File
Create `my_humanoid_description/worlds/simple_humanoid.world`:
```xml
<sdf version="1.7">
  <world name="simple_humanoid_world">
    <!-- Physics settings -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Spawn our humanoid robot -->
    <include>
      <uri>model://simple_humanoid</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>
  </world>
</sdf>
```

#### Step 4: Create Launch File
Create `my_humanoid_description/launch/humanoid_sim.launch.py`:
```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    # Start Gazebo
    gazebo = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'simple_humanoid',
            '-file', PathJoinSubstitution([
                FindPackageShare('my_humanoid_description'),
                'urdf', 'simple_humanoid.urdf'
            ])
        ],
        output='screen'
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_description': PathJoinSubstitution([
                FindPackageShare('my_humanoid_description'),
                'urdf', 'simple_humanoid.urdf'
            ])
        }]
    )

    return LaunchDescription([
        use_sim_time,
        robot_state_publisher,
        gazebo
    ])
```

#### Step 5: Build and Run
```bash
cd ~/humanoid_ws
colcon build --packages-select my_humanoid_description
source install/setup.bash

# Launch Gazebo with your robot
ros2 launch gazebo_ros empty_world.launch.py world_path:=$(ros2 pkg prefix my_humanoid_description)/share/my_humanoid_description/worlds/simple_humanoid.world
```

### Expected Outcome
You should see your simple humanoid robot model loaded in the Gazebo simulation environment.

## Exercise 2: Adding Joint Control to Your Robot

### Objective
Add ROS 2 control interfaces to your simulated humanoid robot.

### Steps

#### Step 1: Install ROS 2 Control Packages
```bash
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers
sudo apt install ros-humble-gazebo-ros2-control
```

#### Step 2: Update URDF with Control Interfaces
Create `my_humanoid_description/urdf/simple_humanoid_control.xacro`:
```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Load the basic robot -->
  <xacro:include filename="simple_humanoid.urdf"/>

  <!-- ros2_control Interface -->
  <ros2_control name="GazeboSystem" type="system">
    <hardware>
      <plugin>gazebo_ros2_control/GazeboSystem</plugin>
    </hardware>

    <joint name="head_joint">
      <command_interface name="position">
        <param name="min">-0.5</param>
        <param name="max">0.5</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>

    <joint name="left_hip_joint">
      <command_interface name="position">
        <param name="min">-1.57</param>
        <param name="max">1.57</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>

    <joint name="left_knee_joint">
      <command_interface name="position">
        <param name="min">0</param>
        <param name="max">2.35</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>

    <joint name="right_hip_joint">
      <command_interface name="position">
        <param name="min">-1.57</param>
        <param name="max">1.57</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>

    <joint name="right_knee_joint">
      <command_interface name="position">
        <param name="min">0</param>
        <param name="max">2.35</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>

    <joint name="left_shoulder_joint">
      <command_interface name="position">
        <param name="min">-1.57</param>
        <param name="max">1.57</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>

    <joint name="left_elbow_joint">
      <command_interface name="position">
        <param name="min">-1.57</param>
        <param name="max">1.57</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>

    <joint name="right_shoulder_joint">
      <command_interface name="position">
        <param name="min">-1.57</param>
        <param name="max">1.57</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>

    <joint name="right_elbow_joint">
      <command_interface name="position">
        <param name="min">-1.57</param>
        <param name="max">1.57</param>
      </command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
  </ros2_control>

  <!-- Load the controller manager -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find my_humanoid_description)/config/humanoid_controllers.yaml</parameters>
    </plugin>
  </gazebo>
</robot>
```

#### Step 3: Create Controller Configuration
Create `my_humanoid_description/config/humanoid_controllers.yaml`:
```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    head_position_controller:
      type: position_controllers/JointPositionController

    left_leg_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    right_leg_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    left_arm_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    right_arm_controller:
      type: joint_trajectory_controller/JointTrajectoryController

head_position_controller:
  ros__parameters:
    joint: head_joint
    interface_name: position

left_leg_controller:
  ros__parameters:
    joints:
      - left_hip_joint
      - left_knee_joint
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity

right_leg_controller:
  ros__parameters:
    joints:
      - right_hip_joint
      - right_knee_joint
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity

left_arm_controller:
  ros__parameters:
    joints:
      - left_shoulder_joint
      - left_elbow_joint
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity

right_arm_controller:
  ros__parameters:
    joints:
      - right_shoulder_joint
      - right_elbow_joint
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity
```

#### Step 4: Create Controller Spawning Launch File
Create `my_humanoid_description/launch/humanoid_control.launch.py`:
```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    # Robot State Publisher
    robot_description_content = Command([
        'xacro ',
        PathJoinSubstitution([
            FindPackageShare('my_humanoid_description'),
            'urdf', 'simple_humanoid_control.xacro'
        ])
    ])

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_description': robot_description_content
        }]
    )

    # Gazebo
    gazebo = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'simple_humanoid',
            '-topic', 'robot_description'
        ],
        output='screen'
    )

    # Load and start controllers after Gazebo has started
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    head_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['head_position_controller'],
        output='screen'
    )

    left_leg_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['left_leg_controller'],
        output='screen'
    )

    right_leg_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['right_leg_controller'],
        output='screen'
    )

    left_arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['left_arm_controller'],
        output='screen'
    )

    right_arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['right_arm_controller'],
        output='screen'
    )

    # Delay controllers startup until Gazebo is ready
    delay_joint_broadcaster_spawner_after_gazebo = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=gazebo,
            on_start=[
                joint_state_broadcaster_spawner,
            ],
        )
    )

    delay_head_controller_spawner_after_joint_broadcaster = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=joint_state_broadcaster_spawner,
            on_start=[
                head_controller_spawner,
                left_leg_controller_spawner,
                right_leg_controller_spawner,
                left_arm_controller_spawner,
                right_arm_controller_spawner
            ],
        )
    )

    return LaunchDescription([
        use_sim_time,
        robot_state_publisher,
        gazebo,
        delay_joint_broadcaster_spawner_after_gazebo,
        delay_head_controller_spawner_after_joint_broadcaster
    ])
```

#### Step 5: Build and Test
```bash
cd ~/humanoid_ws
colcon build --packages-select my_humanoid_description
source install/setup.bash

# Launch with control interfaces
ros2 launch my_humanoid_description humanoid_control.launch.py
```

### Expected Outcome
Your humanoid robot should now have ROS 2 control interfaces and be controllable through ROS 2 topics.

## Exercise 3: Creating Unity Simulation Environment

### Objective
Set up a Unity simulation environment that can communicate with ROS 2.

### Prerequisites
- Unity 2022.3 LTS installed
- ROS-TCP-Connector package installed in Unity

### Steps

#### Step 1: Create Unity Project
1. Open Unity Hub
2. Create a new 3D project named "HumanoidSimulation"
3. Install ROS-TCP-Connector from Package Manager

#### Step 2: Create Robot Model in Unity
1. Import your robot model (or create simple primitives)
2. Set up the robot hierarchy with proper joints
3. Configure physics properties for each part

#### Step 3: Create ROS Connection Manager
Create a C# script `Assets/Scripts/ROSConnectionManager.cs`:
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Control;

public class ROSConnectionManager : MonoBehaviour
{
    [SerializeField] private string rosIPAddress = "127.0.0.1";
    [SerializeField] private int rosPort = 10000;

    private ROSConnection ros;
    private JointStateMsg jointState;

    void Start()
    {
        ros = ROSConnection.instance;
        ros.Initialize(rosIPAddress, rosPort);

        // Subscribe to joint commands
        ros.Subscribe<JointTrajectoryMsg>("/left_arm_controller/joint_trajectory", OnJointTrajectoryReceived);
        ros.Subscribe<JointTrajectoryMsg>("/right_arm_controller/joint_trajectory", OnJointTrajectoryReceived);
    }

    void OnJointTrajectoryReceived(JointTrajectoryMsg trajectory)
    {
        // Process joint trajectory commands
        Debug.Log($"Received trajectory with {trajectory.points.Length} points");

        // Update Unity robot model based on trajectory
        UpdateRobotModel(trajectory);
    }

    void UpdateRobotModel(JointTrajectoryMsg trajectory)
    {
        // This is a simplified example - you'd implement actual joint control here
        if (trajectory.points.Length > 0)
        {
            var point = trajectory.points[0];

            // Example: Update joint positions in Unity
            // You would map ROS joint names to Unity transforms here
        }
    }

    void Update()
    {
        // Publish joint states periodically
        PublishJointStates();
    }

    void PublishJointStates()
    {
        // Create and publish joint state message
        jointState = new JointStateMsg();
        jointState.header.stamp = new builtin_interfaces.msg.Time();
        jointState.name = new string[] { "head_joint", "left_shoulder_joint", "left_elbow_joint" };
        jointState.position = new double[] { 0.1, 0.2, 0.3 }; // Example values

        ros.Publish("/joint_states", jointState);
    }
}
```

#### Step 4: Set Up Scene
1. Create an empty GameObject named "ROSConnection"
2. Attach the ROSConnectionManager script
3. Configure IP address and port to match your ROS 2 setup

#### Step 5: Test Unity-ROS Connection
1. Build and run the Unity application
2. In another terminal, start ROS 2 bridge
3. Verify communication between Unity and ROS 2

### Expected Outcome
Unity should be able to send and receive ROS 2 messages, creating a bridge between high-fidelity visualization and ROS 2 control systems.

## Exercise 4: Validation and Comparison

### Objective
Validate your simulation by comparing behavior between Gazebo and Unity.

### Steps

#### Step 1: Create Validation Node
Create a package for validation:
```bash
cd ~/humanoid_ws/src
ros2 pkg create --build-type ament_python simulation_validation --dependencies rclpy std_msgs sensor_msgs
```

#### Step 2: Create Validation Script
Create `simulation_validation/simulation_validation/validator.py`:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

class SimulationValidator(Node):
    def __init__(self):
        super().__init__('simulation_validator')

        # Subscribe to joint states from both simulators
        self.gazebo_subscription = self.create_subscription(
            JointState,
            'gazebo_joint_states',
            self.gazebo_callback,
            10)

        self.unity_subscription = self.create_subscription(
            JointState,
            'unity_joint_states',
            self.unity_callback,
            10)

        # Publisher for validation results
        self.validation_publisher = self.create_publisher(
            Float64MultiArray,
            'validation_results',
            10)

        self.gazebo_joint_states = None
        self.unity_joint_states = None

        # Timer for validation checks
        self.timer = self.create_timer(1.0, self.validate_callback)

    def gazebo_callback(self, msg):
        self.gazebo_joint_states = msg

    def unity_callback(self, msg):
        self.unity_joint_states = msg

    def validate_callback(self):
        if self.gazebo_joint_states and self.unity_joint_states:
            # Compare joint positions between simulators
            gazebo_positions = np.array(self.gazebo_joint_states.position)
            unity_positions = np.array(self.unity_joint_states.position)

            # Calculate differences
            if len(gazebo_positions) == len(unity_positions):
                differences = np.abs(gazebo_positions - unity_positions)
                max_diff = np.max(differences)

                self.get_logger().info(f'Max position difference: {max_diff}')

                # Publish validation results
                result_msg = Float64MultiArray()
                result_msg.data = [max_diff, np.mean(differences)]
                self.validation_publisher.publish(result_msg)
            else:
                self.get_logger().warn('Joint state arrays have different lengths')

def main(args=None):
    rclpy.init(args=args)
    validator = SimulationValidator()
    rclpy.spin(validator)
    validator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Run Validation
```bash
cd ~/humanoid_ws
colcon build --packages-select simulation_validation
source install/setup.bash

# Run the validator
ros2 run simulation_validation validator
```

### Expected Outcome
The validation node should compare joint positions between simulation environments and report on their consistency.

## Advanced Exercise: Physics Parameter Tuning

### Objective
Tune simulation physics parameters to match real robot behavior.

### Steps

#### Step 1: Create Parameter Tuning Node
Create `simulation_validation/simulation_validation/parameter_tuner.py`:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import yaml

class ParameterTuner(Node):
    def __init__(self):
        super().__init__('parameter_tuner')

        # Publisher for physics parameters
        self.param_publisher = self.create_publisher(
            Float64,
            'physics_parameters',
            10)

        # Timer for parameter updates
        self.timer = self.create_timer(5.0, self.tune_parameters)

        # Initial physics parameters
        self.parameters = {
            'gravity': 9.81,
            'friction': 0.5,
            'restitution': 0.1,
            'max_step_size': 0.001
        }

        self.iteration = 0

    def tune_parameters(self):
        # Simple parameter tuning based on validation feedback
        # In practice, this would use more sophisticated optimization

        if self.iteration % 2 == 0:
            # Try increasing friction
            self.parameters['friction'] += 0.01
        else:
            # Try decreasing restitution
            self.parameters['restitution'] -= 0.01

        self.parameters['restitution'] = max(0.0, self.parameters['restitution'])

        self.get_logger().info(f'Updated parameters: {self.parameters}')

        # Publish updated parameters
        param_msg = Float64()
        param_msg.data = self.parameters['friction']  # Example: publish friction value
        self.param_publisher.publish(param_msg)

        self.iteration += 1

def main(args=None):
    rclpy.init(args=args)
    tuner = ParameterTuner()
    rclpy.spin(tuner)
    tuner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 2: Implement Parameter Updates in Simulation
Configure your simulation environments to accept physics parameter updates and adjust accordingly.

### Expected Outcome
The simulation should adapt its physics parameters based on validation feedback to better match real-world behavior.

## Troubleshooting and Best Practices

### Common Issues
1. **Model Instability**: Check mass properties and joint limits
2. **Communication Failures**: Verify network settings and firewalls
3. **Performance Issues**: Optimize collision meshes and physics settings
4. **Synchronization Problems**: Ensure consistent time bases

### Best Practices
1. Start with simple models and add complexity gradually
2. Validate each component before integration
3. Use version control for simulation assets
4. Document model assumptions and limitations
5. Regularly compare simulation to real robot behavior

## Summary

These exercises have covered fundamental aspects of digital twin implementation for humanoid robotics:
- Creating basic humanoid robot models for simulation
- Adding ROS 2 control interfaces
- Setting up Unity for high-fidelity visualization
- Validating simulation accuracy
- Tuning physics parameters for realism

Complete these exercises to build a solid foundation in simulation for humanoid robotics applications.
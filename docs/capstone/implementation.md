---
sidebar_label: 'Capstone Implementation Guide'
sidebar_position: 2
---

# Capstone Implementation Guide: Building Your Autonomous Humanoid Robot

## Introduction

This implementation guide provides a step-by-step approach to building your autonomous humanoid robot for the capstone project. It covers the integration of all four modules (ROS 2, Digital Twins, Isaac AI, and Vision-Language-Action) into a unified system that demonstrates the concepts learned throughout the course.

### Project Structure Overview

The capstone project will be organized as follows:

```
capstone_project/
├── src/
│   ├── robot_control/          # Low-level control systems
│   ├── perception/            # Vision and sensor processing
│   ├── decision_making/       # AI and decision systems
│   ├── vla_system/          # Vision-Language-Action integration
│   └── system_integration/  # Main integration and coordination
├── config/
│   ├── robot_parameters.yaml
│   ├── perception_config.yaml
│   └── system_config.yaml
├── launch/
│   ├── complete_system.launch.py
│   ├── perception_system.launch.py
│   └── control_system.launch.py
├── test/
│   ├── integration_tests.py
│   └── performance_tests.py
└── docs/
    └── project_documentation.md
```

## Phase 1: System Architecture and Design

### 1.1 System Architecture Design

#### Creating the System Architecture

```bash
# Create the project workspace
mkdir -p ~/capstone_ws/src
cd ~/capstone_ws/src

# Create the main capstone package
ros2 pkg create --build-type ament_python capstone_system --dependencies rclpy std_msgs sensor_msgs geometry_msgs cv_bridge
```

#### System Architecture Document

Create `~/capstone_ws/src/capstone_system/docs/architecture.md`:

```markdown
# Capstone System Architecture

## Overview
The capstone system follows a modular architecture with clear interfaces between components:

- **Control Layer**: Low-level hardware control and basic movement
- **Perception Layer**: Vision, sensor processing, and environment understanding
- **Decision Layer**: AI-powered decision making and planning
- **Interaction Layer**: Human-robot interaction and communication
- **Integration Layer**: System coordination and state management

## Component Interfaces
- All components communicate via ROS 2 topics, services, and actions
- Standard message types are used where possible
- Each component maintains its own state and configuration
- Error handling and recovery mechanisms are implemented at each level
```

### 1.2 Core System Node Implementation

#### Main Integration Node

Create `~/capstone_ws/src/capstone_system/capstone_system/main_integration_node.py`:

```python
#!/usr/bin/env python3
"""
Main integration node for the capstone humanoid robot system.
Coordinates all subsystems and manages overall robot state.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState, Image, Imu
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import threading
import time
from enum import Enum
from typing import Dict, Any, Optional

class RobotState(Enum):
    """Enumeration of robot states"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING_TASK = "executing_task"
    EMERGENCY_STOP = "emergency_stop"
    ERROR = "error"

class CapstoneIntegrationNode(Node):
    def __init__(self):
        super().__init__('capstone_integration_node')

        # Initialize system state
        self.current_state = RobotState.INITIALIZING
        self.previous_state = RobotState.INITIALIZING
        self.system_ready = False
        self.emergency_stop_active = False

        # Robot state tracking
        self.joint_states = JointState()
        self.imu_data = Imu()
        self.odometry_data = Odometry()

        # Create QoS profile for reliable communication
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Subscribers for system status
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, qos_profile
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, qos_profile
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile
        )

        # Subscribers for commands and control
        self.command_sub = self.create_subscription(
            String, '/capstone/commands', self.command_callback, qos_profile
        )
        self.emergency_stop_sub = self.create_subscription(
            Bool, '/emergency_stop', self.emergency_stop_callback, qos_profile
        )

        # Publishers for system status
        self.state_pub = self.create_publisher(String, '/capstone/state', qos_profile)
        self.status_pub = self.create_publisher(String, '/capstone/status', qos_profile)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile)

        # Initialize subsystems
        self.initialize_subsystems()

        # Create timer for state management
        self.state_timer = self.create_timer(0.1, self.state_management_callback)

        # Create timer for system monitoring
        self.monitor_timer = self.create_timer(1.0, self.system_monitor_callback)

        self.get_logger().info('Capstone Integration Node initialized')
        self.transition_to_state(RobotState.READY)

    def initialize_subsystems(self):
        """Initialize all subsystems"""
        self.get_logger().info('Initializing subsystems...')

        # Initialize control subsystem
        self.initialize_control_subsystem()

        # Initialize perception subsystem
        self.initialize_perception_subsystem()

        # Initialize decision subsystem
        self.initialize_decision_subsystem()

        # Initialize interaction subsystem
        self.initialize_interaction_subsystem()

        self.get_logger().info('All subsystems initialized')

    def initialize_control_subsystem(self):
        """Initialize the control subsystem"""
        # This would typically launch or connect to control nodes
        self.get_logger().info('Control subsystem initialized')

    def initialize_perception_subsystem(self):
        """Initialize the perception subsystem"""
        # This would typically launch or connect to perception nodes
        self.get_logger().info('Perception subsystem initialized')

    def initialize_decision_subsystem(self):
        """Initialize the decision subsystem"""
        # This would typically launch or connect to AI nodes
        self.get_logger().info('Decision subsystem initialized')

    def initialize_interaction_subsystem(self):
        """Initialize the interaction subsystem"""
        # This would typically launch or connect to VLA nodes
        self.get_logger().info('Interaction subsystem initialized')

    def joint_state_callback(self, msg: JointState):
        """Update joint state information"""
        self.joint_states = msg

    def imu_callback(self, msg: Imu):
        """Update IMU data"""
        self.imu_data = msg

    def odom_callback(self, msg: Odometry):
        """Update odometry data"""
        self.odometry_data = msg

    def command_callback(self, msg: String):
        """Process system commands"""
        command = msg.data.lower()
        self.get_logger().info(f'Received command: {command}')

        if command == 'emergency_stop':
            self.activate_emergency_stop()
        elif command == 'reset':
            self.reset_system()
        elif command == 'test':
            self.run_system_test()
        else:
            self.get_logger().warn(f'Unknown command: {command}')

    def emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop signals"""
        if msg.data:
            self.activate_emergency_stop()
        else:
            self.deactivate_emergency_stop()

    def activate_emergency_stop(self):
        """Activate emergency stop"""
        self.emergency_stop_active = True
        self.transition_to_state(RobotState.EMERGENCY_STOP)

        # Stop all movement
        self.stop_robot()

        self.get_logger().warn('EMERGENCY STOP ACTIVATED')

    def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        self.emergency_stop_active = False
        self.transition_to_state(RobotState.READY)
        self.get_logger().info('Emergency stop deactivated')

    def reset_system(self):
        """Reset the system to initial state"""
        self.get_logger().info('Resetting system...')

        # Stop all subsystems
        self.stop_robot()

        # Reset state
        self.emergency_stop_active = False
        self.transition_to_state(RobotState.READY)

        self.get_logger().info('System reset complete')

    def run_system_test(self):
        """Run comprehensive system test"""
        self.get_logger().info('Running system test...')

        # Test joint states
        if len(self.joint_states.position) > 0:
            self.get_logger().info('Joint states: OK')
        else:
            self.get_logger().warn('No joint state data received')

        # Test IMU
        if self.imu_data.orientation.w != 0:
            self.get_logger().info('IMU data: OK')
        else:
            self.get_logger().warn('No IMU data received')

        # Test odometry
        if self.odometry_data.pose.pose.position.x != 0:
            self.get_logger().info('Odometry: OK')
        else:
            self.get_logger().warn('No odometry data received')

        self.get_logger().info('System test complete')

    def state_management_callback(self):
        """Manage robot state transitions"""
        # Check for emergency conditions
        if self.emergency_stop_active and self.current_state != RobotState.EMERGENCY_STOP:
            self.transition_to_state(RobotState.EMERGENCY_STOP)
            return

        # Normal state management
        if not self.emergency_stop_active and self.current_state == RobotState.EMERGENCY_STOP:
            self.transition_to_state(RobotState.READY)

        # Publish current state
        state_msg = String()
        state_msg.data = self.current_state.value
        self.state_pub.publish(state_msg)

    def system_monitor_callback(self):
        """Monitor system health and performance"""
        # Check system resources
        # This would include CPU, memory, and communication monitoring

        # Log system status
        status_msg = String()
        status_msg.data = f"State: {self.current_state.value}, Joints: {len(self.joint_states.name)}, Emergency: {self.emergency_stop_active}"
        self.status_pub.publish(status_msg)

    def transition_to_state(self, new_state: RobotState):
        """Safely transition to a new state"""
        self.previous_state = self.current_state
        self.current_state = new_state

        self.get_logger().info(f'State transition: {self.previous_state.value} -> {self.current_state.value}')

        # Execute state-specific actions
        if new_state == RobotState.READY:
            self.on_ready_state()
        elif new_state == RobotState.EMERGENCY_STOP:
            self.on_emergency_stop_state()
        elif new_state == RobotState.ERROR:
            self.on_error_state()

    def on_ready_state(self):
        """Actions to perform when entering READY state"""
        # Enable systems that were disabled
        pass

    def on_emergency_stop_state(self):
        """Actions to perform when entering EMERGENCY_STOP state"""
        # Stop all robot movement
        self.stop_robot()

    def on_error_state(self):
        """Actions to perform when entering ERROR state"""
        # Stop all non-critical systems
        self.stop_robot()

    def stop_robot(self):
        """Stop all robot movement"""
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.linear.y = 0.0
        stop_cmd.linear.z = 0.0
        stop_cmd.angular.x = 0.0
        stop_cmd.angular.y = 0.0
        stop_cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(stop_cmd)

def main(args=None):
    rclpy.init(args=args)
    integration_node = CapstoneIntegrationNode()

    try:
        rclpy.spin(integration_node)
    except KeyboardInterrupt:
        integration_node.get_logger().info('Shutting down capstone integration node')
    finally:
        integration_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 1.3 Configuration Files

#### Robot Parameters Configuration

Create `~/capstone_ws/src/capstone_system/config/robot_parameters.yaml`:

```yaml
capstone_integration_node:
  ros__parameters:
    # Robot physical parameters
    robot_radius: 0.5  # meters
    max_linear_velocity: 1.0  # m/s
    max_angular_velocity: 1.0  # rad/s
    acceleration_limit: 2.0  # m/s^2

    # Safety parameters
    emergency_stop_distance: 0.5  # meters
    collision_threshold: 0.3  # meters
    max_tilt_angle: 15.0  # degrees

    # Control parameters
    control_loop_rate: 50  # Hz
    perception_loop_rate: 10  # Hz
    decision_loop_rate: 5  # Hz

    # System parameters
    system_timeout: 30.0  # seconds
    heartbeat_interval: 1.0  # seconds
    max_joint_error: 0.1  # radians
```

#### Launch File

Create `~/capstone_ws/src/capstone_system/launch/capstone_system.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    # Main integration node
    integration_node = Node(
        package='capstone_system',
        executable='main_integration_node',
        name='capstone_integration_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('capstone_system'),
                'config', 'robot_parameters.yaml'
            ])
        ],
        output='screen'
    )

    # Perception system node (placeholder - will be implemented later)
    perception_node = Node(
        package='capstone_system',
        executable='perception_node',
        name='capstone_perception_node',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # Decision system node (placeholder - will be implemented later)
    decision_node = Node(
        package='capstone_system',
        executable='decision_node',
        name='capstone_decision_node',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time,
        integration_node,
        perception_node,
        decision_node
    ])
```

## Phase 2: Control System Implementation

### 2.1 Low-Level Control Implementation

Create `~/capstone_ws/src/capstone_system/capstone_system/control_node.py`:

```python
#!/usr/bin/env python3
"""
Low-level control node for humanoid robot.
Handles joint control, balance, and basic movement.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float64MultiArray, Bool
from builtin_interfaces.msg import Duration
import numpy as np
import time
from collections import deque

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        # Control parameters
        self.control_rate = 50  # Hz
        self.dt = 1.0 / self.control_rate

        # Robot state
        self.current_joint_positions = {}
        self.current_joint_velocities = {}
        self.imu_orientation = None
        self.imu_angular_velocity = None
        self.imu_linear_acceleration = None

        # Balance control parameters
        self.balance_kp = 10.0  # Proportional gain
        self.balance_kd = 1.0   # Derivative gain
        self.target_orientation = [0.0, 0.0, 0.0, 1.0]  # Quaternion [x,y,z,w]

        # Command tracking
        self.last_cmd_time = time.time()
        self.cmd_timeout = 1.0  # seconds

        # Create QoS profile
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, qos_profile
        )

        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, qos_profile
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, qos_profile
        )

        self.emergency_stop_sub = self.create_subscription(
            Bool, '/emergency_stop', self.emergency_stop_callback, qos_profile
        )

        # Publishers
        self.joint_command_pub = self.create_publisher(
            JointState, '/joint_commands', qos_profile
        )

        self.balance_status_pub = self.create_publisher(
            Float64MultiArray, '/balance_status', qos_profile
        )

        # Control timer
        self.control_timer = self.create_timer(
            self.dt, self.control_loop, clock=self.get_clock()
        )

        # Emergency stop flag
        self.emergency_stop_active = False

        self.get_logger().info('Control node initialized')

    def joint_state_callback(self, msg: JointState):
        """Update current joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]

    def imu_callback(self, msg: Imu):
        """Update IMU data"""
        self.imu_orientation = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]

        self.imu_angular_velocity = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]

        self.imu_linear_acceleration = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ]

    def cmd_vel_callback(self, msg: Twist):
        """Process velocity commands"""
        self.last_cmd_time = time.time()

        # Store command for control loop
        self.desired_linear_vel = msg.linear
        self.desired_angular_vel = msg.angular

    def emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop"""
        self.emergency_stop_active = msg.data

    def control_loop(self):
        """Main control loop"""
        if self.emergency_stop_active:
            self.stop_robot()
            return

        # Check for command timeout
        if time.time() - self.last_cmd_time > self.cmd_timeout:
            self.stop_robot()
            return

        # Perform balance control
        balance_commands = self.balance_control()

        # Perform locomotion control
        locomotion_commands = self.locomotion_control()

        # Combine commands
        final_commands = self.combine_commands(balance_commands, locomotion_commands)

        # Publish commands
        self.publish_joint_commands(final_commands)

        # Publish balance status
        self.publish_balance_status()

    def balance_control(self):
        """Perform balance control using IMU feedback"""
        if self.imu_orientation is None:
            return {}

        # Convert quaternion to roll/pitch angles
        roll, pitch = self.quaternion_to_roll_pitch(self.imu_orientation)

        # Calculate balance error
        target_roll, target_pitch = 0.0, 0.0
        roll_error = target_roll - roll
        pitch_error = target_pitch - pitch

        # Calculate angular velocity error
        if self.imu_angular_velocity:
            roll_vel_error = 0.0 - self.imu_angular_velocity[0]
            pitch_vel_error = 0.0 - self.imu_angular_velocity[1]
        else:
            roll_vel_error = 0.0
            pitch_vel_error = 0.0

        # PD control for balance
        roll_correction = self.balance_kp * roll_error + self.balance_kd * roll_vel_error
        pitch_correction = self.balance_kp * pitch_error + self.balance_kd * pitch_vel_error

        # Generate balance joint commands
        balance_commands = {
            'left_hip_roll': roll_correction,
            'right_hip_roll': -roll_correction,
            'left_hip_pitch': pitch_correction,
            'right_hip_pitch': pitch_correction,
            'torso_pitch': -pitch_correction
        }

        return balance_commands

    def locomotion_control(self):
        """Perform locomotion control based on velocity commands"""
        if not hasattr(self, 'desired_linear_vel'):
            return {}

        # Simple locomotion model - in reality this would be more complex
        commands = {}

        # Map linear velocity to leg movements
        if hasattr(self, 'desired_linear_vel'):
            linear_x = self.desired_linear_vel.x
            linear_y = self.desired_linear_vel.y
            angular_z = self.desired_linear_vel.z

            # Generate simple walking pattern commands
            commands.update({
                'left_hip_pitch': linear_x * 0.1,
                'right_hip_pitch': linear_x * 0.1,
                'left_knee': linear_x * 0.05,
                'right_knee': linear_x * 0.05,
                'left_ankle': -linear_x * 0.02,
                'right_ankle': -linear_x * 0.02
            })

        return commands

    def combine_commands(self, balance_commands, locomotion_commands):
        """Combine balance and locomotion commands"""
        final_commands = {}

        # Add balance commands
        for joint, command in balance_commands.items():
            final_commands[joint] = command

        # Add locomotion commands with appropriate blending
        for joint, command in locomotion_commands.items():
            if joint in final_commands:
                # Blend commands (this is a simple example)
                final_commands[joint] += command
            else:
                final_commands[joint] = command

        return final_commands

    def publish_joint_commands(self, commands):
        """Publish joint commands to robot"""
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.header.frame_id = 'base_link'

        for joint_name, command_value in commands.items():
            joint_cmd.name.append(joint_name)
            joint_cmd.position.append(command_value)

        self.joint_command_pub.publish(joint_cmd)

    def publish_balance_status(self):
        """Publish balance status information"""
        if self.imu_orientation:
            roll, pitch = self.quaternion_to_roll_pitch(self.imu_orientation)

            status = Float64MultiArray()
            status.data = [roll, pitch, 0.0, 0.0]  # roll, pitch, balance_error, stability
            self.balance_status_pub.publish(status)

    def quaternion_to_roll_pitch(self, quat):
        """Convert quaternion to roll and pitch angles"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        return roll, pitch

    def stop_robot(self):
        """Send zero commands to stop robot"""
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.header.frame_id = 'base_link'

        # Add all known joints with zero position
        for joint_name in self.current_joint_positions.keys():
            joint_cmd.name.append(joint_name)
            joint_cmd.position.append(0.0)

        self.joint_command_pub.publish(joint_cmd)

def main(args=None):
    rclpy.init(args=args)
    control_node = ControlNode()

    try:
        rclpy.spin(control_node)
    except KeyboardInterrupt:
        control_node.get_logger().info('Shutting down control node')
    finally:
        control_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Phase 3: Perception System Implementation

### 3.1 Perception Node Implementation

Create `~/capstone_ws/src/capstone_system/capstone_system/perception_node.py`:

```python
#!/usr/bin/env python3
"""
Perception system for the capstone humanoid robot.
Handles vision processing, object detection, and environment understanding.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import PointStamped, PoseArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
import threading
import time

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Image processing parameters
        self.image_queue_size = 10
        self.image_buffer = deque(maxlen=self.image_queue_size)
        self.latest_image = None
        self.image_lock = threading.Lock()

        # Object detection parameters
        self.object_classes = [
            'person', 'cup', 'bottle', 'chair', 'table',
            'phone', 'keys', 'book', 'box', 'door'
        ]

        # Processing parameters
        self.processing_rate = 10  # Hz
        self.min_object_size = 30  # pixels
        self.confidence_threshold = 0.5

        # Create QoS profile
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, qos_profile
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/rgb/camera_info', self.camera_info_callback, qos_profile
        )

        # Publishers
        self.object_pub = self.create_publisher(
            String, '/detected_objects', qos_profile
        )

        self.object_poses_pub = self.create_publisher(
            PoseArray, '/object_poses', qos_profile
        )

        self.processed_image_pub = self.create_publisher(
            Image, '/perception/processed_image', qos_profile
        )

        self.feature_pub = self.create_publisher(
            Float32MultiArray, '/perception/features', qos_profile
        )

        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None

        # Processing timer
        self.processing_timer = self.create_timer(
            1.0 / self.processing_rate, self.processing_callback
        )

        self.get_logger().info('Perception node initialized')

    def image_callback(self, msg: Image):
        """Process incoming images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Store image with thread safety
            with self.image_lock:
                self.latest_image = cv_image.copy()
                self.image_buffer.append(cv_image)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def camera_info_callback(self, msg: CameraInfo):
        """Update camera parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def processing_callback(self):
        """Main processing loop"""
        if self.latest_image is None:
            return

        with self.image_lock:
            current_image = self.latest_image.copy()

        # Perform object detection
        detected_objects = self.detect_objects(current_image)

        # Perform feature extraction
        features = self.extract_features(current_image)

        # Publish results
        self.publish_objects(detected_objects)
        self.publish_features(features)

        # Publish processed image if needed
        if self.processed_image_pub.get_subscription_count() > 0:
            processed_img = self.draw_detections(current_image, detected_objects)
            ros_img = self.bridge.cv2_to_imgmsg(processed_img, "bgr8")
            self.processed_image_pub.publish(ros_img)

    def detect_objects(self, image):
        """Detect objects in the image"""
        height, width = image.shape[:2]
        detected_objects = []

        # This is a simplified object detector for demonstration
        # In practice, you would use a deep learning model like YOLO or similar

        # Generate some example detections
        for i in range(3):  # Create 3 random detections
            # Random bounding box
            x = np.random.randint(0, width - 100)
            y = np.random.randint(0, height - 100)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)

            # Random class
            obj_class = np.random.choice(self.object_classes)
            confidence = np.random.uniform(0.6, 0.95)

            # Only include if confidence is above threshold
            if confidence > self.confidence_threshold:
                detected_objects.append({
                    'class': obj_class,
                    'bbox': [x, y, x + w, y + h],
                    'confidence': confidence,
                    'center': (x + w//2, y + h//2)
                })

        # More realistic detection using OpenCV
        # Detect people using HOG descriptor
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Resize image for detection (HOG works better on smaller images)
        small_img = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        (rects, weights) = hog.detectMultiScale(small_img, winStride=(8, 8), padding=(32, 32), scale=1.05)

        for (x, y, w, h), weight in zip(rects, weights):
            if weight > 0.5:  # Confidence threshold
                # Scale back to original image size
                x, y, w, h = int(x*2), int(y*2), int(w*2), int(h*2)

                detected_objects.append({
                    'class': 'person',
                    'bbox': [x, y, x + w, y + h],
                    'confidence': float(weight),
                    'center': (x + w//2, y + h//2)
                })

        return detected_objects

    def extract_features(self, image):
        """Extract visual features from image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract SIFT features
        sift = cv2.SIFT_create(nfeatures=100)
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        # Calculate image statistics
        mean_color = np.mean(image, axis=(0, 1))
        std_color = np.std(image, axis=(0, 1))

        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Combine features
        features = np.concatenate([
            mean_color,
            std_color,
            [edge_density],
            [len(keypoints) if keypoints is not None else 0]
        ])

        return features

    def draw_detections(self, image, detections):
        """Draw detection results on image"""
        output_image = image.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']

            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(output_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw center
            center_x, center_y = detection['center']
            cv2.circle(output_image, (center_x, center_y), 3, (0, 0, 255), -1)

        return output_image

    def publish_objects(self, detections):
        """Publish detected objects"""
        if not detections:
            return

        # Create string message with detection info
        detection_strs = []
        for detection in detections:
            detection_str = f"{detection['class']}:{detection['confidence']:.2f}"
            detection_strs.append(detection_str)

        detection_msg = String()
        detection_msg.data = "|".join(detection_strs)
        self.object_pub.publish(detection_msg)

    def publish_features(self, features):
        """Publish extracted features"""
        features_msg = Float32MultiArray()
        features_msg.data = features.tolist()
        self.feature_pub.publish(features_msg)

def main(args=None):
    rclpy.init(args=args)
    perception_node = PerceptionNode()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        perception_node.get_logger().info('Shutting down perception node')
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Phase 4: Decision System Implementation

### 4.1 Decision Node Implementation

Create `~/capstone_ws/src/capstone_system/capstone_system/decision_node.py`:

```python
#!/usr/bin/env python3
"""
Decision-making system for the capstone humanoid robot.
Implements AI-powered planning, reasoning, and task execution.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Bool, Float32MultiArray
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import JointState
from action_msgs.msg import GoalStatus
import time
import random
from enum import Enum
from typing import Dict, List, Any, Optional
import threading

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DecisionNode(Node):
    def __init__(self):
        super().__init__('decision_node')

        # Task management
        self.current_task = None
        self.task_queue = []
        self.task_status = TaskStatus.PENDING
        self.task_lock = threading.Lock()

        # Robot state tracking
        self.robot_pose = Pose()
        self.joint_states = JointState()
        self.detected_objects = []
        self.environment_map = {}

        # Decision parameters
        self.decision_rate = 5  # Hz
        self.planning_horizon = 10.0  # seconds
        self.safety_margin = 0.5  # meters

        # Create QoS profile
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Subscribers
        self.task_sub = self.create_subscription(
            String, '/capstone/tasks', self.task_callback, qos_profile
        )

        self.object_sub = self.create_subscription(
            String, '/detected_objects', self.object_callback, qos_profile
        )

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, qos_profile
        )

        self.robot_pose_sub = self.create_subscription(
            String, '/robot_pose', self.pose_callback, qos_profile
        )

        # Publishers
        self.action_pub = self.create_publisher(
            String, '/robot_actions', qos_profile
        )

        self.task_status_pub = self.create_publisher(
            String, '/task_status', qos_profile
        )

        self.navigation_goal_pub = self.create_publisher(
            String, '/navigation/goal', qos_profile
        )

        # Decision timer
        self.decision_timer = self.create_timer(
            1.0 / self.decision_rate, self.decision_callback
        )

        self.get_logger().info('Decision node initialized')

    def task_callback(self, msg: String):
        """Process incoming tasks"""
        task_data = msg.data

        with self.task_lock:
            # Parse task from string
            task = self.parse_task(task_data)
            if task:
                self.task_queue.append(task)
                self.get_logger().info(f'Added task to queue: {task["type"]}')

    def object_callback(self, msg: String):
        """Update detected objects"""
        if msg.data:
            # Parse object detections from string
            objects = self.parse_objects(msg.data)
            self.detected_objects = objects

    def joint_state_callback(self, msg: JointState):
        """Update joint states"""
        self.joint_states = msg

    def pose_callback(self, msg: String):
        """Update robot pose"""
        # Parse pose from string representation
        pose_data = msg.data.split(',')
        if len(pose_data) >= 3:
            try:
                self.robot_pose.position.x = float(pose_data[0])
                self.robot_pose.position.y = float(pose_data[1])
                self.robot_pose.position.z = float(pose_data[2])
            except ValueError:
                pass

    def decision_callback(self):
        """Main decision-making loop"""
        with self.task_lock:
            # Check if we have a current task
            if self.current_task is None and self.task_queue:
                self.current_task = self.task_queue.pop(0)
                self.task_status = TaskStatus.RUNNING
                self.get_logger().info(f'Starting task: {self.current_task["type"]}')

            # Execute current task if we have one
            if self.current_task and self.task_status == TaskStatus.RUNNING:
                task_result = self.execute_task(self.current_task)

                if task_result == 'success':
                    self.task_status = TaskStatus.SUCCESS
                    self.publish_task_status('success')
                    self.current_task = None
                elif task_result == 'failed':
                    self.task_status = TaskStatus.FAILED
                    self.publish_task_status('failed')
                    self.current_task = None
                elif task_result == 'running':
                    self.publish_task_status('running')

        # Perform environmental reasoning
        self.environmental_reasoning()

    def parse_task(self, task_str: str) -> Optional[Dict[str, Any]]:
        """Parse task string into structured task"""
        try:
            # Expected format: "task_type:parameter1=value1,parameter2=value2"
            parts = task_str.split(':', 1)
            if len(parts) != 2:
                return None

            task_type = parts[0]
            params_str = parts[1]

            # Parse parameters
            params = {}
            if params_str:
                param_pairs = params_str.split(',')
                for pair in param_pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        # Try to convert to appropriate type
                        try:
                            # Try integer first
                            params[key] = int(value)
                        except ValueError:
                            try:
                                # Try float
                                params[key] = float(value)
                            except ValueError:
                                # Keep as string
                                params[key] = value

            return {
                'type': task_type,
                'parameters': params,
                'created_time': time.time()
            }
        except Exception as e:
            self.get_logger().error(f'Error parsing task: {e}')
            return None

    def parse_objects(self, objects_str: str) -> List[Dict[str, Any]]:
        """Parse detected objects string"""
        objects = []
        if objects_str:
            obj_pairs = objects_str.split('|')
            for pair in obj_pairs:
                if ':' in pair:
                    class_conf = pair.split(':')
                    if len(class_conf) == 2:
                        objects.append({
                            'class': class_conf[0],
                            'confidence': float(class_conf[1])
                        })
        return objects

    def execute_task(self, task: Dict[str, Any]) -> str:
        """Execute a specific task"""
        task_type = task['type']
        params = task['parameters']

        if task_type == 'find_object':
            return self.execute_find_object_task(params)
        elif task_type == 'navigate_to':
            return self.execute_navigate_task(params)
        elif task_type == 'grasp_object':
            return self.execute_grasp_task(params)
        elif task_type == 'bring_to_location':
            return self.execute_bring_task(params)
        elif task_type == 'follow_person':
            return self.execute_follow_task(params)
        else:
            self.get_logger().warn(f'Unknown task type: {task_type}')
            return 'failed'

    def execute_find_object_task(self, params: Dict[str, Any]) -> str:
        """Execute find object task"""
        target_object = params.get('object', 'object')

        # Check if target object is detected
        for obj in self.detected_objects:
            if target_object in obj['class']:
                self.get_logger().info(f'Found {target_object} with confidence {obj["confidence"]:.2f}')

                # Publish action to approach object
                action_msg = String()
                action_msg.data = f'approach_object:{obj["class"]}'
                self.action_pub.publish(action_msg)

                return 'success'

        self.get_logger().info(f'Still searching for {target_object}')
        return 'running'

    def execute_navigate_task(self, params: Dict[str, Any]) -> str:
        """Execute navigation task"""
        location = params.get('location', 'default')

        # Determine navigation coordinates based on location
        nav_coords = self.get_location_coordinates(location)
        if nav_coords:
            # Publish navigation goal
            nav_msg = String()
            nav_msg.data = f'x={nav_coords[0]},y={nav_coords[1]}'
            self.navigation_goal_pub.publish(nav_msg)

            self.get_logger().info(f'Navigating to {location} at {nav_coords}')
            return 'running'  # Navigation is ongoing
        else:
            self.get_logger().warn(f'Unknown location: {location}')
            return 'failed'

    def execute_grasp_task(self, params: Dict[str, Any]) -> str:
        """Execute grasp task"""
        target_object = params.get('object', 'object')

        # Check if target object is within reach
        for obj in self.detected_objects:
            if target_object in obj['class'] and obj['confidence'] > 0.7:
                # Publish grasp command
                grasp_msg = String()
                grasp_msg.data = f'grasp:{obj["class"]}'
                self.action_pub.publish(grasp_msg)

                self.get_logger().info(f'Attempting to grasp {obj["class"]}')
                return 'running'  # Grasping is ongoing

        self.get_logger().info(f'{target_object} not ready for grasping')
        return 'running'

    def execute_bring_task(self, params: Dict[str, Any]) -> str:
        """Execute bring object task"""
        target_object = params.get('object', 'object')
        destination = params.get('destination', 'user')

        # This would involve multiple subtasks: find, grasp, navigate, place
        # For simplicity, we'll just navigate to destination
        nav_msg = String()
        nav_msg.data = f'navigate_to:{destination}'
        self.navigation_goal_pub.publish(nav_msg)

        self.get_logger().info(f'Bringing {target_object} to {destination}')
        return 'running'

    def execute_follow_task(self, params: Dict[str, Any]) -> str:
        """Execute follow person task"""
        # Publish follow command
        follow_msg = String()
        follow_msg.data = 'follow_person:enabled'
        self.action_pub.publish(follow_msg)

        self.get_logger().info('Following person')
        return 'running'

    def get_location_coordinates(self, location_name: str) -> Optional[tuple]:
        """Get coordinates for predefined locations"""
        locations = {
            'kitchen': (2.0, 0.0),
            'living_room': (0.0, 2.0),
            'bedroom': (-2.0, 0.0),
            'office': (0.0, -2.0),
            'charging_station': (3.0, 3.0),
            'default': (0.0, 0.0)
        }

        return locations.get(location_name)

    def environmental_reasoning(self):
        """Perform environmental reasoning and planning"""
        # Analyze detected objects
        person_detected = any('person' in obj['class'] for obj in self.detected_objects)

        if person_detected:
            # Adjust behavior when person is detected
            self.react_to_person()

    def react_to_person(self):
        """React appropriately when person is detected"""
        # This could trigger social behaviors
        self.get_logger().info('Person detected - ready to interact')

    def publish_task_status(self, status: str):
        """Publish current task status"""
        status_msg = String()
        status_msg.data = f"{status}:{self.current_task['type'] if self.current_task else 'none'}"
        self.task_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    decision_node = DecisionNode()

    try:
        rclpy.spin(decision_node)
    except KeyboardInterrupt:
        decision_node.get_logger().info('Shutting down decision node')
    finally:
        decision_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Phase 5: System Integration and Testing

### 5.1 Update setup.py

Update `~/capstone_ws/src/capstone_system/setup.py`:

```python
from setuptools import find_packages, setup

package_name = 'capstone_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/robot_parameters.yaml']),
        ('share/' + package_name + '/launch', [
            'launch/capstone_system.launch.py'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Capstone System for Autonomous Humanoid Robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main_integration_node = capstone_system.main_integration_node:main',
            'control_node = capstone_system.control_node:main',
            'perception_node = capstone_system.perception_node:main',
            'decision_node = capstone_system.decision_node:main',
        ],
    },
)
```

### 5.2 Build and Test the System

```bash
cd ~/capstone_ws
colcon build --packages-select capstone_system
source install/setup.bash

# Test the complete system
ros2 launch capstone_system capstone_system.launch.py
```

## Phase 6: Documentation and Evaluation

### 6.1 Create Project Documentation

Create `~/capstone_ws/src/capstone_system/docs/project_documentation.md`:

```markdown
# Capstone Project Documentation

## System Architecture

The capstone system follows a modular architecture with clear separation of concerns:

- **Integration Layer**: Coordinates all subsystems and manages overall state
- **Control Layer**: Handles low-level robot control and balance
- **Perception Layer**: Processes visual and sensor data
- **Decision Layer**: Implements AI-powered planning and reasoning

## Key Components

### Main Integration Node
- Manages system state and safety protocols
- Coordinates communication between subsystems
- Implements emergency stop functionality

### Control Node
- Handles joint control and balance maintenance
- Implements locomotion control algorithms
- Processes velocity commands for navigation

### Perception Node
- Processes camera images for object detection
- Extracts visual features for environment understanding
- Publishes detected objects and their properties

### Decision Node
- Processes high-level tasks and goals
- Implements task planning and execution
- Manages robot behavior and interaction

## Performance Results

### Real-time Performance
- Control loop: 50Hz (20ms period)
- Perception loop: 10Hz (100ms period)
- Decision loop: 5Hz (200ms period)

### Resource Usage
- CPU: &lt;60% average utilization
- Memory: &lt;2GB RAM usage
- Communication: &lt;100ms message latency

## Lessons Learned

1. **Modular Design**: Critical for system maintainability and testing
2. **Safety First**: Emergency stop and fail-safe mechanisms are essential
3. **Real-time Constraints**: Careful timing analysis required for robot control
4. **Integration Challenges**: Subsystem communication requires careful design

## Future Improvements

1. Enhanced AI capabilities with deep learning integration
2. Improved navigation with SLAM capabilities
3. Advanced human-robot interaction features
4. Robust error recovery and self-diagnostic capabilities
```

## Conclusion

This implementation guide provides a comprehensive foundation for building an autonomous humanoid robot system that integrates all four modules learned in the course. The modular architecture allows for easy testing and iteration, while the safety-focused design ensures responsible development practices.

The system demonstrates key concepts in:
- ROS 2 communication and architecture
- AI-powered perception and decision-making
- Real-time control systems
- Human-robot interaction

Continue to iterate on this foundation, adding more sophisticated capabilities as you progress through your capstone project. Remember to maintain good documentation, implement comprehensive testing, and prioritize safety in all development decisions.
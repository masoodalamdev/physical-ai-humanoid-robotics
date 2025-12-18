---
sidebar_label: 'ROS 2 Exercises'
sidebar_position: 3
---

# ROS 2 Exercises: Hands-On Learning for Humanoid Robotics

## Exercise 1: Creating Your First Humanoid Robot Node

### Objective
Create a simple ROS 2 node that publishes joint positions for a humanoid robot.

### Prerequisites
- ROS 2 Humble installed
- Basic Python or C++ knowledge
- Completed the setup section

### Steps

#### Step 1: Create a Package
```bash
cd ~/humanoid_ws/src
ros2 pkg create --build-type ament_python joint_publisher --dependencies rclpy std_msgs sensor_msgs
```

#### Step 2: Create the Node
Create `joint_publisher/joint_publisher/joint_publisher_node.py`:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math
import random

class JointPublisher(Node):
    def __init__(self):
        super().__init__('joint_publisher')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

        # Define humanoid joint names
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint',
            'head_joint'
        ]

    def timer_callback(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = []

        # Generate simple oscillating joint positions
        for i, joint_name in enumerate(self.joint_names):
            # Create different oscillation patterns for different joint groups
            if 'hip' in joint_name or 'knee' in joint_name:
                # Leg joints with walking pattern
                position = 0.5 * math.sin(self.i * 0.1 + i)
            elif 'shoulder' in joint_name or 'elbow' in joint_name:
                # Arm joints with reaching pattern
                position = 0.3 * math.sin(self.i * 0.07 + i)
            elif 'head' in joint_name:
                # Head with subtle movement
                position = 0.1 * math.sin(self.i * 0.05)
            else:
                # Other joints
                position = 0.2 * math.sin(self.i * 0.08 + i)

            msg.position.append(position)

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing joint states: {self.i}')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    joint_publisher = JointPublisher()
    rclpy.spin(joint_publisher)
    joint_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Update setup.py
In `joint_publisher/setup.py`, add:
```python
entry_points={
    'console_scripts': [
        'joint_publisher = joint_publisher.joint_publisher_node:main',
    ],
},
```

#### Step 4: Build and Run
```bash
cd ~/humanoid_ws
colcon build --packages-select joint_publisher
source install/setup.bash
ros2 run joint_publisher joint_publisher
```

#### Step 5: Visualize with RViz
In a new terminal:
```bash
source install/setup.bash
rviz2
```
In RViz, add a RobotModel display and set the TF topic to visualize the joint movements.

### Expected Outcome
You should see a humanoid robot model with oscillating joints, demonstrating ROS 2's ability to publish joint state information.

## Exercise 2: Creating a Service Node

### Objective
Create a ROS 2 service that accepts commands to move humanoid robot joints to specific positions.

### Steps

#### Step 1: Define the Service
Create `joint_publisher/srv/JointCommand.srv`:
```
string joint_name
float64 position
---
bool success
string message
```

#### Step 2: Create the Service Server
Create `joint_publisher/joint_publisher/joint_service_server.py`:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from joint_publisher.srv import JointCommand

class JointServiceServer(Node):
    def __init__(self):
        super().__init__('joint_service_server')
        self.srv = self.create_service(JointCommand, 'move_joint', self.move_joint_callback)
        self.get_logger().info('Joint service server ready')

    def move_joint_callback(self, request, response):
        # In a real robot, this would send the command to the hardware
        # Here we just simulate the movement

        self.get_logger().info(f'Received request to move {request.joint_name} to {request.position}')

        # Validate the request
        if request.position < -3.14 or request.position > 3.14:
            response.success = False
            response.message = f'Position {request.position} out of range [-3.14, 3.14]'
        else:
            response.success = True
            response.message = f'Successfully moved {request.joint_name} to {request.position}'

        return response

def main(args=None):
    rclpy.init(args=args)
    joint_service_server = JointServiceServer()
    rclpy.spin(joint_service_server)
    joint_service_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Create the Service Client
Create `joint_publisher/joint_publisher/joint_service_client.py`:
```python
#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from joint_publisher.srv import JointCommand

class JointServiceClient(Node):
    def __init__(self):
        super().__init__('joint_service_client')
        self.cli = self.create_client(JointCommand, 'move_joint')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = JointCommand.Request()

    def send_request(self, joint_name, position):
        self.req.joint_name = joint_name
        self.req.position = position
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main():
    rclpy.init()

    if len(sys.argv) != 3:
        print("Usage: ros2 run joint_publisher joint_service_client <joint_name> <position>")
        return

    joint_name = sys.argv[1]
    position = float(sys.argv[2])

    joint_service_client = JointServiceClient()
    response = joint_service_client.send_request(joint_name, position)

    if response:
        print(f'Result: {response.success}, Message: {response.message}')
    else:
        print('Service call failed')

    joint_service_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 4: Update setup.py
Add the client to entry points:
```python
entry_points={
    'console_scripts': [
        'joint_publisher = joint_publisher.joint_publisher_node:main',
        'joint_service_server = joint_publisher.joint_service_server:main',
        'joint_service_client = joint_publisher.joint_service_client:main',
    ],
},
```

#### Step 5: Build and Test
```bash
cd ~/humanoid_ws
colcon build --packages-select joint_publisher
source install/setup.bash

# Terminal 1: Start the service server
ros2 run joint_publisher joint_service_server

# Terminal 2: Call the service
ros2 run joint_publisher joint_service_client left_shoulder_joint 1.57
```

### Expected Outcome
The service should accept joint movement commands and respond with success/failure status.

## Exercise 3: Action Server for Complex Movements

### Objective
Create an action server that can execute complex movements (like walking) with feedback and the ability to cancel.

### Steps

#### Step 1: Define the Action
Create `joint_publisher/action/Walk.action`:
```
# Goal: Request to walk to a specific location
float64 x
float64 y
float64 theta
---
# Result: Outcome of the walk action
bool success
string message
float64 final_x
float64 final_y
float64 final_theta
---
# Feedback: Current progress of the walk
float64 current_x
float64 current_y
float64 current_theta
float64 distance_remaining
string status
```

#### Step 2: Create the Action Server
Create `joint_publisher/joint_publisher/walk_action_server.py`:
```python
#!/usr/bin/env python3
import time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from joint_publisher.action import Walk

class WalkActionServer(Node):
    def __init__(self):
        super().__init__('walk_action_server')
        self._action_server = ActionServer(
            self,
            Walk,
            'walk_to_pose',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Received walk goal request')

        # Simulate walking to the goal
        feedback_msg = Walk.Feedback()
        feedback_msg.current_x = 0.0
        feedback_msg.current_y = 0.0
        feedback_msg.current_theta = 0.0
        feedback_msg.status = "Walking to destination"

        # Simulate the walk progress
        for i in range(10):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = Walk.Result()
                result.success = False
                result.message = "Goal was canceled"
                self.get_logger().info('Walk goal canceled')
                return result

            # Update feedback
            feedback_msg.current_x = goal_handle.request.x * (i + 1) / 10
            feedback_msg.current_y = goal_handle.request.y * (i + 1) / 10
            feedback_msg.current_theta = goal_handle.request.theta * (i + 1) / 10
            feedback_msg.distance_remaining = abs(goal_handle.request.x - feedback_msg.current_x) + \
                                             abs(goal_handle.request.y - feedback_msg.current_y)
            feedback_msg.status = f"Progress: {i+1}/10"

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {feedback_msg.status}')

            time.sleep(0.5)  # Simulate time to move

        # Complete the action
        goal_handle.succeed()
        result = Walk.Result()
        result.success = True
        result.message = "Successfully reached destination"
        result.final_x = goal_handle.request.x
        result.final_y = goal_handle.request.y
        result.final_theta = goal_handle.request.theta

        self.get_logger().info(f'Walk completed: ({result.final_x}, {result.final_y}, {result.final_theta})')
        return result

def main(args=None):
    rclpy.init(args=args)
    walk_action_server = WalkActionServer()
    rclpy.spin(walk_action_server)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Create the Action Client
Create `joint_publisher/joint_publisher/walk_action_client.py`:
```python
#!/usr/bin/env python3
import sys
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from joint_publisher.action import Walk

class WalkActionClient(Node):
    def __init__(self):
        super().__init__('walk_action_client')
        self._action_client = ActionClient(self, Walk, 'walk_to_pose')

    def send_goal(self, x, y, theta):
        goal_msg = Walk.Goal()
        goal_msg.x = x
        goal_msg.y = y
        goal_msg.theta = theta

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Received feedback: {feedback.status} - Distance: {feedback.distance_remaining:.2f}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.success}, {result.message}')
        rclpy.shutdown()

def main():
    rclpy.init()

    if len(sys.argv) != 4:
        print("Usage: ros2 run joint_publisher walk_action_client <x> <y> <theta>")
        return

    x = float(sys.argv[1])
    y = float(sys.argv[2])
    theta = float(sys.argv[3])

    action_client = WalkActionClient()
    action_client.send_goal(x, y, theta)

    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
```

#### Step 4: Update setup.py
Add the action server and client:
```python
entry_points={
    'console_scripts': [
        'joint_publisher = joint_publisher.joint_publisher_node:main',
        'joint_service_server = joint_publisher.joint_service_server:main',
        'joint_service_client = joint_publisher.joint_service_client:main',
        'walk_action_server = joint_publisher.walk_action_server:main',
        'walk_action_client = joint_publisher.walk_action_client:main',
    ],
},
```

#### Step 5: Build and Test
```bash
cd ~/humanoid_ws
colcon build --packages-select joint_publisher
source install/setup.bash

# Terminal 1: Start the action server
ros2 run joint_publisher walk_action_server

# Terminal 2: Call the action
ros2 run joint_publisher walk_action_client 2.0 1.0 1.57
```

### Expected Outcome
The action server should accept walk commands, provide feedback during execution, and return results when complete.

## Exercise 4: Launch File for Coordinated Operation

### Objective
Create a launch file that starts multiple nodes simultaneously for coordinated humanoid robot operation.

### Steps

#### Step 1: Create the Launch File
Create `joint_publisher/launch/humanoid_robot.launch.py`:
```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        # Joint publisher node
        Node(
            package='joint_publisher',
            executable='joint_publisher',
            name='joint_publisher',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            output='screen'
        ),

        # Joint service server
        Node(
            package='joint_publisher',
            executable='joint_service_server',
            name='joint_service_server',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            output='screen'
        ),

        # Robot state publisher (for visualization)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            output='screen'
        ),
    ])
```

#### Step 2: Launch the System
```bash
cd ~/humanoid_ws
source install/setup.bash
ros2 launch joint_publisher humanoid_robot.launch.py
```

### Expected Outcome
All nodes should start simultaneously, creating a coordinated humanoid robot system.

## Advanced Exercise: Integration with Simulation

### Objective
Integrate your ROS 2 nodes with Gazebo simulation to control a virtual humanoid robot.

### Steps

#### Step 1: Install Gazebo Integration
```bash
sudo apt install ros-humble-gazebo-ros2-control ros-humble-gazebo-dev
```

#### Step 2: Create a Simple Robot URDF
Create `joint_publisher/urdf/simple_humanoid.urdf`:
```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
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
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.2 0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.2"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 0.8"/>
      </material>
    </visual>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.15"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 0.8"/>
      </material>
    </visual>
  </link>

  <!-- Similar definitions for right arm, legs, etc. -->
</robot>
```

#### Step 3: Test Integration
```bash
# Launch RViz with the robot model
ros2 run rviz2 rviz2

# In another terminal, publish the robot description
ros2 run joint_publisher joint_publisher
```

## Troubleshooting and Best Practices

### Common Issues
1. **Node Not Found**: Ensure you've sourced the workspace after building
2. **Permission Errors**: Check that your user is in the dialout group
3. **Network Issues**: Verify ROS_DOMAIN_ID matches across machines
4. **TF Issues**: Ensure proper frame relationships in URDF

### Best Practices
1. Use meaningful names for nodes, topics, and services
2. Implement proper error handling in all nodes
3. Use ROS parameters for configurable values
4. Follow ROS 2 coding standards and conventions
5. Test components individually before integration

## Summary

These exercises have covered fundamental ROS 2 concepts for humanoid robotics:
- Creating publishers and subscribers for joint state communication
- Implementing services for synchronous robot commands
- Developing action servers for complex, cancellable operations
- Using launch files for coordinated system startup
- Preparing for integration with simulation environments

Complete these exercises to build a strong foundation in ROS 2 for humanoid robotics applications.
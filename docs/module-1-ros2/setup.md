---
sidebar_label: 'ROS 2 Setup for Humanoid Robotics'
sidebar_position: 2
---

# ROS 2 Setup for Humanoid Robotics

## Prerequisites

Before installing ROS 2, ensure your system meets the following requirements:

### System Requirements
- **Operating System**: Ubuntu 22.04 LTS (Jammy Jellyfish) recommended for humanoid robotics development
- **Processor**: Multi-core processor (Intel i5 or equivalent recommended)
- **Memory**: 8GB RAM minimum, 16GB+ recommended for simulation
- **Storage**: 20GB free space minimum
- **Network**: Internet connection for package installation

### Software Prerequisites
- Basic understanding of Linux command line
- Git for version control
- Python 3.8 or higher
- C++ compiler (GCC/G++)

## Installing ROS 2 Humble Hawksbill

ROS 2 Humble Hawksbill is the recommended version for humanoid robotics due to its long-term support and extensive hardware compatibility.

### Step 1: Set Locale
Ensure your locale is set to UTF-8:
```bash
locale  # Check for UTF-8
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

### Step 2: Add ROS 2 Repository
```bash
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### Step 3: Install ROS 2 Packages
```bash
sudo apt update
sudo apt install ros-humble-desktop-full
```

### Step 4: Install Additional Dependencies
```bash
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### Step 5: Initialize rosdep
```bash
sudo rosdep init
rosdep update
```

## Environment Setup

### Source ROS 2 Environment
Add the following line to your `~/.bashrc` file:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Create a Workspace
For humanoid robotics development, create a dedicated workspace:
```bash
mkdir -p ~/humanoid_ws/src
cd ~/humanoid_ws
colcon build
source install/setup.bash
```

Add the workspace to your environment:
```bash
echo "source ~/humanoid_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Essential ROS 2 Tools for Humanoid Robotics

### RViz Visualization
RViz is crucial for visualizing robot state, sensors, and planning results:
```bash
sudo apt install ros-humble-rviz2
```

### Robot State Publisher
Publishes the robot's state to TF (Transform) tree:
```bash
sudo apt install ros-humble-robot-state-publisher
```

### Joint State Publisher
Publishes joint states for visualization and control:
```bash
sudo apt install ros-humble-joint-state-publisher ros-humble-joint-state-publisher-gui
```

### Navigation 2
For humanoid robot navigation capabilities:
```bash
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
```

### MoveIt 2
Motion planning framework for humanoid robots:
```bash
sudo apt install ros-humble-moveit
```

## Humanoid-Specific Packages

### ROS Control
Essential for controlling humanoid robot joints:
```bash
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers
```

### Gazebo Integration
For simulation with Gazebo:
```bash
sudo apt install ros-humble-gazebo-ros2-control ros-humble-gazebo-dev
```

### URDF Tools
For robot description and visualization:
```bash
sudo apt install ros-humble-urdf ros-humble-xacro
```

## IDE Setup for Humanoid Robotics Development

### VS Code Configuration
Install the ROS extension pack for VS Code:
- ROS
- C/C++
- Python
- GitLens

Create a `.vscode/settings.json` in your workspace:
```json
{
    "python.defaultInterpreterPath": "/usr/bin/python3",
    "cmake.configureArgs": [
        "-DCMAKE_BUILD_TYPE=Debug"
    ],
    "terminal.integrated.env.linux": {
        "ROS_DISTRO": "humble"
    }
}
```

### Colcon Build Configuration
Create a `colcon.meta` file in your workspace root:
```json
{
    "names": {
        "package_name": {
            "cmake-args": [
                "-DCMAKE_BUILD_TYPE=Debug"
            ]
        }
    }
}
```

## Testing Your Installation

### Basic ROS 2 Test
```bash
# Terminal 1
ros2 run demo_nodes_cpp talker

# Terminal 2
ros2 run demo_nodes_py listener
```

### Check Available Packages
```bash
ros2 pkg list | grep -i robot
```

### Launch a Simple Robot Model
```bash
# Download a sample robot URDF
mkdir -p ~/humanoid_ws/src/my_robot_description/urdf
# Add a simple URDF file here

# Launch robot state publisher
ros2 launch robot_state_publisher robot_state_publisher.launch.py --ros-args -p robot_description:='$(find my_robot_description)/urdf/robot.urdf'
```

## Network Configuration for Distributed Robotics

For humanoid robots with distributed computing, configure DDS:
```bash
# Set up ROS domain ID to avoid interference
echo "export ROS_DOMAIN_ID=42" >> ~/.bashrc
source ~/.bashrc
```

## Troubleshooting Common Issues

### Permission Issues
```bash
# Fix ROS package permissions
sudo apt autoremove && sudo apt update
```

### Network Discovery Issues
```bash
# Check network configuration
ip addr show
# Ensure both machines are on same network if using distributed setup
```

### Real-time Performance
For time-critical humanoid control:
```bash
# Install real-time kernel
sudo apt install linux-image-rt-generic
# Configure user for real-time access
sudo usermod -a -G dialout $USER
```

## Next Steps

With ROS 2 installed and configured, you're ready to:
1. Explore ROS 2 concepts in more depth
2. Create your first humanoid robot node
3. Set up simulation environments
4. Integrate with AI systems

The next section covers creating your first ROS 2 nodes specifically designed for humanoid robotics applications.
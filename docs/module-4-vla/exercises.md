---
sidebar_label: 'VLA System Exercises'
sidebar_position: 3
---

# VLA System Exercises: Hands-On Implementation of Vision-Language-Action Systems

## Exercise 1: Building a Basic VLA Pipeline

### Objective
Create a simple Vision-Language-Action pipeline that can understand a basic command, perceive objects in the environment, and execute a simple action.

### Prerequisites
- ROS 2 Humble installed
- Basic Python programming knowledge
- Completed Modules 1-3 (ROS 2, Digital Twins, Isaac AI)

### Steps

#### Step 1: Create a VLA Package
```bash
cd ~/humanoid_ws/src
ros2 pkg create --build-type ament_python vla_pipeline --dependencies rclpy std_msgs sensor_msgs geometry_msgs cv_bridge
```

#### Step 2: Create the Basic VLA Node
Create `vla_pipeline/vla_pipeline/basic_vla_node.py`:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import json

class BasicVLAPipeline(Node):
    def __init__(self):
        super().__init__('basic_vla_pipeline')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/vla_commands',
            self.command_callback,
            10
        )

        # Create publishers
        self.action_pub = self.create_publisher(
            String,
            '/robot_actions',
            10
        )

        self.visualization_pub = self.create_publisher(
            Image,
            '/vla_visualization',
            10
        )

        # State variables
        self.current_image = None
        self.detected_objects = []
        self.current_command = None

        # Simple object detector (for demonstration)
        self.object_classes = ['cup', 'bottle', 'box', 'person']

        self.get_logger().info('Basic VLA Pipeline initialized')

    def image_callback(self, msg):
        """Process incoming images"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.detect_objects()
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg):
        """Process incoming commands"""
        self.current_command = msg.data
        self.get_logger().info(f'Received command: {self.current_command}')

        # Process the command and execute action
        self.process_command()

    def detect_objects(self):
        """Simple object detection for demonstration"""
        if self.current_image is None:
            return

        # This is a simplified object detector for demonstration
        # In practice, you would use a deep learning model
        height, width, _ = self.current_image.shape

        # Generate some random detections for demonstration
        self.detected_objects = []
        for i in range(3):  # Create 3 random detections
            x = np.random.randint(0, width - 100)
            y = np.random.randint(0, height - 100)
            w = np.random.randint(50, 100)
            h = np.random.randint(50, 100)

            obj_class = np.random.choice(self.object_classes)
            confidence = np.random.uniform(0.7, 0.95)

            self.detected_objects.append({
                'class': obj_class,
                'bbox': [x, y, x + w, y + h],
                'confidence': confidence
            })

    def process_command(self):
        """Process the command and generate appropriate action"""
        if not self.current_command or not self.detected_objects:
            return

        command = self.current_command.lower()

        # Simple command parsing
        if 'find' in command or 'locate' in command:
            self.handle_find_command(command)
        elif 'pick' in command or 'grasp' in command:
            self.handle_pick_command(command)
        elif 'move' in command or 'go to' in command:
            self.handle_move_command(command)
        else:
            self.get_logger().info(f'Command not understood: {command}')

    def handle_find_command(self, command):
        """Handle find/locate commands"""
        # Extract object to find from command
        target_object = self.extract_target_object(command)

        # Find the object in detected objects
        found_objects = [obj for obj in self.detected_objects
                        if target_object in obj['class']]

        if found_objects:
            # Report found objects
            for obj in found_objects:
                self.get_logger().info(f'Found {obj["class"]} with confidence {obj["confidence"]:.2f}')

            # Publish action to highlight objects
            action_msg = String()
            action_msg.data = f'highlight_objects:{target_object}'
            self.action_pub.publish(action_msg)
        else:
            self.get_logger().info(f'Could not find {target_object} in the scene')

    def handle_pick_command(self, command):
        """Handle pick/grasp commands"""
        target_object = self.extract_target_object(command)

        # Find the object to pick
        target = None
        for obj in self.detected_objects:
            if target_object in obj['class']:
                target = obj
                break

        if target:
            # Calculate grasp position (center of bounding box)
            x1, y1, x2, y2 = target['bbox']
            grasp_x = (x1 + x2) / 2
            grasp_y = (y1 + y2) / 2

            # Create grasp action
            action_msg = String()
            action_msg.data = f'grasp:{grasp_x},{grasp_y}'
            self.action_pub.publish(action_msg)

            self.get_logger().info(f'Planning to grasp {target["class"]} at ({grasp_x}, {grasp_y})')
        else:
            self.get_logger().info(f'Could not find {target_object} to grasp')

    def handle_move_command(self, command):
        """Handle move/go to commands"""
        # Extract target location from command
        if 'kitchen' in command:
            target_location = 'kitchen'
        elif 'living room' in command:
            target_location = 'living_room'
        elif 'bedroom' in command:
            target_location = 'bedroom'
        else:
            target_location = 'default'

        # Publish navigation command
        action_msg = String()
        action_msg.data = f'navigate_to:{target_location}'
        self.action_pub.publish(action_msg)

        self.get_logger().info(f'Planning to navigate to {target_location}')

    def extract_target_object(self, command):
        """Extract target object from command"""
        # Simple keyword matching for demonstration
        for obj_class in self.object_classes:
            if obj_class in command:
                return obj_class
        return 'object'  # Default if no specific object found

    def visualize_detections(self):
        """Visualize object detections on image"""
        if self.current_image is None or not self.detected_objects:
            return None

        vis_image = self.current_image.copy()

        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj['bbox']

            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{obj['class']} {obj['confidence']:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return vis_image

def main(args=None):
    rclpy.init(args=args)
    vla_node = BasicVLAPipeline()

    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        pass
    finally:
        vla_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Create a Simple Command Publisher
Create `vla_pipeline/vla_pipeline/command_publisher.py`:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class CommandPublisher(Node):
    def __init__(self):
        super().__init__('command_publisher')
        self.publisher = self.create_publisher(String, '/vla_commands', 10)

        # Create a timer to send commands periodically
        self.timer = self.create_timer(5.0, self.send_command)
        self.command_index = 0

        self.commands = [
            "find the cup",
            "locate the bottle",
            "pick up the box",
            "move to kitchen",
            "find person"
        ]

    def send_command(self):
        """Send a command from the list"""
        if self.command_index < len(self.commands):
            cmd = String()
            cmd.data = self.commands[self.command_index]

            self.publisher.publish(cmd)
            self.get_logger().info(f'Sent command: {cmd.data}')

            self.command_index += 1
        else:
            self.get_logger().info('All commands sent')

def main(args=None):
    rclpy.init(args=args)
    cmd_publisher = CommandPublisher()

    try:
        rclpy.spin(cmd_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        cmd_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 4: Update setup.py
Add the new executables to `vla_pipeline/setup.py`:
```python
entry_points={
    'console_scripts': [
        'basic_vla_node = vla_pipeline.basic_vla_node:main',
        'command_publisher = vla_pipeline.command_publisher:main',
    ],
},
```

#### Step 5: Build and Test
```bash
cd ~/humanoid_ws
colcon build --packages-select vla_pipeline
source install/setup.bash

# Terminal 1: Run the VLA pipeline
ros2 run vla_pipeline basic_vla_node

# Terminal 2: Publish commands
ros2 run vla_pipeline command_publisher
```

### Expected Outcome
The VLA pipeline should process simple commands like "find the cup" or "move to kitchen" and publish appropriate actions based on object detections in the camera feed.

## Exercise 2: Integrating with Language Models

### Objective
Enhance the basic VLA pipeline by integrating with a language processing model to better understand complex commands.

### Steps

#### Step 1: Install Language Processing Dependencies
```bash
pip3 install transformers torch sentence-transformers
```

#### Step 2: Create Enhanced Language Processing Node
Create `vla_pipeline/vla_pipeline/language_processor.py`:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import numpy as np

class LanguageProcessor(Node):
    def __init__(self):
        super().__init__('language_processor')

        # Subscribe to raw commands
        self.command_sub = self.create_subscription(
            String,
            '/raw_commands',
            self.command_callback,
            10
        )

        # Publish processed commands
        self.processed_cmd_pub = self.create_publisher(
            String,
            '/vla_commands',
            10
        )

        # Initialize NLP models
        self.setup_nlp_models()

        self.get_logger().info('Language Processor initialized')

    def setup_nlp_models(self):
        """Setup natural language processing models"""
        try:
            # Initialize intent classification pipeline
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )

            # Initialize tokenizer for more detailed processing
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = AutoModel.from_pretrained("bert-base-uncased")

            self.get_logger().info('NLP models loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Error loading NLP models: {e}')
            # Fallback to simple keyword matching
            self.classifier = None

    def command_callback(self, msg):
        """Process incoming command with NLP"""
        raw_command = msg.data
        self.get_logger().info(f'Processing raw command: {raw_command}')

        # Process command using NLP
        processed_command = self.process_with_nlp(raw_command)

        # Publish processed command
        cmd_msg = String()
        cmd_msg.data = processed_command
        self.processed_cmd_pub.publish(cmd_msg)

        self.get_logger().info(f'Published processed command: {processed_command}')

    def process_with_nlp(self, command):
        """Process command using natural language understanding"""
        if self.classifier is None:
            # Fallback to simple processing
            return self.simple_command_processing(command)

        # Define possible intents
        candidate_labels = [
            "find object", "grasp object", "navigate",
            "answer question", "follow person", "avoid obstacle"
        ]

        try:
            # Classify the intent
            result = self.classifier(command, candidate_labels)

            # Extract entities (objects, locations, etc.)
            entities = self.extract_entities(command)

            # Create structured command
            structured_command = self.create_structured_command(
                result, entities, command
            )

            return structured_command

        except Exception as e:
            self.get_logger().error(f'Error in NLP processing: {e}')
            return command  # Return original command as fallback

    def extract_entities(self, command):
        """Extract entities like objects, locations, people from command"""
        entities = {
            'objects': [],
            'locations': [],
            'people': [],
            'actions': []
        }

        # Simple keyword-based entity extraction
        object_keywords = ['cup', 'bottle', 'box', 'book', 'phone', 'keys']
        location_keywords = ['kitchen', 'living room', 'bedroom', 'office', 'hallway']
        action_keywords = ['find', 'grasp', 'pick', 'move', 'go', 'bring', 'show']

        command_lower = command.lower()

        for obj in object_keywords:
            if obj in command_lower:
                entities['objects'].append(obj)

        for loc in location_keywords:
            if loc in command_lower:
                entities['locations'].append(loc.replace(' ', '_'))

        for act in action_keywords:
            if act in command_lower:
                entities['actions'].append(act)

        return entities

    def create_structured_command(self, classification_result, entities, original_command):
        """Create a structured command from classification and entities"""
        top_intent = classification_result['labels'][0]
        confidence = classification_result['scores'][0]

        # Only proceed if confidence is high enough
        if confidence < 0.5:
            return original_command  # Return original if not confident

        # Create structured command based on intent
        if "find object" in top_intent:
            if entities['objects']:
                return f"find {entities['objects'][0]}"
        elif "grasp object" in top_intent:
            if entities['objects']:
                return f"grasp {entities['objects'][0]}"
        elif "navigate" in top_intent:
            if entities['locations']:
                return f"navigate_to {entities['locations'][0]}"

        return original_command  # Return original if no clear mapping

    def simple_command_processing(self, command):
        """Simple fallback command processing"""
        # Map common phrases to standard commands
        phrase_mappings = {
            "can you find": "find",
            "could you locate": "find",
            "please pick up": "grasp",
            "grab the": "grasp",
            "go to the": "navigate_to",
            "move to": "navigate_to"
        }

        processed = command.lower()
        for phrase, replacement in phrase_mappings.items():
            if phrase in processed:
                processed = processed.replace(phrase, replacement)
                break

        return processed

def main(args=None):
    rclpy.init(args=args)
    lang_processor = LanguageProcessor()

    try:
        rclpy.spin(lang_processor)
    except KeyboardInterrupt:
        pass
    finally:
        lang_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Update setup.py
Add the language processor to `vla_pipeline/setup.py`:
```python
entry_points={
    'console_scripts': [
        'basic_vla_node = vla_pipeline.basic_vla_node:main',
        'command_publisher = vla_pipeline.command_publisher:main',
        'language_processor = vla_pipeline.language_processor:main',
    ],
},
```

#### Step 4: Build and Test Enhanced Pipeline
```bash
cd ~/humanoid_ws
colcon build --packages-select vla_pipeline
source install/setup.bash

# Terminal 1: Run the enhanced language processor
ros2 run vla_pipeline language_processor

# Terminal 2: Run the basic VLA pipeline
ros2 run vla_pipeline basic_vla_node

# Terminal 3: Send more complex commands
ros2 topic pub /raw_commands std_msgs/String "data: 'Could you please find the red cup in the kitchen'"
```

### Expected Outcome
The enhanced pipeline should better understand complex commands and extract meaningful intent and entities from natural language input.

## Exercise 3: Creating a Vision-Language-Action Integration Demo

### Objective
Create a comprehensive demo that integrates vision, language, and action systems in a realistic scenario.

### Steps

#### Step 1: Create the Demo Package
```bash
cd ~/humanoid_ws/src
ros2 pkg create --build-type ament_python vla_demo --dependencies rclpy std_msgs sensor_msgs geometry_msgs cv_bridge vla_pipeline
```

#### Step 2: Create the Demo Node
Create `vla_demo/vla_demo/coffee_fetching_demo.py`:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose, Point
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from threading import Lock

class CoffeeFetchingDemo(Node):
    def __init__(self):
        super().__init__('coffee_fetching_demo')

        # Initialize CV bridge
        self.bridge = CvBridge()
        self.lock = Lock()

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/demo_commands',
            self.command_callback,
            10
        )

        # Create publishers
        self.action_pub = self.create_publisher(String, '/robot_actions', 10)
        self.speech_pub = self.create_publisher(String, '/speech_output', 10)
        self.status_pub = self.create_publisher(String, '/demo_status', 10)

        # State variables
        self.current_image = None
        self.detected_objects = []
        self.current_command = None
        self.demo_state = 'idle'  # idle, searching, grasping, delivering, returning
        self.coffee_location = None
        self.user_location = None

        # Object detection parameters
        self.object_classes = ['cup', 'mug', 'coffee_cup', 'bottle', 'glass']
        self.user_detected = False

        self.get_logger().info('Coffee Fetching Demo initialized')

    def image_callback(self, msg):
        """Process incoming images"""
        with self.lock:
            try:
                self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.detect_objects_and_people()
            except Exception as e:
                self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg):
        """Process demo commands"""
        command = msg.data.lower()
        self.get_logger().info(f'Received demo command: {command}')

        if 'start demo' in command:
            self.start_demo()
        elif 'find coffee' in command:
            self.find_coffee()
        elif 'bring coffee' in command:
            self.bring_coffee()
        elif 'reset' in command:
            self.reset_demo()

    def start_demo(self):
        """Start the coffee fetching demo"""
        self.demo_state = 'searching'
        self.publish_status('Demo started: Searching for coffee')
        self.speak('Starting coffee fetching demo. I will now search for a coffee cup.')

        # Start searching for coffee
        self.find_coffee()

    def find_coffee(self):
        """Find a coffee cup in the environment"""
        self.demo_state = 'searching'
        self.publish_status('Searching for coffee cup')

        # Continuously search until coffee is found or timeout
        search_start = time.time()
        search_timeout = 10.0  # 10 seconds to find coffee

        while self.demo_state == 'searching' and (time.time() - search_start) < search_timeout:
            with self.lock:
                if self.current_image is not None:
                    # Detect objects
                    self.detect_objects_and_people()

                    # Check if coffee cup is detected
                    coffee_cup = self.find_coffee_cup()
                    if coffee_cup:
                        self.coffee_location = self.get_object_center(coffee_cup)
                        self.demo_state = 'found'
                        self.publish_status(f'Found coffee cup at {self.coffee_location}')
                        self.speak(f'I found a coffee cup. It is at coordinates {self.coffee_location}')
                        break

            time.sleep(0.1)  # Small delay to prevent excessive CPU usage

        if self.demo_state != 'found':
            self.demo_state = 'idle'
            self.publish_status('Could not find coffee cup')
            self.speak('I could not find a coffee cup. Please place one in my view and try again.')

    def bring_coffee(self):
        """Bring coffee to the user"""
        if self.demo_state != 'found':
            self.speak('I need to find the coffee first. Starting search.')
            self.find_coffee()
            return

        self.demo_state = 'grasping'
        self.publish_status('Approaching coffee cup')

        # Navigate to coffee location
        self.navigate_to_location(self.coffee_location)

        # Grasp the coffee cup
        self.grasp_coffee_cup()

        # Find user and navigate to them
        self.find_and_deliver_to_user()

    def detect_objects_and_people(self):
        """Detect objects and people in the current image"""
        if self.current_image is None:
            return

        height, width, _ = self.current_image.shape

        # This is a simplified detector for demonstration
        # In practice, you would use a deep learning model
        self.detected_objects = []

        # Generate some detections for demonstration
        for i in range(3):
            x = np.random.randint(0, width - 100)
            y = np.random.randint(0, height - 100)
            w = np.random.randint(50, 100)
            h = np.random.randint(50, 100)

            obj_class = np.random.choice(self.object_classes)
            confidence = np.random.uniform(0.7, 0.95)

            self.detected_objects.append({
                'class': obj_class,
                'bbox': [x, y, x + w, y + h],
                'confidence': confidence
            })

        # Simulate person detection
        self.user_detected = np.random.choice([True, False], p=[0.3, 0.7])

    def find_coffee_cup(self):
        """Find a coffee cup among detected objects"""
        for obj in self.detected_objects:
            if any(coffee_type in obj['class'] for coffee_type in ['cup', 'mug', 'coffee_cup']):
                return obj
        return None

    def get_object_center(self, obj):
        """Get the center coordinates of an object"""
        x1, y1, x2, y2 = obj['bbox']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (center_x, center_y)

    def navigate_to_location(self, location):
        """Navigate to a specific location"""
        self.publish_status(f'Navigating to location {location}')
        self.speak(f'Navigating to the coffee cup location.')

        # Publish navigation command
        nav_msg = String()
        nav_msg.data = f'navigate_absolute:{location[0]},{location[1]}'
        self.action_pub.publish(nav_msg)

        # Simulate navigation time
        time.sleep(2.0)

    def grasp_coffee_cup(self):
        """Grasp the coffee cup"""
        self.demo_state = 'grasping'
        self.publish_status('Grasping coffee cup')
        self.speak('Grasping the coffee cup.')

        # Publish grasp command
        grasp_msg = String()
        grasp_msg.data = f'grasp:{self.coffee_location[0]},{self.coffee_location[1]}'
        self.action_pub.publish(grasp_msg)

        # Simulate grasp time
        time.sleep(2.0)

        self.demo_state = 'delivering'
        self.publish_status('Successfully grasped coffee cup')

    def find_and_deliver_to_user(self):
        """Find the user and deliver the coffee"""
        self.publish_status('Searching for user to deliver coffee')
        self.speak('Searching for you to deliver the coffee.')

        # Simulate finding user
        search_start = time.time()
        search_timeout = 5.0

        while self.demo_state == 'delivering' and (time.time() - search_start) < search_timeout:
            with self.lock:
                if self.user_detected:
                    self.demo_state = 'delivering_to_user'
                    self.publish_status('Found user, delivering coffee')
                    self.speak('I found you. Delivering the coffee now.')
                    break

            time.sleep(0.1)

        # Navigate to user
        self.navigate_to_user()

        # Release coffee cup
        self.release_coffee_cup()

        self.demo_state = 'completed'
        self.publish_status('Coffee delivery completed')
        self.speak('I have delivered your coffee. Enjoy!')

    def navigate_to_user(self):
        """Navigate to the user"""
        # Publish navigation command to user location
        nav_msg = String()
        nav_msg.data = 'navigate_to_user'
        self.action_pub.publish(nav_msg)

        # Simulate navigation time
        time.sleep(3.0)

    def release_coffee_cup(self):
        """Release the coffee cup"""
        self.publish_status('Releasing coffee cup')
        self.speak('Releasing the coffee cup.')

        # Publish release command
        release_msg = String()
        release_msg.data = 'release_object'
        self.action_pub.publish(release_msg)

        # Simulate release time
        time.sleep(1.0)

    def reset_demo(self):
        """Reset the demo to initial state"""
        self.demo_state = 'idle'
        self.coffee_location = None
        self.user_detected = False
        self.publish_status('Demo reset')
        self.speak('Demo has been reset. Ready for new commands.')

    def speak(self, text):
        """Publish speech output"""
        speech_msg = String()
        speech_msg.data = text
        self.speech_pub.publish(speech_msg)

    def publish_status(self, status):
        """Publish demo status"""
        status_msg = String()
        status_msg.data = f'Demo Status: {status}'
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    demo_node = CoffeeFetchingDemo()

    try:
        rclpy.spin(demo_node)
    except KeyboardInterrupt:
        pass
    finally:
        demo_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Create Demo Launcher
Create `vla_demo/launch/coffee_demo.launch.py`:
```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    # VLA Demo Node
    vla_demo_node = Node(
        package='vla_demo',
        executable='coffee_fetching_demo',
        name='coffee_fetching_demo',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # Language Processor (if available)
    language_processor = Node(
        package='vla_pipeline',
        executable='language_processor',
        name='language_processor',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen',
        condition=launch.conditions.IfCondition(LaunchConfiguration('enable_language'))
    )

    # Basic VLA Node (if available)
    basic_vla_node = Node(
        package='vla_pipeline',
        executable='basic_vla_node',
        name='basic_vla_node',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time,
        DeclareLaunchArgument(
            'enable_language',
            default_value='false',
            description='Enable language processing component'
        ),
        vla_demo_node,
        language_processor,
        basic_vla_node
    ])
```

#### Step 4: Update setup.py for Demo Package
Create `vla_demo/setup.py`:
```python
from setuptools import find_packages, setup

package_name = 'vla_demo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/coffee_demo.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='VLA Demo Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'coffee_fetching_demo = vla_demo.coffee_fetching_demo:main',
        ],
    },
)
```

#### Step 5: Build and Run the Demo
```bash
cd ~/humanoid_ws
colcon build --packages-select vla_demo
source install/setup.bash

# Run the demo
ros2 launch vla_demo coffee_demo.launch.py

# In another terminal, send demo commands
ros2 topic pub /demo_commands std_msgs/String "data: 'start demo'"
ros2 topic pub /demo_commands std_msgs/String "data: 'bring coffee'"
```

### Expected Outcome
The demo should simulate a complete coffee fetching task, including searching for a coffee cup, navigating to it, grasping it, finding the user, and delivering the coffee.

## Exercise 4: Performance Monitoring and Optimization

### Objective
Implement performance monitoring for the VLA system and optimize its performance.

### Steps

#### Step 1: Create Performance Monitoring Node
Create `vla_demo/vla_demo/performance_monitor.py`:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
import time
import psutil
import GPUtil
from collections import deque
import threading

class VLAPerformanceMonitor(Node):
    def __init__(self):
        super().__init__('vla_performance_monitor')

        # Subscribe to system metrics
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.command_sub = self.create_subscription(String, '/vla_commands', self.command_callback, 10)

        # Publish performance metrics
        self.cpu_pub = self.create_publisher(Float32, '/performance/cpu_usage', 10)
        self.gpu_pub = self.create_publisher(Float32, '/performance/gpu_usage', 10)
        self.memory_pub = self.create_publisher(Float32, '/performance/memory_usage', 10)
        self.latency_pub = self.create_publisher(Float32, '/performance/processing_latency', 10)
        self.fps_pub = self.create_publisher(Float32, '/performance/processing_fps', 10)

        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.start_time = time.time()

        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.system_monitor_loop)
        self.monitor_thread.start()

        self.get_logger().info('VLA Performance Monitor initialized')

    def image_callback(self, msg):
        """Monitor image processing performance"""
        start_time = time.time()

        # Simulate image processing
        self.process_image(msg)

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        # Calculate and publish latency
        latency_msg = Float32()
        latency_msg.data = processing_time
        self.latency_pub.publish(latency_msg)

        # Calculate FPS
        current_time = time.time()
        self.frame_count += 1

        if current_time - self.last_frame_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_frame_time)
            self.frame_count = 0
            self.last_frame_time = current_time

            fps_msg = Float32()
            fps_msg.data = fps
            self.fps_pub.publish(fps_msg)

    def command_callback(self, msg):
        """Monitor command processing performance"""
        start_time = time.time()

        # Simulate command processing
        self.process_command(msg)

        processing_time = time.time() - start_time

        # Add to processing times for overall monitoring
        self.processing_times.append(processing_time)

    def process_image(self, image_msg):
        """Simulate image processing"""
        # In a real system, this would involve object detection, etc.
        time.sleep(0.01)  # Simulate processing time

    def process_command(self, cmd_msg):
        """Simulate command processing"""
        # In a real system, this would involve NLP processing
        time.sleep(0.005)  # Simulate processing time

    def system_monitor_loop(self):
        """Monitor system resources"""
        while self.monitoring:
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            cpu_msg = Float32()
            cpu_msg.data = cpu_percent
            self.cpu_pub.publish(cpu_msg)

            # Memory usage
            memory_percent = psutil.virtual_memory().percent
            memory_msg = Float32()
            memory_msg.data = memory_percent
            self.memory_pub.publish(memory_msg)

            # GPU usage (if available)
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_load = gpus[0].load * 100
                gpu_msg = Float32()
                gpu_msg.data = gpu_load
                self.gpu_pub.publish(gpu_msg)
            else:
                gpu_msg = Float32()
                gpu_msg.data = 0.0
                self.gpu_pub.publish(gpu_msg)

            time.sleep(1.0)  # Update every second

    def destroy_node(self):
        """Clean up monitoring thread"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    monitor = VLAPerformanceMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 2: Create Performance Analysis Tool
Create `vla_demo/vla_demo/performance_analyzer.py`:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

class PerformanceAnalyzer(Node):
    def __init__(self):
        super().__init__('performance_analyzer')

        # Subscribe to performance metrics
        self.cpu_sub = self.create_subscription(Float32, '/performance/cpu_usage', self.cpu_callback, 10)
        self.gpu_sub = self.create_subscription(Float32, '/performance/gpu_usage', self.gpu_callback, 10)
        self.memory_sub = self.create_subscription(Float32, '/performance/memory_usage', self.memory_callback, 10)
        self.latency_sub = self.create_subscription(Float32, '/performance/processing_latency', self.latency_callback, 10)
        self.fps_sub = self.create_subscription(Float32, '/performance/processing_fps', self.fps_callback, 10)

        # Data storage
        self.cpu_data = deque(maxlen=1000)
        self.gpu_data = deque(maxlen=1000)
        self.memory_data = deque(maxlen=1000)
        self.latency_data = deque(maxlen=1000)
        self.fps_data = deque(maxlen=1000)

        # Time tracking
        self.time_stamps = deque(maxlen=1000)
        self.start_time = time.time()

        # Analysis timer
        self.analysis_timer = self.create_timer(10.0, self.run_analysis)

        self.get_logger().info('Performance Analyzer initialized')

    def cpu_callback(self, msg):
        self.cpu_data.append(msg.data)
        self.time_stamps.append(time.time() - self.start_time)

    def gpu_callback(self, msg):
        self.gpu_data.append(msg.data)

    def memory_callback(self, msg):
        self.memory_data.append(msg.data)

    def latency_callback(self, msg):
        self.latency_data.append(msg.data)

    def fps_callback(self, msg):
        self.fps_data.append(msg.data)

    def run_analysis(self):
        """Run performance analysis and generate reports"""
        if len(self.cpu_data) < 10:  # Need minimum data points
            return

        # Calculate statistics
        cpu_stats = self.calculate_statistics(self.cpu_data, 'CPU')
        gpu_stats = self.calculate_statistics(self.gpu_data, 'GPU')
        memory_stats = self.calculate_statistics(self.memory_data, 'Memory')
        latency_stats = self.calculate_statistics(self.latency_data, 'Latency')
        fps_stats = self.calculate_statistics(self.fps_data, 'FPS')

        # Log statistics
        self.log_statistics(cpu_stats, 'CPU')
        self.log_statistics(gpu_stats, 'GPU')
        self.log_statistics(memory_stats, 'Memory')
        self.log_statistics(latency_stats, 'Latency')
        self.log_statistics(fps_stats, 'FPS')

        # Check for performance issues
        self.check_performance_issues(cpu_stats, gpu_stats, memory_stats, latency_stats, fps_stats)

    def calculate_statistics(self, data, name):
        """Calculate statistics for performance data"""
        if not data:
            return None

        data_array = np.array(data)
        stats = {
            'mean': np.mean(data_array),
            'std': np.std(data_array),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'median': np.median(data_array),
            'percentile_95': np.percentile(data_array, 95),
            'current': data_array[-1] if len(data_array) > 0 else 0
        }
        return stats

    def log_statistics(self, stats, name):
        """Log performance statistics"""
        if stats is None:
            return

        self.get_logger().info(f'{name} Performance Stats:')
        self.get_logger().info(f'  Mean: {stats["mean"]:.2f}')
        self.get_logger().info(f'  Std: {stats["std"]:.2f}')
        self.get_logger().info(f'  Min: {stats["min"]:.2f}')
        self.get_logger().info(f'  Max: {stats["max"]:.2f}')
        self.get_logger().info(f'  Median: {stats["median"]:.2f}')
        self.get_logger().info(f'  95th Percentile: {stats["percentile_95"]:.2f}')
        self.get_logger().info(f'  Current: {stats["current"]:.2f}')

    def check_performance_issues(self, cpu_stats, gpu_stats, memory_stats, latency_stats, fps_stats):
        """Check for performance issues and log warnings"""
        issues = []

        # Check CPU usage
        if cpu_stats and cpu_stats['mean'] > 80:
            issues.append(f'High CPU usage: {cpu_stats["mean"]:.2f}%')

        # Check GPU usage
        if gpu_stats and gpu_stats['mean'] > 85:
            issues.append(f'High GPU usage: {gpu_stats["mean"]:.2f}%')

        # Check memory usage
        if memory_stats and memory_stats['mean'] > 80:
            issues.append(f'High memory usage: {memory_stats["mean"]:.2f}%')

        # Check latency
        if latency_stats and latency_stats['percentile_95'] > 0.1:  # > 100ms
            issues.append(f'High processing latency: {latency_stats["percentile_95"]:.3f}s (95th percentile)')

        # Check FPS
        if fps_stats and fps_stats['mean'] < 10:  # < 10 FPS
            issues.append(f'Low processing rate: {fps_stats["mean"]:.2f} FPS')

        # Log issues
        if issues:
            for issue in issues:
                self.get_logger().warn(f'Performance Issue: {issue}')
        else:
            self.get_logger().info('All performance metrics within acceptable ranges')

def main(args=None):
    rclpy.init(args=args)
    analyzer = PerformanceAnalyzer()

    try:
        rclpy.spin(analyzer)
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Build and Test Performance Tools
```bash
cd ~/humanoid_ws
colcon build --packages-select vla_demo
source install/setup.bash

# Terminal 1: Run the VLA demo
ros2 run vla_demo coffee_fetching_demo

# Terminal 2: Run the performance monitor
ros2 run vla_demo performance_monitor

# Terminal 3: Run the performance analyzer
ros2 run vla_demo performance_analyzer
```

### Expected Outcome
The performance monitoring tools should track and analyze the VLA system's performance, identifying potential bottlenecks and optimization opportunities.

## Advanced Exercise: Integration with Real Hardware

### Objective
Integrate the VLA system with real robotic hardware for physical demonstration.

### Steps

#### Step 1: Create Hardware Interface Layer
Create `vla_demo/vla_demo/hardware_interface.py`:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np

class HardwareInterface(Node):
    def __init__(self):
        super().__init__('hardware_interface')

        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Subscribe to VLA commands
        self.vla_command_sub = self.create_subscription(
            String, '/vla_commands', self.vla_command_callback, qos_profile
        )

        # Subscribe to sensor data
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, qos_profile
        )

        # Publish to hardware controllers
        self.arm_command_pub = self.create_publisher(JointState, '/arm_controller/commands', qos_profile)
        self.base_command_pub = self.create_publisher(Twist, '/base_controller/cmd_vel', qos_profile)
        self.gripper_command_pub = self.create_publisher(Bool, '/gripper_controller/command', qos_profile)

        # Hardware state
        self.current_joint_states = JointState()
        self.hardware_connected = True  # Assume connected for simulation

        self.get_logger().info('Hardware Interface initialized')

    def vla_command_callback(self, msg):
        """Process VLA commands and convert to hardware commands"""
        command = msg.data
        self.get_logger().info(f'Received VLA command: {command}')

        # Parse and execute command
        if command.startswith('grasp:'):
            self.execute_grasp_command(command)
        elif command.startswith('navigate_to:'):
            self.execute_navigation_command(command)
        elif command.startswith('navigate_absolute:'):
            self.execute_absolute_navigation_command(command)
        elif command == 'release_object':
            self.execute_release_command()
        elif command == 'navigate_to_user':
            self.execute_user_navigation_command()

    def execute_grasp_command(self, command):
        """Execute grasp command"""
        try:
            # Parse coordinates from command (format: "grasp:x,y")
            coords_str = command.split(':')[1]
            x, y = map(float, coords_str.split(','))

            self.get_logger().info(f'Executing grasp at coordinates: ({x}, {y})')

            # Calculate joint angles for grasping
            joint_angles = self.calculate_grasp_angles(x, y)

            # Send joint commands to arm
            joint_msg = JointState()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.name = ['joint_1', 'joint_2', 'joint_3']  # Example joint names
            joint_msg.position = joint_angles
            joint_msg.velocity = [0.0, 0.0, 0.0]
            joint_msg.effort = [0.0, 0.0, 0.0]

            self.arm_command_pub.publish(joint_msg)

            # Close gripper
            gripper_msg = Bool()
            gripper_msg.data = True
            self.gripper_command_pub.publish(gripper_msg)

        except Exception as e:
            self.get_logger().error(f'Error executing grasp command: {e}')

    def execute_navigation_command(self, command):
        """Execute navigation command"""
        try:
            # Parse location from command (format: "navigate_to:location_name")
            location = command.split(':')[1]

            self.get_logger().info(f'Navigating to location: {location}')

            # Convert location to coordinates or use predefined locations
            target_pose = self.get_location_coordinates(location)

            # Generate navigation commands
            self.navigate_to_pose(target_pose)

        except Exception as e:
            self.get_logger().error(f'Error executing navigation command: {e}')

    def execute_absolute_navigation_command(self, command):
        """Execute absolute navigation command"""
        try:
            # Parse coordinates from command (format: "navigate_absolute:x,y")
            coords_str = command.split(':')[1]
            x, y = map(float, coords_str.split(','))

            self.get_logger().info(f'Navigating to absolute coordinates: ({x}, {y})')

            # Create target pose
            target_pose = Pose()
            target_pose.position.x = x
            target_pose.position.y = y
            target_pose.position.z = 0.0

            # Navigate to pose
            self.navigate_to_pose(target_pose)

        except Exception as e:
            self.get_logger().error(f'Error executing absolute navigation: {e}')

    def execute_release_command(self):
        """Execute object release command"""
        self.get_logger().info('Releasing object')

        # Open gripper
        gripper_msg = Bool()
        gripper_msg.data = False
        self.gripper_command_pub.publish(gripper_msg)

    def execute_user_navigation_command(self):
        """Execute navigation to user command"""
        self.get_logger().info('Navigating to user')

        # This would typically involve person-following behavior
        # For simulation, just move forward
        twist_msg = Twist()
        twist_msg.linear.x = 0.2  # Move forward at 0.2 m/s
        twist_msg.angular.z = 0.0

        self.base_command_pub.publish(twist_msg)

        # Stop after 3 seconds
        self.create_timer(3.0, self.stop_base)

    def calculate_grasp_angles(self, x, y):
        """Calculate joint angles for grasping at given coordinates"""
        # This is a simplified calculation
        # In practice, this would involve inverse kinematics
        angles = [0.0, 0.0, 0.0]  # Default angles

        # Simulate some basic IK calculation
        angles[0] = np.arctan2(y, x)  # Base rotation
        angles[1] = np.clip(x * 0.1, -1.5, 1.5)  # Elbow angle
        angles[2] = np.clip(y * 0.1, -1.0, 1.0)  # Wrist angle

        return angles

    def get_location_coordinates(self, location_name):
        """Get coordinates for predefined locations"""
        locations = {
            'kitchen': Pose(position=Point(x=2.0, y=0.0, z=0.0)),
            'living_room': Pose(position=Point(x=0.0, y=2.0, z=0.0)),
            'bedroom': Pose(position=Point(x=-2.0, y=0.0, z=0.0)),
            'office': Pose(position=Point(x=0.0, y=-2.0, z=0.0))
        }

        return locations.get(location_name, Pose())

    def navigate_to_pose(self, target_pose):
        """Navigate to a target pose"""
        # This is a simplified navigation
        # In practice, this would use navigation2 or similar
        current_pos = self.get_current_position()

        # Calculate direction to target
        dx = target_pose.position.x - current_pos.x
        dy = target_pose.position.y - current_pos.y

        # Create navigation command
        twist_msg = Twist()
        twist_msg.linear.x = np.clip(np.sqrt(dx*dx + dy*dy) * 0.5, 0.0, 0.5)  # Forward speed
        twist_msg.angular.z = np.arctan2(dy, dx) * 0.5  # Turn toward target

        self.base_command_pub.publish(twist_msg)

    def get_current_position(self):
        """Get current robot position (simplified)"""
        # In a real system, this would come from localization
        return Point(x=0.0, y=0.0, z=0.0)

    def joint_state_callback(self, msg):
        """Update current joint states"""
        self.current_joint_states = msg

    def stop_base(self):
        """Stop the base movement"""
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.base_command_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    hw_interface = HardwareInterface()

    try:
        rclpy.spin(hw_interface)
    except KeyboardInterrupt:
        pass
    finally:
        hw_interface.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 2: Update setup.py to include hardware interface
Add to `vla_demo/setup.py`:
```python
entry_points={
    'console_scripts': [
        'coffee_fetching_demo = vla_demo.coffee_fetching_demo:main',
        'performance_monitor = vla_demo.performance_monitor:main',
        'performance_analyzer = vla_demo.performance_analyzer:main',
        'hardware_interface = vla_demo.hardware_interface:main',
    ],
},
```

#### Step 3: Create Complete System Launch File
Create `vla_demo/launch/vla_complete_system.launch.py`:
```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    enable_language = DeclareLaunchArgument(
        'enable_language',
        default_value='true',
        description='Enable language processing component'
    )

    # VLA Demo Node
    vla_demo_node = Node(
        package='vla_demo',
        executable='coffee_fetching_demo',
        name='coffee_fetching_demo',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # Language Processor
    language_processor = Node(
        package='vla_pipeline',
        executable='language_processor',
        name='language_processor',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # Performance Monitor
    performance_monitor = Node(
        package='vla_demo',
        executable='performance_monitor',
        name='performance_monitor',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # Performance Analyzer
    performance_analyzer = Node(
        package='vla_demo',
        executable='performance_analyzer',
        name='performance_analyzer',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # Hardware Interface
    hardware_interface = Node(
        package='vla_demo',
        executable='hardware_interface',
        name='hardware_interface',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time,
        enable_language,
        vla_demo_node,
        language_processor,
        performance_monitor,
        performance_analyzer,
        hardware_interface
    ])
```

### Expected Outcome
The complete VLA system should now be integrated with a hardware interface layer that can translate VLA commands into actual robot actions, with performance monitoring and analysis capabilities.

## Troubleshooting and Best Practices

### Common Issues

1. **Latency Problems**: Ensure all nodes are running efficiently and communication is optimized
2. **Object Detection Failures**: Verify camera calibration and lighting conditions
3. **Command Understanding**: Test with clear, simple commands initially
4. **Safety Issues**: Always have emergency stop procedures in place

### Best Practices

1. **Modular Design**: Keep components independent for easier debugging
2. **Error Handling**: Implement robust error handling and recovery
3. **Performance Monitoring**: Continuously monitor system performance
4. **Safety First**: Implement safety checks at every level
5. **Testing**: Test components individually before system integration

## Summary

These exercises have covered the implementation of Vision-Language-Action systems for humanoid robotics, including:
- Basic VLA pipeline with vision, language, and action components
- Integration with natural language processing models
- Complete demonstration system for a practical task
- Performance monitoring and optimization
- Hardware integration for real-world deployment

Complete these exercises to gain hands-on experience with VLA systems and their practical implementation in humanoid robotics applications.
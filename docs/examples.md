# Examples

This section provides examples demonstrating how to use the Physics Simulator. All examples are based on actual working code from the examples directory.

## Basic Usage

### Setting Up a Basic Simulation

The most fundamental usage pattern for the Physics Simulator.

**File:** `examples/basic_usage.py`

```python
from physics_simulator import PhysicsSimulator
from synthnova_config import PhysicsSimulatorConfig

def main():
    # Initialize the simulator
    my_config = PhysicsSimulatorConfig()
    synthnova_physics_simulator = PhysicsSimulator(my_config)

    # Add default scene
    synthnova_physics_simulator.add_default_scene()

    # Initialize the simulator
    synthnova_physics_simulator.initialize()

    # Run the display loop
    synthnova_physics_simulator.loop()

    # Close the simulator
    synthnova_physics_simulator.close()


if __name__ == "__main__":
    main()
```

## Adding Entities

### Adding Basic Geometric Objects

Learn how to add primitive objects like cubes to your simulation.

**File:** `examples/add_entity/add_basic_geom.py`

```python
from synthnova_config import PhysicsSimulatorConfig, CuboidConfig
import os
from physics_simulator import PhysicsSimulator

def main():
    # Instantiate the simulator
    my_config = PhysicsSimulatorConfig()
    synthnova_physics_simulator = PhysicsSimulator(my_config)

    # Add default ground plane
    synthnova_physics_simulator.add_default_scene()

    # Add a red dynamic cube
    cube_1_config = CuboidConfig(
        prim_path=os.path.join(synthnova_physics_simulator.root_prim_path, "cube_1"),
        position=[2, 2, 2],
        orientation=[0, 0, 0, 1],
        scale=[1, 1, 1],
        color=[1.0, 0.0, 0.0],
    )
    cube_1_path = synthnova_physics_simulator.add_object(cube_1_config)

    # Add a green static cube
    cube_2_config = CuboidConfig(
        prim_path=os.path.join(synthnova_physics_simulator.root_prim_path, "cube_2"),
        position=[0, 0, 10],
        orientation=[0, 0, 0, 1],
        scale=[1, 1, 1],
        color=[0.0, 1.0, 0.0],
        interaction_type="static",
    )
    cube_2_path = synthnova_physics_simulator.add_object(cube_2_config)

    # Play the simulator
    synthnova_physics_simulator.play()

    # Run the display loop
    synthnova_physics_simulator.loop()


if __name__ == "__main__":
    main()
```

### Adding Mesh Objects

Import complex mesh objects into your simulation.

**File:** `examples/add_entity/add_mesh.py`

```python
from physics_simulator import PhysicsSimulator
from synthnova_config import PhysicsSimulatorConfig, RobotConfig, MeshConfig
from pathlib import Path


def main():
    # Create sim config
    my_config = PhysicsSimulatorConfig()

    # Initialize the simulator
    synthnova_physics_simulator = PhysicsSimulator(my_config)

    # Add default scene
    synthnova_physics_simulator.add_default_scene()

    # Add robot first
    robot_config = RobotConfig(
        prim_path="/World/Galbot",
        name="galbot_one_foxtrot",
        mjcf_path=Path()
            .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("robots")
            .joinpath("galbot_one_foxtrot_description")
            .joinpath("galbot_one_foxtrot.xml"),
        position=[0, 0, 0],
        orientation=[0, 0, 0, 1]
    )
    synthnova_physics_simulator.add_robot(robot_config)

    # Add a mesh object (mug)
    mug_config = MeshConfig(
        prim_path="/World/mug",
        name="closet",
        mjcf_path=Path()
            .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("closet")
            .joinpath("closet.xml"),
        position=[3, 0, 0],
        orientation=[0, 0, 0, 1]
    )
    synthnova_physics_simulator.add_object(mug_config)

    # Play and run simulation
    synthnova_physics_simulator.play()
    synthnova_physics_simulator.loop()


if __name__ == "__main__":
    main()

```

### Adding Robots

Add robots with MJCF/XML configurations.

**File:** `examples/add_entity/add_robot.py`

```python
from physics_simulator import PhysicsSimulator
from synthnova_config import PhysicsSimulatorConfig, RobotConfig
from pathlib import Path

def main():
    # Create sim config
    my_config = PhysicsSimulatorConfig()

    # Initialize the simulator
    synthnova_physics_simulator = PhysicsSimulator(my_config)

    # Add default scene
    synthnova_physics_simulator.add_default_scene()

    # Add Galbot robot
    robot_config = RobotConfig(
        prim_path="/World/Galbot",
        name="galbot_one_foxtrot",
        mjcf_path=Path()
            .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("robots")
            .joinpath("galbot_one_foxtrot_description")
            .joinpath("galbot_one_foxtrot.xml"),
        position=[0, 0, 0],
        orientation=[0, 0, 0, 1]
    )
    robot_path = synthnova_physics_simulator.add_robot(robot_config)
    
    # Initialize and run simulation
    synthnova_physics_simulator.initialize()
    synthnova_physics_simulator.loop()
```

## Galbot Interface Examples

### Basic Robot Control

Examples of controlling individual robot modules using the Galbot Interface.

#### Chassis Control

Control the mobile base for omnidirectional movement.

**File:** `examples/galbot_interface_examples/basic/chassis.py`

```python
from physics_simulator import PhysicsSimulator
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
from physics_simulator.utils.data_types import JointTrajectory
from synthnova_config import PhysicsSimulatorConfig, RobotConfig
import numpy as np
from pathlib import Path

def main():
    # Create sim config
    my_config = PhysicsSimulatorConfig()
    synthnova_physics_simulator = PhysicsSimulator(my_config)

    # Add default ground plane and robot
    synthnova_physics_simulator.add_default_scene()
    robot_config = RobotConfig(
        prim_path="/World/Galbot",
        name="galbot_one_foxtrot",
        mjcf_path=Path()
            .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("robots")
            .joinpath("galbot_one_foxtrot_description")
            .joinpath("galbot_one_foxtrot.xml"),
        position=[0, 0, 0],
        orientation=[0, 0, 0, 1]
    )
    robot_path = synthnova_physics_simulator.add_robot(robot_config)

    # Initialize the simulator
    synthnova_physics_simulator.initialize()

    # Initialize the galbot interface
    galbot_interface_config = GalbotInterfaceConfig()
    galbot_interface_config.modules_manager.enabled_modules.append("chassis")
    galbot_interface_config.chassis.joint_names = [
        f"{robot_config.name}/mobile_forward_joint",
        f"{robot_config.name}/mobile_side_joint",
        f"{robot_config.name}/mobile_yaw_joint",
    ]
    galbot_interface_config.robot.prim_path = robot_path

    galbot_interface = GalbotInterface(galbot_interface_config, synthnova_physics_simulator)
    galbot_interface.initialize()

    # Control the chassis
    galbot_interface.chassis.set_joint_velocities([0.5, 0.0, 0.0])  # Move forward
    
    synthnova_physics_simulator.loop()
```

#### Arm Control

Control robotic arms with joint position control.

**File:** `examples/galbot_interface_examples/basic/left_arm.py`

```python
# Similar setup as chassis example, but enable left_arm module
galbot_interface_config.modules_manager.enabled_modules.append("left_arm")
galbot_interface_config.left_arm.joint_names = [
    f"{robot_config.name}/left_arm_joint_1",
    f"{robot_config.name}/left_arm_joint_2",
    f"{robot_config.name}/left_arm_joint_3",
    f"{robot_config.name}/left_arm_joint_4",
    f"{robot_config.name}/left_arm_joint_5",
    f"{robot_config.name}/left_arm_joint_6",
    f"{robot_config.name}/left_arm_joint_7",
]

# Control the arm
start_positions = galbot_interface.left_arm.get_joint_positions()
target_positions = [0.0, -0.5, 0.0, 1.0, 0.0, 0.5, 0.0]
galbot_interface.left_arm.set_joint_positions(target_positions)
```

#### Gripper Control

Control grippers for object manipulation.

**File:** `examples/galbot_interface_examples/basic/left_gripper.py`

```python
# Configure gripper module
galbot_interface_config.modules_manager.enabled_modules.append("left_gripper")
galbot_interface_config.left_gripper.joint_names = [
    f"{robot_config.name}/left_gripper_joint_1",
    f"{robot_config.name}/left_gripper_joint_2",
]

# Control gripper
galbot_interface.left_gripper.close()  # Close gripper
galbot_interface.left_gripper.open()   # Open gripper
```

### Sensor Examples

Examples demonstrating sensor data collection from robot cameras.

#### Front Head Camera

Capture RGB images from the front head camera.

**File:** `examples/galbot_interface_examples/sensors/front_head_camera.py`

```python
from physics_simulator import PhysicsSimulator
from synthnova_config import (
    PhysicsSimulatorConfig,
    RobotConfig,
    RgbCameraConfig,
)
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
import cv2
from pathlib import Path

def main():
    # Setup simulator and robot (similar to previous examples)
    my_config = PhysicsSimulatorConfig()
    synthnova_physics_simulator = PhysicsSimulator(my_config)
    synthnova_physics_simulator.add_default_scene()
    
    # Add robot and camera
    robot_config = RobotConfig(...)
    robot_path = synthnova_physics_simulator.add_robot(robot_config)

    # Add front head RGB camera
    front_head_rgb_camera_config = RgbCameraConfig(
        name="front_head_rgb_camera",
        prim_path=os.path.join(
            robot_path,
            "head_link2",
            "head_end_effector_mount_link",
            "front_head_rgb_camera",
        ),
        translation=[0.0858, -0.1017, -0.0078],
        rotation=[0.5, -0.5, 0.5, -0.5],
        resolution=[640, 480]
    )
    synthnova_physics_simulator.add_camera(front_head_rgb_camera_config)

    # Initialize galbot interface with camera
    galbot_interface_config = GalbotInterfaceConfig()
    galbot_interface_config.modules_manager.enabled_modules.append("front_head_camera")
    galbot_interface_config.front_head_camera.camera_name = "front_head_rgb_camera"
    
    galbot_interface = GalbotInterface(galbot_interface_config, synthnova_physics_simulator)
    galbot_interface.initialize()

    # Capture images
    while True:
        rgb_data = galbot_interface.front_head_camera.get_rgb_data()
        cv2.imshow("Front Head Camera", rgb_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

#### Wrist Cameras

Capture images from wrist-mounted cameras.

**Files:** 
- `examples/galbot_interface_examples/sensors/left_wrist_camera.py`
- `examples/galbot_interface_examples/sensors/right_wrist_camera.py`

Similar setup to front head camera, but configured for wrist camera positions.

## IOAI Environment Examples

Advanced environments for AI training and testing.

### Navigation Environment

Environment for training navigation algorithms.

**File:** `examples/ioai_examples/ioai_nav_env.py`

```python
from physics_simulator import PhysicsSimulator
from synthnova_config import PhysicsSimulatorConfig, RobotConfig, MujocoConfig
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
from pathlib import Path
import numpy as np

class OlympicNavEnv:
    def __init__(self, headless=False):
        self.simulator = None
        self.robot = None
        self.interface = None
        
        # Path planning setup
        self.planner = AStarPathPlanner(grid_size=1, obstacle_radius=0.5)
        self.path_follower = BasicPathFollower(velocity=0.8)
        
        self._setup_simulator(headless)
        self._setup_interface()
        self._init_pose()

    def _setup_simulator(self, headless):
        """Initialize the physics simulator with basic configuration."""
        sim_config = PhysicsSimulatorConfig(
            mujoco_config=MujocoConfig(headless=headless)
        )
        
        self.simulator = PhysicsSimulator(sim_config)
        self.simulator.add_default_scene()

        # Add Galbot Charlie robot
        robot_config = RobotConfig(
            prim_path="/World/Galbot",
            name="galbot_one_charlie",
            mjcf_path=Path()
                .joinpath(self.simulator.synthnova_assets_directory)
                .joinpath("synthnova_assets")
                .joinpath("robots")
                .joinpath("galbot_one_charlie_description")
                .joinpath("galbot_one_charlie.xml"),
            position=[0, 0, 0],
            orientation=[0, 0, 0, 1]
        )
        self.robot_path = self.simulator.add_robot(robot_config)
        self.simulator.initialize()

    def run_episode(self):
        """Run one navigation episode."""
        while not self.is_goal_reached():
            # Get current state
            current_pos, current_vel = self._get_current_state()
            
            # Compute next action
            action = self.path_follower.compute_action(current_pos, self.path)
            
            # Apply action
            self.interface.chassis.set_joint_velocities(action)
            
            # Step simulation
            self.simulator.step()
```

### Grasp Environment

Environment for training grasping and manipulation tasks.

**File:** `examples/ioai_examples/ioai_grasp_env.py`

```python
class IoaiGraspEnv:
    def __init__(self, headless=False):
        self._setup_simulator(headless)
        self._setup_robot_interface()
        self._setup_objects()

    def _setup_objects(self):
        """Add objects for grasping tasks."""
        # Add various objects like mugs, tools, etc.
        mug_config = MeshConfig(
            prim_path="/World/mug",
            name="mug",
            mjcf_path=Path()
                .joinpath(self.simulator.synthnova_assets_directory)
                .joinpath("synthnova_assets")
                .joinpath("objects")
                .joinpath("mug")
                .joinpath("mug.xml"),
            position=[0.5, 0, 1],
            orientation=[0, 0, 0, 1]
        )
        self.simulator.add_object(mug_config)

    def execute_grasp(self, target_position):
        """Execute a grasping action."""
        # Move arm to target
        self.interface.left_arm.set_joint_positions(target_position)
        
        # Close gripper
        self.interface.left_gripper.close()
        
        # Lift object
        lift_position = target_position.copy()
        lift_position[2] += 0.1  # Lift 10cm
        self.interface.left_arm.set_joint_positions(lift_position)
```

## YOLO Segmentation Examples

Examples for computer vision and object detection integration.

### Object Detection and Segmentation

Train and use YOLO models for object segmentation.

**File:** `examples/yolo_seg_examples/predict.py`

```python
from ultralytics import YOLO
import cv2
import numpy as np

def main():
    # Load trained model
    model = YOLO("assets/yolo_seg/best.pt")
    
    # Load test image
    image = cv2.imread("assets/yolo_seg/test_image.png")
    
    # Run prediction
    results = model(image)
    
    # Draw predictions
    output_img = draw_predictions(image, results[0])
    
    # Display results
    cv2.imshow("YOLO Segmentation", output_img)
    cv2.waitKey(0)

def draw_predictions(image, result):
    """Draw bounding boxes, masks, and labels on the image."""
    output_img = image.copy()
    
    if result.boxes is None:
        return output_img
    
    # Get prediction data
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    
    # Draw masks if available
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            # Create colored overlay
            color = np.random.randint(0, 255, 3).tolist()
            colored_overlay = np.zeros_like(output_img)
            colored_overlay[mask > 0.5] = color
            output_img = cv2.addWeighted(output_img, 1, colored_overlay, 0.3, 0)
    
    return output_img
```

### Training YOLO Models

Train custom YOLO models for specific objects.

**File:** `examples/yolo_seg_examples/train.py`

```python
from ultralytics import YOLO
import yaml

def main():
    # Load a model
    model = YOLO('yolov8n-seg.pt')  # Load a pretrained model
    
    # Train the model
    results = model.train(
        data='dataset.yaml',  # Path to dataset config
        epochs=100,
        imgsz=640,
        device='gpu',
        batch=16
    )
    
    # Validate the model
    metrics = model.val()
    
    # Export the model
    model.export(format='onnx')
```

## Scenario Management

Examples for importing and exporting simulation scenarios.

### Exporting Scenarios

Save simulation configurations for later use.

**File:** `examples/import_and_export_scenario_config/export_scenario_from_config.py`

```python
from physics_simulator import PhysicsSimulator
from synthnova_config import PhysicsSimulatorConfig, RobotConfig, CuboidConfig

def main():
    # Setup simulation with objects and robots
    my_config = PhysicsSimulatorConfig()
    simulator = PhysicsSimulator(my_config)
    
    # Add entities
    simulator.add_default_scene()
    
    # Add robot
    robot_config = RobotConfig(...)
    simulator.add_robot(robot_config)
    
    # Add objects
    cube_config = CuboidConfig(...)
    simulator.add_object(cube_config)
    
    # Export scenario
    simulator.export_scenario(
        file_path="my_scenario.json",
        scenario_name="Demo Scenario",
        scenario_description="Example scenario with robot and objects"
    )
```

### Importing Scenarios

Load previously saved simulation configurations.

**File:** `examples/import_and_export_scenario_config/import_scenario_from_config.py`

```python
from physics_simulator import PhysicsSimulator
from synthnova_config import PhysicsSimulatorConfig, ScenarioConfig

def main():
    # Create simulator
    my_config = PhysicsSimulatorConfig()
    simulator = PhysicsSimulator(my_config)
    
    # Import scenario
    scenario_config = ScenarioConfig(file_path="my_scenario.json")
    simulator.import_scenario(scenario_config)
    
    # Initialize and run
    simulator.initialize()
    simulator.loop()
```

## Common Patterns

### Error Handling

```python
try:
    simulator.initialize()
    galbot_interface.initialize()
    
    # Your simulation code here
    simulator.loop()
    
except RuntimeError as e:
    print(f"Simulation error: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
finally:
    simulator.close()
```

### Trajectory Execution

```python
from physics_simulator.utils.data_types import JointTrajectory
import numpy as np

# Create trajectory
start_pos = galbot_interface.left_arm.get_joint_positions()
end_pos = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
positions = np.linspace(start_pos, end_pos, 100)

trajectory = JointTrajectory(
    positions=positions.tolist(),
    times=np.linspace(0, 5, 100).tolist()  # 5 second trajectory
)

# Execute trajectory
galbot_interface.left_arm.follow_trajectory(trajectory)
```

### Multi-Module Control

```python
# Enable multiple modules
config.modules_manager.enabled_modules = [
    "chassis", "left_arm", "right_arm", 
    "left_gripper", "right_gripper", "head"
]

# Configure all modules
config.chassis.joint_names = [...]
config.left_arm.joint_names = [...]
config.right_arm.joint_names = [...]

# Initialize interface
galbot_interface = GalbotInterface(config, simulator)
galbot_interface.initialize()

# Coordinated control
galbot_interface.chassis.set_joint_velocities([0.1, 0, 0])
galbot_interface.left_arm.set_joint_positions([...])
galbot_interface.right_arm.set_joint_positions([...])
``` 
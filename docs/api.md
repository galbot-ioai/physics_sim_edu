# API Reference

This section provides comprehensive documentation for all public classes and methods in the Physics Simulator.

## Core Simulator

### PhysicsSimulator (MujocoSimulator)

The main simulation class that provides the primary interface to the MuJoCo physics engine.

**Import:**

```python
from physics_simulator import PhysicsSimulator
from synthnova_config import PhysicsSimulatorConfig
```

**Initialization:**

```python
config = PhysicsSimulatorConfig()
sim = PhysicsSimulator(config)
```

#### Core Simulation Methods

**add_default_scene()**
Add the default ground plane to the simulation.

```python
sim.add_default_scene()
```

**add_object(config)**
Add an object to the simulation using a configuration object.

- `config`: Object configuration (e.g., CuboidConfig, MeshConfig)
- Returns: Path of the added object

```python
from synthnova_config import CuboidConfig
cube_config = CuboidConfig(
    prim_path="/World/cube_1",
    position=[0, 0, 1],
    scale=[1, 1, 1],
    color=[1.0, 0.0, 0.0]
)
cube_path = sim.add_object(cube_config)
```

**add_robot(config)**
Add a robot to the simulation using a robot configuration.

- `config`: RobotConfig object specifying robot parameters
- Returns: Path of the added robot

```python
from synthnova_config import RobotConfig
from pathlib import Path

robot_config = RobotConfig(
    prim_path="/World/Galbot",
    name="galbot_one_foxtrot",
    mjcf_path=Path("assets/robots/galbot_one_foxtrot.xml"),
    position=[0, 0, 0],
    orientation=[0, 0, 0, 1]
)
robot_path = sim.add_robot(robot_config)
```

**initialize()**
Initialize the MuJoCo model, data structures, and viewer. Must be called before stepping the simulation.

```python
sim.initialize()
```

**play()**
Start the simulation (unpause).

```python
sim.play()
```

**pause()**
Pause the simulation.

```python
sim.pause()
```

**loop()**
Run the main simulation loop with viewer.

```python
sim.loop()
```

**step()**
Step the simulation forward by one timestep.

```python
sim.step()
```

**close()**
Clean up and close the simulation.

```python
sim.close()
```

#### Properties

**root_prim_path**
The root path for simulation entities.

**synthnova_assets_directory**
Path to the SynthNova assets directory.

---

## Galbot Interface

The Galbot Interface provides high-level control for modular robot systems.

### GalbotInterface

**Import:**

```python
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
```

**Initialization:**

```python
config = GalbotInterfaceConfig()
# Configure enabled modules
config.modules_manager.enabled_modules = ["chassis", "left_arm", "head"]
interface = GalbotInterface(config, sim)
interface.initialize()
```

#### Available Modules

**Chassis**
Mobile base control for omnidirectional movement.

```python
# Configure chassis joints
config.chassis.joint_names = [
    "robot_name/mobile_forward_joint",
    "robot_name/mobile_side_joint", 
    "robot_name/mobile_yaw_joint"
]

# Control chassis
interface.chassis.set_joint_velocities([forward_vel, side_vel, yaw_vel])
```

**Arms (LeftArm, RightArm)**
Robotic arm control with joint position and velocity commands.

```python
# Configure arm joints
config.left_arm.joint_names = [
    "robot_name/left_arm_joint_1",
    "robot_name/left_arm_joint_2",
    # ... more joints
]

# Control arm
target_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
interface.left_arm.set_joint_positions(target_positions)
```

**Grippers (LeftGripper, RightGripper)**
Gripper control for grasping operations.

```python
# Configure gripper joints
config.left_gripper.joint_names = [
    "robot_name/left_gripper_joint_1",
    "robot_name/left_gripper_joint_2"
]

# Control gripper
interface.left_gripper.close()
interface.left_gripper.open()
```

**Head**
Head control for pan/tilt movements.

```python
# Configure head joints
config.head.joint_names = [
    "robot_name/head_joint_1",
    "robot_name/head_joint_2"
]

# Control head
interface.head.set_joint_positions([pan_angle, tilt_angle])
```

**Legs**
Leg control for locomotion.

```python
# Configure leg joints
config.leg.joint_names = [
    "robot_name/leg_joint_1",
    "robot_name/leg_joint_2",
    # ... more joints
]

# Control legs
interface.leg.set_joint_positions(target_positions)
```

**Cameras**
Camera modules for sensor data collection.

- `FrontHeadCamera`: Front-facing head camera
- `LeftWristCamera`: Left wrist camera  
- `RightWristCamera`: Right wrist camera

```python
# Configure camera
config.front_head_camera.camera_name = "front_head_rgb_camera"

# Get camera data
rgb_data = interface.front_head_camera.get_rgb_data()
```

---

## Sensors

### RGB Camera

**Import:**

```python
from physics_simulator.sensor import MujocoRgbCamera
```

**Usage:**

```python
from synthnova_config import RgbCameraConfig

camera_config = RgbCameraConfig(
    name="my_camera",
    prim_path="/World/camera",
    translation=[0, 0, 1],
    rotation=[0, 0, 0, 1],
    resolution=[640, 480]
)
sim.add_camera(camera_config)
```

### Depth Camera

**Import:**

```python
from physics_simulator.sensor import MujocoDepthCamera
```

**Usage:**

```python
from synthnova_config import DepthCameraConfig

depth_config = DepthCameraConfig(
    name="depth_camera",
    prim_path="/World/depth_camera",
    translation=[0, 0, 1],
    resolution=[640, 480]
)
sim.add_camera(depth_config)
```

---

## Objects and Robots

### MujocoObject

Base class for all objects in the simulation.

**Import:**

```python
from physics_simulator.object import MujocoObject, MujocoXMLObject, PrimitiveObject
```

### MujocoRobot

Robot representation in the simulation.

**Import:**

```python
from physics_simulator.robot import MujocoRobot
```

---

## Utilities

### Data Types

**Import:**

```python
from physics_simulator.utils.data_types import JointTrajectory
```

**JointTrajectory**
Represents a trajectory of joint positions over time.

```python
trajectory = JointTrajectory(
    joint_names=["joint_1", "joint_2"],
    positions=[[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]],
    times=[0.0, 1.0, 2.0]
)
```

### Control Utils

**Import:**

```python
from physics_simulator.utils.control_utils import *
```

Provides utility functions for robot control and trajectory planning.

### File Processing

**Import:**

```python
from physics_simulator.utils.file_process import *
```

Utilities for file handling and asset management.

### Error Handling

**Import:**

```python
from physics_simulator.utils.errors import *
```

Custom exception classes for physics simulator errors.

---

## Configuration Classes

All configuration classes are imported from `synthnova_config`:

```python
from synthnova_config import (
    PhysicsSimulatorConfig,
    RobotConfig,
    CuboidConfig, 
    MeshConfig,
    RgbCameraConfig,
    DepthCameraConfig,
    MujocoConfig
)
```

### PhysicsSimulatorConfig

Main configuration for the physics simulator.

```python
config = PhysicsSimulatorConfig(
    mujoco_config=MujocoConfig(headless=False)
)
```

### RobotConfig

Configuration for adding robots to the simulation.

```python
robot_config = RobotConfig(
    prim_path="/World/Robot",
    name="robot_name",
    mjcf_path=Path("path/to/robot.xml"),
    position=[0, 0, 0],
    orientation=[0, 0, 0, 1]
)
```

### Object Configurations

**CuboidConfig**
Configuration for primitive cuboid objects.

```python
cube_config = CuboidConfig(
    prim_path="/World/cube",
    position=[0, 0, 1],
    orientation=[0, 0, 0, 1],
    scale=[1, 1, 1],
    color=[1.0, 0.0, 0.0],
    interaction_type="dynamic"  # or "static"
)
```

**MeshConfig**
Configuration for mesh objects.

```python
mesh_config = MeshConfig(
    prim_path="/World/mesh",
    mesh_path=Path("path/to/mesh.obj"),
    position=[0, 0, 0],
    orientation=[0, 0, 0, 1],
    scale=[1, 1, 1]
)
```

---

## Error Handling

Common exceptions that may be raised:

- `RuntimeError`: General simulation errors
- `ValueError`: Invalid configuration parameters  
- `FileNotFoundError`: Missing asset files
- `KeyError`: Entity not found at specified path

**Best Practices:**

```python
try:
    sim.initialize()
    # Simulation code
except RuntimeError as e:
    print(f"Simulation error: {e}")
finally:
    sim.close()
``` 
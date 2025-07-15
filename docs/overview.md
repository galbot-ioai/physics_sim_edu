# Overview

Physics Simulator provides a MuJoCo-based physics simulation environment for robotics applications.

## Core Components

**PhysicsSimulator (MujocoSimulator)**
Main simulation class that provides physics simulation, robot control, and sensor management.

**GalbotInterface**
High-level interface for controlling robots with modular components:

* **Arms**: `left_arm`, `right_arm` - Joint position and trajectory control
* **Grippers**: `left_gripper`, `right_gripper` - Gripper width control and state
* **Cameras**: `left_wrist_camera`, `right_wrist_camera`, `front_head_camera` - RGB/depth sensing
* **Chassis**: Mobile base control for navigation
* **Head/Leg**: Additional joint control modules

Available modules: `chassis`, `left_arm`, `right_arm`, `leg`, `head`, `left_gripper`, `right_gripper`, `left_wrist_camera`, `right_wrist_camera`, `front_head_camera`

**MujocoRobot**
Low-level robot control interface providing:

* Joint position/velocity control
* Forward kinematics and Jacobian computation
* Robot state monitoring

**Sensors**
Camera system supporting:

* RGB image capture
* Depth sensing and point cloud generation
* Segmentation masks

**Objects**
Support for primitive shapes (boxes, spheres) and mesh objects from MJCF files.

## Key Features

* **Real-time Physics**: MuJoCo engine with accurate dynamics and contact handling
* **Modular Control**: Configure only needed robot modules (arms, grippers, cameras)
* **Sensor Integration**: RGB/depth cameras with real-time data access
* **Asset Management**: Automatic discovery of robot and object assets
* **Configuration System**: Code-based configuration for all simulation elements
* **Physics Callbacks**: Custom behavior injection during simulation 

## Applications
* **Robotics Research**: Simulate and test robot kinematics, dynamics, and control algorithms in a realistic environment.
* **Algorithm Prototyping**: Rapidly prototype and validate motion planning, manipulation, and navigation algorithms.
* **Sensor Data Generation**: Generate synchronized RGB, depth, and segmentation data for computer vision and machine learning tasks.



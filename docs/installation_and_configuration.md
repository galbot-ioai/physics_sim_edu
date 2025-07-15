# Installation and Configuration

## Requirements

* Python 3.10+
* MuJoCo physics engine
* Compatible with Ubuntu 22.04

## Installation

We recommend using a conda environment to isolate dependencies and avoid conflicts:

```bash
# Create a new conda environment with Python 3.10.15
conda create -n synth_nova python=3.10.15

# Activate the environment
conda activate synth_nova
```

**1. Install the package:**

   ```bash
   git clone <repository-url>
   cd physics_simulator
   pip install -e ./synthnova_config
   pip install -e .
   ```

**2. Verify installation:**

   ```python
   from physics_simulator import PhysicsSimulator
   from synthnova_config import PhysicsSimulatorConfig
   
   # Test basic import
   config = PhysicsSimulatorConfig()
   sim = PhysicsSimulator(config)
   print("Installation successful!")
   ```

## Configuration System

The simulator uses a configuration-based architecture with Pydantic models for type validation and serialization. All configuration classes are available from `synthnova_config`.

### Core Configuration Classes

#### PhysicsSimulatorConfig

Main configuration class that orchestrates all simulation components.

```python
from synthnova_config import PhysicsSimulatorConfig, MujocoConfig

config = PhysicsSimulatorConfig(
    mujoco_config=MujocoConfig(
        headless=False,           # Enable/disable GUI
        realtime_sync=True,       # Sync with real time
        timestep=0.002,          # Physics timestep (seconds)
        gravity="0 0 -9.81"      # Gravity vector
    )
)
```

#### MujocoConfig

Physics engine configuration with core simulation parameters.

**Key Parameters:**
- `headless`: Run without GUI (default: False)
- `realtime_sync`: Synchronize with real time (default: False)
- `timestep`: Physics simulation timestep in seconds (default: 0.002)
- `gravity`: Gravitational acceleration vector (default: "0 0 -9.81")
- `integrator`: Numerical integration method ("implicitfast", "implicit", "explicit")
- `solver`: Constraint solver ("Newton", "CG", "PGS")
- `iterations`: Maximum solver iterations (default: 100)

```python
from synthnova_config import MujocoConfig

mujoco_config = MujocoConfig(
    headless=True,
    timestep=0.001,
    gravity="0 0 -9.81",
    solver="Newton",
    iterations=200
)
```

### Entity Configuration

#### RobotConfig

Configuration for adding robots to the simulation.

**Required Parameters:**
- `prim_path`: Unique path in scene graph
- `name`: Robot identifier
- `mjcf_path`: Path to MuJoCo XML file

**Optional Parameters:**
- `position`: Global position [x, y, z]
- `orientation`: Quaternion [x, y, z, w]
- `scale`: Scale factors [x, y, z]

```python
from synthnova_config import RobotConfig
from pathlib import Path

robot_config = RobotConfig(
    prim_path="/World/Robot",
    name="galbot_one_foxtrot",
    mjcf_path=Path("assets/robots/galbot.xml"),
    position=[0, 0, 0],
    orientation=[0, 0, 0, 1]
)
```

#### Object Configurations

**CuboidConfig** - Primitive cuboid objects:

```python
from synthnova_config import CuboidConfig

cube_config = CuboidConfig(
    prim_path="/World/cube",
    position=[1, 1, 1],
    size=0.5,                    # Edge length
    color=[1.0, 0.0, 0.0],      # RGB color
    interaction_type="dynamic"   # "dynamic" or "static"
)
```

**MeshConfig** - Complex mesh objects:

```python
from synthnova_config import MeshConfig

mesh_config = MeshConfig(
    prim_path="/World/object",
    name="custom_object",
    mjcf_path=Path("assets/objects/object.xml"),
    position=[0, 0, 1],
    scale=[1, 1, 1]
)
```

**Other Primitive Objects:**
- `SphereConfig`: Spherical objects
- `CylinderConfig`: Cylindrical objects
- `CapsuleConfig`: Capsule-shaped objects
- `ConeConfig`: Cone-shaped objects

### Sensor Configuration

#### RGB Camera

```python
from synthnova_config import RgbCameraConfig

rgb_config = RgbCameraConfig(
    name="front_camera",
    prim_path="/World/robot/camera",
    translation=[0.1, 0, 0.1],     # Local position
    rotation=[0, 0, 0, 1],          # Local rotation
    resolution=[640, 480],          # Image size
    horizontal_fov=90               # Field of view (degrees)
)
```

#### Depth Camera

```python
from synthnova_config import DepthCameraConfig

depth_config = DepthCameraConfig(
    name="depth_camera",
    prim_path="/World/robot/depth_camera",
    translation=[0.1, 0, 0.1],
    resolution=[640, 480],
    clipping_range=[0.1, 10.0]      # Near/far distances
)
```

### Advanced Configuration

#### Interaction Types

Objects can have different interaction behaviors:
- `"dynamic"`: Participates in physics simulation
- `"static"`: Fixed in place, affects other objects

#### Collision Types

For mesh objects, collision geometry generation:
- `"convex_decomposition"`: High accuracy, better performance
- `"mesh"`: Exact mesh collision (slower)
- `"convex_hull"`: Simple convex approximation

#### Scenario Management

```python
from synthnova_config import ScenarioConfig

# Export current state
sim.export_scenario(
    file_path="my_scenario.json",
    scenario_name="Demo Scene"
)

# Import saved scenario
scenario_config = ScenarioConfig(file_path="my_scenario.json")
sim.import_scenario(scenario_config)
```

### Configuration Best Practices

**1. Use Path objects for file paths:**
```python
from pathlib import Path
mjcf_path = Path("assets/robots/robot.xml")
```

**2. Validate configurations early:**
```python
try:
    config = PhysicsSimulatorConfig(...)
except ValidationError as e:
    print(f"Configuration error: {e}")
```

**3. Use factory defaults when possible:**
```python
# Simple setup with defaults
config = PhysicsSimulatorConfig()
# Customize only what you need
config.mujoco_config.headless = True
```

**4. Combine configurations for complex scenes:**
```python
# Physics settings
physics_config = PhysicsSimulatorConfig(
    mujoco_config=MujocoConfig(timestep=0.001)
)

# Add entities
sim = PhysicsSimulator(physics_config)
sim.add_robot(robot_config)
sim.add_object(cube_config)
```

### Common Configuration Patterns

**Development Setup (with GUI):**
```python
dev_config = PhysicsSimulatorConfig(
    mujoco_config=MujocoConfig(
        headless=False,
        realtime_sync=True
    )
)
```

**Production Setup (headless):**
```python
prod_config = PhysicsSimulatorConfig(
    mujoco_config=MujocoConfig(
        headless=True,
        realtime_sync=False,
        timestep=0.001
    )
)
```

**High-Performance Setup:**
```python
performance_config = PhysicsSimulatorConfig(
    mujoco_config=MujocoConfig(
        headless=True,
        solver="Newton",
        iterations=50,
        tolerance=1e-6
    )
)
```

## Next Steps

* See [Examples](examples.md) for usage examples
* Check [API Reference](api.md) for complete API reference 
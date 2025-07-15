# SynthNova Physics Simulator

Physics Simulator is a MuJoCo-based simulation package that provides accurate physics simulations, high-level robot control interfaces, and comprehensive sensor systems.

**Key Features:**

* MuJoCo physics engine integration
* High-level GalbotInterface for modular robot control
* RGB/depth cameras and sensor simulation
* Support for robots, objects, and custom physics callbacks

**Quick Start:**

```python
from physics_simulator import PhysicsSimulator
from synthnova_config import PhysicsSimulatorConfig

# Create and initialize simulator
config = PhysicsSimulatorConfig()
sim = PhysicsSimulator(config)
sim.add_default_scene()
sim.initialize()

# Run simulation
sim.loop()
```

## Documentation

* [📄 Overview](overview.md)
* [🛠️ Installation and Configuration](installation_and_configuration.md)
* [💡 Examples](examples.md)
* [🖥️ API Reference](api.md)
* [🔧 Troubleshooting](troubleshooting.md)
* [📜 License](license.md)

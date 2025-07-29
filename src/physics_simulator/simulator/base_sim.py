#####################################################################################
# Copyright (c) 2023-2025 Galbot. All Rights Reserved.
#
# This software contains confidential and proprietary information of Galbot, Inc.
# ("Confidential Information"). You shall not disclose such Confidential Information
# and shall use it only in accordance with the terms of the license agreement you
# entered into with Galbot, Inc.
#
# UNAUTHORIZED COPYING, USE, OR DISTRIBUTION OF THIS SOFTWARE, OR ANY PORTION OR
# DERIVATIVE THEREOF, IS STRICTLY PROHIBITED. IF YOU HAVE RECEIVED THIS SOFTWARE IN
# ERROR, PLEASE NOTIFY GALBOT, INC. IMMEDIATELY AND DELETE IT FROM YOUR SYSTEM.
#####################################################################################
#          _____             _   _       _   _
#         / ____|           | | | |     | \ | |
#        | (___  _   _ _ __ | |_| |__   |  \| | _____   ____ _
#         \___ \| | | | '_ \| __| '_ \  | . ` |/ _ \ \ / / _` |
#         ____) | |_| | | | | |_| | | | | |\  | (_) \ V / (_| |
#        |_____/ \__, |_| |_|\__|_| |_| |_| \_|\___/ \_/ \__,_|
#                 __/ |
#                |___/
#
#
#####################################################################################
#
# Description: Base Sim Class for SynthNova Physics Simulator
# Author: Chenyu Cao@Galbot
# Date: 2025-04-27
#
#####################################################################################
from auro_utils import Logger
from abc import ABC, abstractmethod
from synthnova_config import (
    PhysicsSimulatorConfig,
    LoggerConfig,
    RobotConfig,
    ObjectConfig,
    SensorConfig,
    MujocoConfig,
)


class BaseSim(ABC):
    """Base simulator class that defines common interfaces and basic functionality for simulators.

    This abstract base class provides the interface for all simulator implementations.
    All concrete simulator classes must implement all abstract methods defined here.
    """

    def __init__(self, physics_simulator_config: PhysicsSimulatorConfig):
        """Initialize the base simulator.

        Args:
            physics_simulator_config: Physics simulator configuration object containing all necessary settings
        """
        pass

    @abstractmethod
    def _load_logger(self, logger_config: LoggerConfig) -> Logger:
        """Load and configure the logging system.

        Args:
            logger_config: Logger configuration object

        Returns:
            Logger: Configured logger instance
        """
        pass

    @abstractmethod
    def _load_simulator(self, mujoco_config: MujocoConfig):
        """Load the specific simulator implementation.

        Args:
            mujoco_config: Simulator configuration object
        """
        pass

    @abstractmethod
    def step(self):
        """Execute one simulation time step."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the simulator is currently running.

        Returns:
            bool: Simulator running status
        """
        pass

    @abstractmethod
    def loop(self):
        """Main simulation loop."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the simulator to its initial state."""
        pass

    @abstractmethod
    def close(self):
        """Close the simulator and release resources."""
        pass

    @abstractmethod
    def add_default_scene(self):
        """Add default scene to the simulator."""
        pass

    @abstractmethod
    def add_robot(self, robot_config: RobotConfig):
        """Add a robot to the simulator.

        Args:
            robot_config: Robot configuration object
        """
        pass

    @abstractmethod
    def get_robot(self, prim_path: str):
        """Retrieve a robot by its path.

        Args:
            prim_path: Path identifier for the robot

        Returns:
            Robot object
        """
        pass

    @abstractmethod
    def remove_robot(self, prim_path: str):
        """Remove a robot from the simulator.

        Args:
            prim_path: Path identifier for the robot
        """
        pass

    @abstractmethod
    def add_object(self, object_config: ObjectConfig):
        """Add an object to the simulator.

        Args:
            object_config: Object configuration object
        """
        pass

    @abstractmethod
    def get_object(self, prim_path: str):
        """Retrieve an object by its path.

        Args:
            prim_path: Path identifier for the object

        Returns:
            Object instance
        """
        pass

    @abstractmethod
    def remove_object(self, prim_path: str):
        """Remove an object from the simulator.

        Args:
            prim_path: Path identifier for the object
        """
        pass

    @abstractmethod
    def add_sensor(self, sensor_config: SensorConfig):
        """Add a sensor to the simulator.

        Args:
            sensor_config: Sensor configuration object
        """
        pass

    @abstractmethod
    def get_sensor(self, prim_path: str):
        """Retrieve a sensor by its path.

        Args:
            prim_path: Path identifier for the sensor

        Returns:
            Sensor object
        """
        pass

    @abstractmethod
    def remove_sensor(self, prim_path: str):
        """Remove a sensor from the simulator.

        Args:
            prim_path: Path identifier for the sensor
        """
        pass

    @abstractmethod
    def get_robot_state(self, prim_path: str):
        """Get the state of a specific robot.

        Args:
            prim_path: Path identifier for the robot

        Returns:
            Robot state information
        """
        pass

    @abstractmethod
    def get_object_state(self, prim_path: str):
        """Get the state of a specific object.

        Args:
            prim_path: Path identifier for the object

        Returns:
            Object state information
        """
        pass

    @abstractmethod
    def get_sensor_state(self, prim_path: str):
        """Get the state of a specific sensor.

        Args:
            prim_path: Path identifier for the sensor

        Returns:
            Sensor state information
        """
        pass

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
#####################################################################################
#
# Description: Robot interface for robot control
# Author: Chenyu Cao@Galbot
# Date: 2025-05-01
#
#####################################################################################

from physics_simulator.galbot_interface.config import GalbotInterfaceConfig
from physics_simulator import PhysicsSimulator
from typing import Dict, Any


class Robot:
    """
    Interface for controlling the robot as a whole entity.
    """
    def __init__(
        self, galbot_interface_config: GalbotInterfaceConfig, simulator: PhysicsSimulator
    ):
        self.galbot_interface_config = galbot_interface_config
        self.simulator = simulator
        self.logger = simulator.logger
        self.robot_instance = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the robot interface"""
        self.robot_instance = self.simulator.get_robot(
            self.galbot_interface_config.robot.prim_path
        )
        self.initialized = True
        self.logger.log_debug(f"{self.__class__.__name__} initialized")
        
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the robot"""
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized")
        with self.simulator.lock:
            return self.simulator.get_robot_state(self.galbot_interface_config.robot.prim_path) 
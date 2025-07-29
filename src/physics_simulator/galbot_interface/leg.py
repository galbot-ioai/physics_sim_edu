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
# Description: Leg interface for robot control
# Author: Chenyu Cao@Galbot
# Date: 2025-05-01
#
#####################################################################################

from physics_simulator.galbot_interface.config import GalbotInterfaceConfig
from physics_simulator import PhysicsSimulator
from physics_simulator.galbot_interface.basic_part import BasicLimb


class Leg(BasicLimb):
    """
    Interface for controlling the leg of the robot.
    """
    def __init__(
        self, galbot_interface_config: GalbotInterfaceConfig, simulator: PhysicsSimulator
    ):
        super().__init__(galbot_interface_config, simulator)
        self.module_name = "leg" 
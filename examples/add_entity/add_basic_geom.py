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
# Description: Example of adding basic geom objects to the simulation
# Author: Chenyu Cao, Herman Ye@Galbot
# Date: 2025-04-02
#
#####################################################################################

from synthnova_config import PhysicsSimulatorConfig, CuboidConfig
import os
from physics_simulator import PhysicsSimulator

def main():
    # Instantiate the simulator
    my_config = PhysicsSimulatorConfig()
    synthnova_physics_simulator = PhysicsSimulator(my_config)

    # Add default ground plane if you need
    synthnova_physics_simulator.add_default_scene()

    # Add cube
    cube_1_config = CuboidConfig(
        prim_path=os.path.join(synthnova_physics_simulator.root_prim_path, "cube_1"),
        position=[2, 2, 2],
        orientation=[0, 0, 0, 1],
        scale=[1, 1, 1],
        color=[1.0, 0.0, 0.0],
    )
    cube_1_path = synthnova_physics_simulator.add_object(cube_1_config)

    # Add cube
    cube_2_config = CuboidConfig(
        prim_path=os.path.join(synthnova_physics_simulator.root_prim_path, "cube_2"),
        position=[0, 0, 10],
        orientation=[0, 0, 0, 1],
        scale=[1, 1, 1],
        color=[0.0, 1.0, 0.0],
        interaction_type="static",
    )
    cube_2_path = synthnova_physics_simulator.add_object(cube_2_config)

    # Add cube
    cube_3_config = CuboidConfig(
        prim_path=os.path.join(synthnova_physics_simulator.root_prim_path, "cube_3"),
        position=[-2, -2, 2],
        orientation=[0, 0, 0, 1],
        scale=[1, 1, 1],
        color=[0.0, 0.0, 1.0],
    )
    cube_3_path = synthnova_physics_simulator.add_object(cube_3_config)

    # Play the simulator
    synthnova_physics_simulator.play()

    # Run the display loop
    synthnova_physics_simulator.loop()

    # Close the simulator
    synthnova_physics_simulator.close()


if __name__ == "__main__":
    main()

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
# Description: Import a scenario from a config file
# Author: Chenyu Cao, Herman Ye@Galbot
# Date: 2025-04-27
#
#####################################################################################

from physics_simulator import PhysicsSimulator
from synthnova_config import PhysicsSimulatorConfig, ScenarioConfig
from pathlib import Path


def main():
    # Create sim config
    my_config = PhysicsSimulatorConfig()
    # Load the scenario config from a file (please make sure the file is in the assets folder)
    my_config.scenario_config = ScenarioConfig.load_from_file(
        Path()
        .joinpath(PhysicsSimulator.get_root_directory())
        .joinpath("assets") 
        .joinpath("exported_scenario.json")
    )
    # Instantiate the simulator
    synthnova_physics_simulator = PhysicsSimulator(my_config)

    # Initialize the simulator
    synthnova_physics_simulator.initialize()

    # Run the step loop
    synthnova_physics_simulator.loop()

    # Close the simulator
    synthnova_physics_simulator.close()


if __name__ == "__main__":
    main()

######################################################################################
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
######################################################################################
#
#  ██████  ██    ██ ██    ██ ████████ ██     ██ ██    ██  ███████  ██     ██    ███
# ██    ██  ██  ██  ███   ██    ██    ██     ██ ███   ██ ██     ██ ██     ██   ██ ██
# ██         ████   ████  ██    ██    ██     ██ ████  ██ ██     ██ ██     ██  ██   ██
#  ██████     ██    ██ ██ ██    ██    █████████ ██ ██ ██ ██     ██ ██     ██ ██     ██
#       ██    ██    ██  ████    ██    ██     ██ ██  ████ ██     ██  ██   ██  █████████
# ██    ██    ██    ██   ███    ██    ██     ██ ██   ███ ██     ██   ██ ██   ██     ██
#  ██████     ██    ██    ██    ██    ██     ██ ██    ██  ███████     ███    ██     ██
#
######################################################################################
#
# Description: IOAI Competition Main Entry Point
# Author: Chenyu Cao, Herman Ye@Galbot
#
# IMPORTANT NOTICE:
# This file serves as the main entry point for the IOAI competition. The provided
# implementation is a baseline that demonstrates the complete pipeline structure.
# Participants are encouraged to replace, rewrite, or extend any part of this code
# to develop their own optimal solutions for competitive advantage.
#
######################################################################################

# Import IOAI environment
from ioai_env import IOAIEnv

# Import object pose estimators
from object_pose_estimator import (
    BaseObjectPoseEstimator,
    GroundTruthObjectPoseEstimator,
    YoloSegObjectPoseEstimator,
)

# Import grasp pose predictors
from grasp_pose_predictor import BaseGraspPosePredictor, OfficialGraspPosePredictor

# Import motion planners
from motion_planner import BaseMotionPlanner, InterpolationMotionPlanner

# Import path planners
from path_planner import BasePathPlanner, InterpolationPathPlanner

# Import robot state machine
from robot_state_machine import IOAIRobotStateMachine

import os
from pathlib import Path


def main():
    """
    Main entry point for the IOAI competition.

    This function initializes and configures the complete robotic system pipeline,
    including the simulation environment, perception modules, planning components,
    and control systems. It sets up the state machine that orchestrates the entire
    robotic operation workflow.
    """

    # Initialize the simulation environment
    # NOTE: `standard_scenario=True` will use the standard scenario.
    env = IOAIEnv(headless=False, standard_scenario=True)

    # Initialize object pose estimator using ground truth data
    object_pose_estimator = GroundTruthObjectPoseEstimator(env)
    # yolo_seg_model_path = os.path.join(Path(__file__).parent, "yolo_seg/ckpts/ioai.pt")
    # object_pose_estimator = YoloSegObjectPoseEstimator(env, yolo_seg_model_path)

    # Initialize grasp pose predictor using official implementation
    grasp_pose_predictor = OfficialGraspPosePredictor(env)

    # Initialize arm motion planner using interpolation-based approach
    motion_planner = InterpolationMotionPlanner(env)

    # Initialize chassis path planner using interpolation-based approach
    path_planner = InterpolationPathPlanner(env)

    # Initialize the robot state machine with all components
    robot_state_machine = IOAIRobotStateMachine(
        env=env,
        object_pose_estimator=object_pose_estimator,
        grasp_pose_predictor=grasp_pose_predictor,
        motion_planner=motion_planner,
        path_planner=path_planner,
    )

    def state_machine_callback():
        """
        Physics step callback function that executes the robot state machine.

        This function is called at each physics simulation step and triggers
        the execution of the robot state machine, which manages the coordination
        between perception, planning, and control activities.
        """
        robot_state_machine.execute()

    # Register the callback function to be executed at each physics step
    env.simulator.add_physics_callback("state_machine_callback", state_machine_callback)

    # Start the simulation and run until completion
    env.run()


if __name__ == "__main__":
    main()

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
# Description: IOAI Competition Main Entry Point with SimpleStateMachine
# Author: Chenyu Cao, Herman Ye@Galbot
#
# COMPETITION GUIDELINES:
# - Replace baseline implementations with your own algorithms for higher scores
# - Focus on: object pose estimation, grasp planning, motion planning, and state management
# - Performance and robustness are key evaluation criteria
#
######################################################################################

from ioai_env import IOAIEnv
from object_pose_estimator import (
    BaseObjectPoseEstimator,
    GroundTruthObjectPoseEstimator,
    YoloSegObjectPoseEstimator,
)
from grasp_pose_predictor import BaseGraspPosePredictor, OfficialGraspPosePredictor
from motion_planner import BaseMotionPlanner, InterpolationMotionPlanner
from path_planner import BasePathPlanner, InterpolationPathPlanner
from robot_state_machine import IOAIRobotStateMachine

import os
from pathlib import Path

current_dir = Path(__file__).parent


def main():
    """
    Main entry point for IOAI competition using SimpleStateMachine.
    
    COMPETITION FOCUS AREAS:
    1. Object Pose Estimation: Implement vision-based methods for better accuracy
    2. Grasp Planning: Develop robust grasp pose prediction algorithms
    3. Motion Planning: Optimize trajectory generation for efficiency and safety
    4. State Management: Design intelligent state transitions and error handling
    """
    # Initialize simulation environment
    env = IOAIEnv(headless=False)

    # TODO: Object Pose Estimation
    # Replace with your vision-based pose estimation implementation
    # Baseline: GroundTruthObjectPoseEstimator (perfect but unrealistic)
    # Target: Robust pose estimation from camera data
    # object_pose_estimator = GroundTruthObjectPoseEstimator(env)
    yolo_seg_model_path = os.path.join(current_dir, "yolo_seg/ckpts/cotrain_all_class_0731_1.pt")
    object_pose_estimator = YoloSegObjectPoseEstimator(env, yolo_seg_model_path)

    # TODO: Grasp Pose Prediction  
    # Replace with your grasp planning algorithm
    # Baseline: OfficialGraspPosePredictor (basic implementation)
    # Target: High-success-rate grasp pose generation
    grasp_pose_predictor = OfficialGraspPosePredictor(env)

    # TODO: Motion Planning
    # Replace with your motion planning solution
    # Baseline: InterpolationMotionPlanner (simple interpolation)
    # Target: Collision-free, efficient trajectory planning
    motion_planner = InterpolationMotionPlanner(env)

    # TODO: Path Planning
    # Replace with your path planning solution
    # Baseline: InterpolationPathPlanner (simple interpolation)
    # Target: Collision-free, efficient path planning
    path_planner = InterpolationPathPlanner(env)

    # TODO: Robot State Mahine
    # Baseline: OfficialRobotStateMachine (basic state transitions)
    # Target: Intelligent task execution with error recovery
    robot_state_machine = IOAIRobotStateMachine(
        env=env,
        object_pose_estimator=object_pose_estimator,
        grasp_pose_predictor=grasp_pose_predictor,
        motion_planner=motion_planner,
        path_planner=path_planner
    )

    def ioai_main_callback():
        """Main simulation loop - executes state machine each physics step"""
        robot_state_machine.execute()

    # Register callback and run simulation
    env.simulator.add_physics_callback("ioai_main_callback", ioai_main_callback)
    env.run()


if __name__ == "__main__":
    main()

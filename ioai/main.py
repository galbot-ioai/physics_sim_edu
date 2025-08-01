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
)
from grasp_pose_predictor import BaseGraspPosePredictor, OfficialGraspPosePredictor
from motion_planner import BaseMotionPlanner, InterpolationMotionPlanner
from path_planner import BasePathPlanner, InterpolationPathPlanner
from physics_simulator.utils.state_machine import SimpleStateMachine
import numpy as np
import time
import math


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
    object_pose_estimator = GroundTruthObjectPoseEstimator(env)

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

    # Current: AStarPathPlanner (A* algorithm with interpolation)
    path_planner = InterpolationPathPlanner(env)

    # Initialize SimpleStateMachine
    state_machine = SimpleStateMachine(max_states=-1)
    state_machine.state_first_entry = True
    state_machine.object_name = "cube"
    state_machine.object_pose = None
    state_machine.grasp_pose = None
    state_machine.bin_pose = None
    state_machine.objects_processed = 0
    state_machine.total_objects = 4  # cube, power_drill, extrusion, toy
    state_machine.wait_start_time = None

    # Wire components to state machine
    state_machine.pose_estimator = object_pose_estimator
    state_machine.grasp_predictor = grasp_pose_predictor
    state_machine.motion_planner = motion_planner
    state_machine.path_planner = path_planner

    # Helper function to check motion completion
    def is_callback_complete(callback_name):
        return not env.simulator.physics_callback_exists(callback_name)

    # Helper function for joint interpolation
    def move_joints_to_target(module, target_positions, steps=500):
        from physics_simulator.utils.data_types import JointTrajectory
        start_positions = module.get_joint_positions()
        joint_positions = np.linspace(start_positions, target_positions, steps)
        joint_trajectory = JointTrajectory(positions=np.array(joint_positions))
        module.follow_trajectory(joint_trajectory)

    # Phase 1: Initial Setup and Navigation
    def initialize_robot_safe_pose_state():
        if state_machine.state_first_entry:
            state_machine.wait_start_time = time.time()
            poses = {
                env.interface.head: [0.0, 0.26],
                env.interface.leg: [0.0821758285164833, 0.6340972781181335,0.5227039456367493, -0.00001198422432935331],
                env.interface.left_arm: [2.00, -1.60, -0.60, -1.70, 0.00, -0.80, 0.00],
                env.interface.right_arm: [-2.00, 1.60, 0.60, 1.70, 0.00, 0.80, 0.00]
            }
            for module, pose in poses.items():
                module.set_joint_positions(pose, immediate=False)
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Initializing robot to safe pose")
        # Wait for 3 seconds
        return time.time() - state_machine.wait_start_time >= 3
    state_machine.add_state("Initialize Robot Safe Pose", initialize_robot_safe_pose_state)

    def navigate_to_table_front_state():
        if state_machine.state_first_entry:
            init_pos = env.robot.get_position()[:2]
            target_pos = [0, -0.3]
            waypoints = state_machine.path_planner.plan_path(init_pos, target_pos, 30)
            env.move_chassis_follow_path(waypoints)
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Navigating to table front")
        return is_callback_complete("follow_path_callback")
    state_machine.add_state("Navigate to Table Front", navigate_to_table_front_state)

    def rotate_to_face_table_state():
        if state_machine.state_first_entry:
            env.move_chassis_rotate(0)
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Rotating to face table")
        return is_callback_complete("rotate_callback")
    state_machine.add_state("Rotate to Face Table", rotate_to_face_table_state)

    def detect_bin_with_head_camera_state():
        if state_machine.state_first_entry:
            # Simulate bin detection with head camera
            state_machine.bin_pose = state_machine.pose_estimator.estimate_pose("bin")
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Detecting bin with head camera")
        return state_machine.bin_pose is not None
    state_machine.add_state("Detect Bin with Head Camera", detect_bin_with_head_camera_state)

    def adjust_to_table_grasping_pose_state():
        if state_machine.state_first_entry:
            pos_wrt_robot = np.array([0.49, 0.035, 0.8])
            ori_wrt_robot = np.array([0.49212, 0.48182, -0.47995, 0.54343])
            env.move_left_arm_to_pose(pos_wrt_robot, ori_wrt_robot)
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Adjusting to table grasping pose")
        return is_callback_complete("LeftArm_follow_trajectory_callback")
    state_machine.add_state("Adjust to Table Grasping Pose", adjust_to_table_grasping_pose_state)

    # # Phase 2: Object Grasping Loop
    # def detect_table_objects_state():
    #     if state_machine.state_first_entry:
    #         # Simulate object detection
    #         object_sequence = ["cube", "power_drill", "extrusion", "toy"]
    #         if state_machine.objects_processed < len(object_sequence):
    #             state_machine.object_name = object_sequence[state_machine.objects_processed]
    #         state_machine.state_first_entry = False
    #         print(f"State: {state_machine.get_state_name()} - Detecting table objects")
    #     return True
    # state_machine.add_state("Detect Table Objects", detect_table_objects_state)

    # def get_object_grasp_pose_state():
    #     if state_machine.state_first_entry:
    #         # Simulate grasp pose estimation
    #         state_machine.object_pose = state_machine.pose_estimator.estimate_pose(state_machine.object_name)
    #         combined_pose = np.concatenate([state_machine.object_pose[0], state_machine.object_pose[1]])
    #         state_machine.grasp_pose = state_machine.grasp_predictor.predict_grasp(state_machine.object_name, combined_pose)
    #         state_machine.state_first_entry = False
    #         print(f"State: {state_machine.get_state_name()} - Getting grasp pose for {state_machine.object_name}")
    #     return state_machine.grasp_pose is not None and not env.simulator.physics_callback_exists("LeftArm_follow_trajectory_callback")
    # state_machine.add_state("Get Object Grasp Pose", get_object_grasp_pose_state)

    # def move_to_object_pre_grasp_state():
    #     if state_machine.state_first_entry:
    #         pos_wrt_robot = state_machine.grasp_pose[:3] + np.array([0, 0, 0.3])
    #         ori_wrt_robot = state_machine.grasp_pose[3:]
    #         env.move_left_arm_to_pose(pos_wrt_robot, ori_wrt_robot)
    #         state_machine.state_first_entry = False
    #         print(f"State: {state_machine.get_state_name()} - Moving to pre-grasp position")
    #     return is_callback_complete("LeftArm_follow_trajectory_callback")
    # state_machine.add_state("Move to Object Pre Grasp", move_to_object_pre_grasp_state)

    # def move_to_object_grasp_state():
    #     if state_machine.state_first_entry:
    #         pos_wrt_robot = state_machine.grasp_pose[:3] + np.array([0, 0, 0.02])
    #         ori_wrt_robot = state_machine.grasp_pose[3:]
    #         env.move_left_arm_to_pose(pos_wrt_robot, ori_wrt_robot)
    #         state_machine.state_first_entry = False
    #         print(f"State: {state_machine.get_state_name()} - Moving to grasp position")
    #     return is_callback_complete("LeftArm_follow_trajectory_callback")
    # state_machine.add_state("Move to Object Grasp", move_to_object_grasp_state)

    # def grasp_object_state():
    #     if state_machine.state_first_entry:
    #         env.interface.left_gripper.set_gripper_close()
    #         state_machine.state_first_entry = False
    #         state_machine.wait_start_time = time.time()
    #         print(f"State: {state_machine.get_state_name()} - Grasping {state_machine.object_name}")
    #     return time.time() - state_machine.wait_start_time >= 3
    # state_machine.add_state("Grasp Object", grasp_object_state)

    # def move_to_object_retreat_state():
    #     if state_machine.state_first_entry:
    #         pos_wrt_robot = state_machine.grasp_pose[:3] + np.array([0, 0, 0.4])
    #         ori_wrt_robot = state_machine.grasp_pose[3:]
    #         env.move_left_arm_to_pose(pos_wrt_robot, ori_wrt_robot)
    #         state_machine.state_first_entry = False
    #         print(f"State: {state_machine.get_state_name()} - Moving to retreat position")
    #     return is_callback_complete("LeftArm_follow_trajectory_callback")
    # state_machine.add_state("Move to Object Retreat", move_to_object_retreat_state)

    # def move_to_bin_place_pose_state():
    #     if state_machine.state_first_entry:
    #         pos_wrt_robot = state_machine.bin_pose[0] + np.array([0, 0, 0.4])
    #         # ori_wrt_robot = [0, 0.7071, 0, 0.7071]
    #         ori_wrt_robot = [0, 0, 0, 1]
    #         env.move_left_arm_to_pose(pos_wrt_robot, ori_wrt_robot)
    #         state_machine.state_first_entry = False
    #         print(f"State: {state_machine.get_state_name()} - Moving to bin place position")
    #     return is_callback_complete("LeftArm_follow_trajectory_callback")
    # state_machine.add_state("Move to Bin Place Pose", move_to_bin_place_pose_state)

    # def release_object_state():
    #     if state_machine.state_first_entry:
    #         env.interface.left_gripper.set_gripper_open()
    #         state_machine.state_first_entry = False
    #         state_machine.wait_start_time = time.time()
    #         print(f"State: {state_machine.get_state_name()} - Releasing {state_machine.object_name}")
    #     return time.time() - state_machine.wait_start_time >= 3
    # state_machine.add_state("Release Object", release_object_state)

    # def return_to_table_grasping_pose_state():
    #     if state_machine.state_first_entry:
    #         pos_wrt_robot = np.array([0.49, 0.035, 0.8])
    #         ori_wrt_robot = np.array([0.49212, 0.48182, -0.47995, 0.54343])
    #         env.move_left_arm_to_pose(pos_wrt_robot, ori_wrt_robot)
    #         state_machine.state_first_entry = False
    #         print(f"State: {state_machine.get_state_name()} - Returning to table grasping pose")
    #     return is_callback_complete("LeftArm_follow_trajectory_callback")
    # state_machine.add_state("Return to Table Grasping Pose", return_to_table_grasping_pose_state)

    # Phase 3: Bin Placement
    def initialize_robot_for_bin_grasp_state():
        if state_machine.state_first_entry:
            move_joints_to_target(env.interface.left_arm, [2.00, -1.60, -0.60, -1.70, 0.00, -0.80, 0.00], 500)
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Initializing robot for bin grasp")
        return is_callback_complete("LeftArm_follow_trajectory_callback")
    state_machine.add_state("Initialize Robot for Bin Grasp", initialize_robot_for_bin_grasp_state)

    def navigate_to_bin_side_state():
        if state_machine.state_first_entry:
            current_pos = env.robot.get_position()[:2]
            waypoints_1 = state_machine.path_planner.plan_path(current_pos, [0, 1], 30)
            waypoints_2 = state_machine.path_planner.plan_path([0, 1], [0.65, 1], 30)
            waypoints = waypoints_1 + waypoints_2
            env.move_chassis_follow_path(waypoints)
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Navigating to bin side")
        return is_callback_complete("follow_path_callback")
    state_machine.add_state("Navigate to Bin Side", navigate_to_bin_side_state)

    def rotate_to_face_bin_state():
        if state_machine.state_first_entry:
            env.move_chassis_rotate(-math.pi / 2)
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Rotating to face bin")
        return is_callback_complete("rotate_callback")
    state_machine.add_state("Rotate to Face Bin", rotate_to_face_bin_state)

    def detect_bin_pose_state():
        if state_machine.state_first_entry:
            # Simulate bin pose detection
            state_machine.bin_pose = state_machine.pose_estimator.estimate_pose("bin")
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Detecting bin pose")
        return state_machine.bin_pose is not None
    state_machine.add_state("Detect Bin Pose", detect_bin_pose_state)

    def plan_dual_arm_pre_grasp_state():
        if state_machine.state_first_entry:
            # Simulate dual arm pre-grasp planning
            left_pos_wrt_robot = state_machine.bin_pose[0] + np.array([0, 0.2, 0.4])
            left_ori_wrt_robot = [0, 0, 0, 1]
            right_pos_wrt_robot = state_machine.bin_pose[0] + np.array([0, -0.2, 0.4])
            right_ori_wrt_robot = [0, 0, 0, 1]
            env.move_left_arm_to_pose(left_pos_wrt_robot, left_ori_wrt_robot)
            env.move_right_arm_to_pose(right_pos_wrt_robot, right_ori_wrt_robot)
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Planning dual arm pre-grasp")
        return is_callback_complete("LeftArm_follow_trajectory_callback") and is_callback_complete("RightArm_follow_trajectory_callback")
    state_machine.add_state("Plan Dual Arm Pre Grasp", plan_dual_arm_pre_grasp_state)

    def plan_dual_arm_grasp_state():
        if state_machine.state_first_entry:
            # Simulate dual arm grasp planning
            left_pos_wrt_robot = state_machine.bin_pose[0] + np.array([0, 0.16, 0.2])
            left_ori_wrt_robot = [0, 0, 0, 1]
            right_pos_wrt_robot = state_machine.bin_pose[0] + np.array([0, -0.16, 0.2])
            right_ori_wrt_robot = [0, 0, 0, 1]
            env.move_left_arm_to_pose(left_pos_wrt_robot, left_ori_wrt_robot)
            env.move_right_arm_to_pose(right_pos_wrt_robot, right_ori_wrt_robot)
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Planning dual arm grasp")
        return is_callback_complete("LeftArm_follow_trajectory_callback") and is_callback_complete("RightArm_follow_trajectory_callback")
    state_machine.add_state("Plan Dual Arm Grasp", plan_dual_arm_grasp_state)

    def grasp_bin_with_dual_arms_state():
        if state_machine.state_first_entry:
            # Simulate dual arm bin grasping
            env.interface.left_gripper.set_gripper_close()
            env.interface.right_gripper.set_gripper_close()
            state_machine.state_first_entry = False
            state_machine.wait_start_time = time.time()
            print(f"State: {state_machine.get_state_name()} - Grasping bin with dual arms")
        return time.time() - state_machine.wait_start_time >= 3
    state_machine.add_state("Grasp Bin with Dual Arms", grasp_bin_with_dual_arms_state)

    def lift_bin_with_dual_arms_state():
        if state_machine.state_first_entry:
            # Simulate lifting bin
            env.interface.leg.set_joint_positions(
                [0.239, 0.97, 0.692, 0]
            )
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Lifting bin with dual arms")
        return time.time() - state_machine.wait_start_time >= 3
    state_machine.add_state("Lift Bin with Dual Arms", lift_bin_with_dual_arms_state)

    def rotate_to_face_shelf_state():
        if state_machine.state_first_entry:
            env.move_chassis_rotate(0)
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Rotating to face shelf")
        return is_callback_complete("rotate_callback")
    state_machine.add_state("Rotate to Face Shelf", rotate_to_face_shelf_state)

    def navigate_to_shelf_front_state():
        if state_machine.state_first_entry:
            # Simulate navigation to shelf
            current_pos = env.robot.get_position()[:2]
            waypoints = state_machine.path_planner.plan_path(current_pos, [3, 4], 30)
            env.move_chassis_follow_path(waypoints)
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Navigating to shelf front")
        return is_callback_complete("follow_path_callback")
    state_machine.add_state("Navigate to Shelf Front", navigate_to_shelf_front_state)

    def rotate_to_face_shelf_final_state():
        if state_machine.state_first_entry:
            # Simulate final rotation to shelf
            env.move_chassis_rotate(0)
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Rotating to face shelf final")
        return is_callback_complete("rotate_callback")
    state_machine.add_state("Rotate to Face Shelf Final", rotate_to_face_shelf_final_state)

    def extend_arms_forward_state():
        if state_machine.state_first_entry:
            # Simulate extending arms forward
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Extending arms forward")
        return True
    state_machine.add_state("Extend Arms Forward", extend_arms_forward_state)

    def release_bin_on_shelf_state():
        if state_machine.state_first_entry:
            # Simulate releasing bin on shelf
            state_machine.state_first_entry = False
            state_machine.wait_start_time = time.time()
            print(f"State: {state_machine.get_state_name()} - Releasing bin on shelf")
        return time.time() - state_machine.wait_start_time >= 3
    state_machine.add_state("Release Bin on Shelf", release_bin_on_shelf_state)

    def retract_arms_state():
        if state_machine.state_first_entry:
            # Simulate retracting arms
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Retracting arms")
        return True
    state_machine.add_state("Retract Arms", retract_arms_state)

    # Phase 4: Final Navigation
    def initialize_robot_for_exit_state():
        if state_machine.state_first_entry:
            pos_wrt_robot = np.array([0.5, 0.1, 0.8])
            ori_wrt_robot = np.array([0, 0.7071, 0, 0.7071])
            env.move_left_arm_to_pose(pos_wrt_robot, ori_wrt_robot)
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Initializing robot for exit")
        return is_callback_complete("LeftArm_follow_trajectory_callback")
    state_machine.add_state("Initialize Robot for Exit", initialize_robot_for_exit_state)

    def rotate_to_exit_direction_state():
        if state_machine.state_first_entry:
            env.move_chassis_rotate(math.pi)
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Rotating to exit direction")
        return is_callback_complete("rotate_callback")
    state_machine.add_state("Rotate to Exit Direction", rotate_to_exit_direction_state)

    def navigate_to_final_destination_state():
        if state_machine.state_first_entry:
            # Simulate navigation to final destination
            state_machine.state_first_entry = False
            print(f"State: {state_machine.get_state_name()} - Navigating to final destination")
        return True
    state_machine.add_state("Navigate to Final Destination", navigate_to_final_destination_state)

    def ioai_main_callback():
        """Main simulation loop - executes state machine each physics step"""
        # First check if this is a new state (trigger should be called first)
        if state_machine.trigger():
            state_machine.state_first_entry = True
            print(f"Current state: {state_machine.get_state_name()}")
        
        # Then execute current state and move to next when complete
        if state_machine.execute_current_state():
            # Handle special transition logic after object release
            if state_machine.get_state_name() == "Release Object":
                state_machine.objects_processed += 1
                
                if state_machine.objects_processed < state_machine.total_objects:
                    # Continue with next object - jump back to object detection
                    state_machine.set_state(5)  # Detect Table Objects state index
                    state_machine.state_first_entry = True
                    print(f"Object {state_machine.objects_processed}/{state_machine.total_objects} completed, continuing with next object")
                else:
                    # All objects processed, continue normal progression
                    print("All objects processed, continuing to bin placement phase")
                    state_machine.next()
                    state_machine.state_first_entry = True
            else:
                # Normal state progression
                if not state_machine.next():
                    # Task completed, reset state machine for next cycle
                    print("Task completed successfully!")
                    state_machine.reset()
                    state_machine.state_first_entry = True
                    state_machine.objects_processed = 0

    # Register callback and run simulation
    env.simulator.add_physics_callback("ioai_main_callback", ioai_main_callback)
    env.run()


if __name__ == "__main__":
    main()

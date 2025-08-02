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
# Description: IOAI Competition Robot State Machine
# Author: Chenyu Cao, Herman Ye@Galbot
#
# COMPETITION GUIDELINES:
# - Replace baseline implementations with your own algorithms for higher scores
# - Focus on: object pose estimation, grasp planning, motion planning, and state management
# - Performance and robustness are key evaluation criteria
#
######################################################################################

from physics_simulator.utils.state_machine import SimpleStateMachine
import numpy as np
import time
import math


class IOAIRobotStateMachine:
    """
    Robot State Machine for IOAI Competition
    
    This class encapsulates all the state machine logic for the IOAI competition,
    including object grasping, bin placement, and navigation phases.
    """
    
    def __init__(self, env, object_pose_estimator, grasp_pose_predictor, motion_planner, path_planner):
        """
        Initialize the robot state machine with all required components.
        
        Args:
            env: IOAI environment instance
            object_pose_estimator: Object pose estimation module
            grasp_pose_predictor: Grasp pose prediction module
            motion_planner: Motion planning module
            path_planner: Path planning module
        """
        self.env = env
        self.pose_estimator = object_pose_estimator
        self.grasp_predictor = grasp_pose_predictor
        self.motion_planner = motion_planner
        self.path_planner = path_planner
        
        # Initialize state machine
        self.state_machine = SimpleStateMachine(max_states=-1)
        self.state_machine.state_first_entry = True
        self.state_machine.object_name = "cube"
        self.state_machine.object_pose = None
        self.state_machine.grasp_pose = None
        self.state_machine.bin_pose = None
        self.state_machine.objects_processed = 0
        self.state_machine.total_objects = 4  # cube, power_drill, extrusion, toy
        self.state_machine.wait_start_time = None
        
        # Wire components to state machine
        self.state_machine.pose_estimator = object_pose_estimator
        self.state_machine.grasp_predictor = grasp_pose_predictor
        self.state_machine.motion_planner = motion_planner
        self.state_machine.path_planner = path_planner
        
        # Setup all states
        self._setup_states()
    
    def _setup_states(self):
        """Setup all state machine states"""
        # Phase 1: Initial Setup and Navigation
        self.state_machine.add_state("Initialize Robot Safe Pose", self._initialize_robot_safe_pose_state)
        self.state_machine.add_state("Navigate to Table Front", self._navigate_to_table_front_state)
        self.state_machine.add_state("Rotate to Face Table", self._rotate_to_face_table_state)
        self.state_machine.add_state("Detect Bin with Head Camera", self._detect_bin_with_head_camera_state)
        self.state_machine.add_state("Adjust to Table Grasping Pose", self._adjust_to_table_grasping_pose_state)
        
        # Phase 2: Object Grasping Loop
        self.state_machine.add_state("Detect Table Objects", self._detect_table_objects_state)
        self.state_machine.add_state("Get Object Grasp Pose", self._get_object_grasp_pose_state)
        self.state_machine.add_state("Move to Object Pre Grasp", self._move_to_object_pre_grasp_state)
        self.state_machine.add_state("Move to Object Grasp", self._move_to_object_grasp_state)
        self.state_machine.add_state("Grasp Object", self._grasp_object_state)
        self.state_machine.add_state("Move to Object Retreat", self._move_to_object_retreat_state)
        self.state_machine.add_state("Move to Bin Place Pose", self._move_to_bin_place_pose_state)
        self.state_machine.add_state("Release Object", self._release_object_state)
        self.state_machine.add_state("Return to Table Grasping Pose", self._return_to_table_grasping_pose_state)
        
        # Phase 3: Bin Placement
        self.state_machine.add_state("Initialize Robot for Bin Grasp", self._initialize_robot_for_bin_grasp_state)
        self.state_machine.add_state("Navigate to Bin Side", self._navigate_to_bin_side_state)
        self.state_machine.add_state("Rotate to Face Bin", self._rotate_to_face_bin_state)
        self.state_machine.add_state("Detect Bin Pose", self._detect_bin_pose_state)
        self.state_machine.add_state("Plan Dual Arm Pre Grasp", self._plan_dual_arm_pre_grasp_state)
        self.state_machine.add_state("Plan Dual Arm Grasp", self._plan_dual_arm_grasp_state)
        self.state_machine.add_state("Grasp Bin with Dual Arms", self._grasp_bin_with_dual_arms_state)
        self.state_machine.add_state("Lift Bin with Dual Arms", self._lift_bin_with_dual_arms_state)
        self.state_machine.add_state("Rotate to Face Shelf", self._rotate_to_face_shelf_state)
        self.state_machine.add_state("Navigate to Shelf Front", self._navigate_to_shelf_front_state)
        self.state_machine.add_state("Lift Legs", self._lift_legs_state)
        self.state_machine.add_state("Move Forward to Shelf", self._move_forward_to_shelf_state)
        self.state_machine.add_state("Rotate to Face Shelf Final", self._rotate_to_face_shelf_final_state)
        self.state_machine.add_state("Release Bin on Shelf", self._release_bin_on_shelf_state)
        # Note: Retract Arms state is commented out in the original code
        # self.state_machine.add_state("Retract Arms", self._retract_arms_state)
        self.state_machine.add_state("Move Backward", self._move_backward_state)
        
        # Phase 4: Final Navigation
        self.state_machine.add_state("Initialize Robot for Exit", self._initialize_robot_for_exit_state)
        self.state_machine.add_state("Rotate to Exit Direction", self._rotate_to_exit_direction_state)
        self.state_machine.add_state("Navigate to Final Destination", self._navigate_to_final_destination_state)
    
    def _is_callback_complete(self, callback_name):
        """Helper function to check motion completion"""
        return not self.env.simulator.physics_callback_exists(callback_name)
    
    def _move_joints_to_target(self, module, target_positions, steps=500):
        """Helper function for joint interpolation"""
        from physics_simulator.utils.data_types import JointTrajectory
        start_positions = module.get_joint_positions()
        joint_positions = np.linspace(start_positions, target_positions, steps)
        joint_trajectory = JointTrajectory(positions=np.array(joint_positions))
        module.follow_trajectory(joint_trajectory)
    
    # Phase 1: Initial Setup and Navigation States
    def _initialize_robot_safe_pose_state(self):
        if self.state_machine.state_first_entry:
            self.state_machine.wait_start_time = time.time()
            poses = {
                self.env.interface.head: [0.0, 0.26],
                self.env.interface.leg: [0.0821758285164833, 0.6340972781181335, 0.5227039456367493, -0.00001198422432935331],
                self.env.interface.left_arm: [2.00, -1.60, -0.60, -1.70, 0.00, -0.80, 0.00],
                self.env.interface.right_arm: [-2.00, 1.60, 0.60, 1.70, 0.00, 0.80, 0.00]
            }
            for module, pose in poses.items():
                module.set_joint_positions(pose, immediate=False)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Initializing robot to safe pose")
        return time.time() - self.state_machine.wait_start_time >= 3
    
    def _navigate_to_table_front_state(self):
        if self.state_machine.state_first_entry:
            init_pos = self.env.robot.get_position()[:2]
            target_pos = [0, -0.3]
            waypoint_1 = self.state_machine.path_planner.plan_path(init_pos, [0.65, 4], 30)
            waypoint_2 = self.state_machine.path_planner.plan_path([0.65, 4], [0.65, 1], 30)
            waypoint_3 = self.state_machine.path_planner.plan_path([0.65, 1], [0, 1], 30)
            waypoint_4 = self.state_machine.path_planner.plan_path([0, 1], [0, -0.3], 30)

            waypoints = waypoint_1 + waypoint_2 + waypoint_3 + waypoint_4
            self.env.move_chassis_follow_path(waypoints)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Navigating to table front")
        return self._is_callback_complete("follow_path_callback")
    
    def _rotate_to_face_table_state(self):
        if self.state_machine.state_first_entry:
            self.env.move_chassis_rotate(0)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Rotating to face table")
        return self._is_callback_complete("rotate_callback")
    
    def _detect_bin_with_head_camera_state(self):
        if self.state_machine.state_first_entry:
            # Simulate bin detection with head camera
            self.state_machine.bin_pose = self.state_machine.pose_estimator.estimate_pose("bin")
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Detecting bin with head camera")
        return self.state_machine.bin_pose is not None
    
    def _adjust_to_table_grasping_pose_state(self):
        if self.state_machine.state_first_entry:
            pos_wrt_robot = np.array([0.49, 0.035, 0.8])
            ori_wrt_robot = np.array([0.49212, 0.48182, -0.47995, 0.54343])
            self.env.move_left_arm_to_pose(pos_wrt_robot, ori_wrt_robot)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Adjusting to table grasping pose")
        return self._is_callback_complete("LeftArm_follow_trajectory_callback")
    
    # Phase 2: Object Grasping Loop States
    def _detect_table_objects_state(self):
        if self.state_machine.state_first_entry:
            # Simulate object detection
            object_sequence = ["cube", "power_drill", "extrusion", "toy"]
            if self.state_machine.objects_processed < len(object_sequence):
                self.state_machine.object_name = object_sequence[self.state_machine.objects_processed]
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Detecting table objects")
        return True
    
    def _get_object_grasp_pose_state(self):
        if self.state_machine.state_first_entry:
            # Simulate grasp pose estimation
            self.state_machine.object_pose = self.state_machine.pose_estimator.estimate_pose(self.state_machine.object_name)
            combined_pose = np.concatenate([self.state_machine.object_pose[0], self.state_machine.object_pose[1]])
            self.state_machine.grasp_pose = self.state_machine.grasp_predictor.predict_grasp(self.state_machine.object_name, combined_pose)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Getting grasp pose for {self.state_machine.object_name}")
        return self.state_machine.grasp_pose is not None and not self.env.simulator.physics_callback_exists("LeftArm_follow_trajectory_callback")
    
    def _move_to_object_pre_grasp_state(self):
        if self.state_machine.state_first_entry:
            pos_wrt_robot = self.state_machine.grasp_pose[:3] + np.array([0, 0, 0.3])
            ori_wrt_robot = self.state_machine.grasp_pose[3:]
            self.env.move_left_arm_to_pose(pos_wrt_robot, ori_wrt_robot)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Moving to pre-grasp position")
        return self._is_callback_complete("LeftArm_follow_trajectory_callback")
    
    def _move_to_object_grasp_state(self):
        if self.state_machine.state_first_entry:
            pos_wrt_robot = self.state_machine.grasp_pose[:3] + np.array([0, 0, 0.02])
            ori_wrt_robot = self.state_machine.grasp_pose[3:]
            self.env.move_left_arm_to_pose(pos_wrt_robot, ori_wrt_robot)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Moving to grasp position")
        return self._is_callback_complete("LeftArm_follow_trajectory_callback")
    
    def _grasp_object_state(self):
        if self.state_machine.state_first_entry:
            self.env.interface.left_gripper.set_gripper_close()
            self.state_machine.state_first_entry = False
            self.state_machine.wait_start_time = time.time()
            print(f"State: {self.state_machine.get_state_name()} - Grasping {self.state_machine.object_name}")
        return time.time() - self.state_machine.wait_start_time >= 3
    
    def _move_to_object_retreat_state(self):
        if self.state_machine.state_first_entry:
            pos_wrt_robot = self.state_machine.grasp_pose[:3] + np.array([0, 0, 0.4])
            ori_wrt_robot = self.state_machine.grasp_pose[3:]
            self.env.move_left_arm_to_pose(pos_wrt_robot, ori_wrt_robot)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Moving to retreat position")
        return self._is_callback_complete("LeftArm_follow_trajectory_callback")
    
    def _move_to_bin_place_pose_state(self):
        if self.state_machine.state_first_entry:
            pos_wrt_robot = self.state_machine.bin_pose[0] + np.array([0, 0, 0.4])
            ori_wrt_robot = [0, 0, 0, 1]
            self.env.move_left_arm_to_pose(pos_wrt_robot, ori_wrt_robot)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Moving to bin place position")
        return self._is_callback_complete("LeftArm_follow_trajectory_callback")
    
    def _release_object_state(self):
        if self.state_machine.state_first_entry:
            self.env.interface.left_gripper.set_gripper_open()
            self.state_machine.state_first_entry = False
            self.state_machine.wait_start_time = time.time()
            print(f"State: {self.state_machine.get_state_name()} - Releasing {self.state_machine.object_name}")
        return time.time() - self.state_machine.wait_start_time >= 3
    
    def _return_to_table_grasping_pose_state(self):
        if self.state_machine.state_first_entry:
            pos_wrt_robot = np.array([0.49, 0.035, 0.8])
            ori_wrt_robot = np.array([0.49212, 0.48182, -0.47995, 0.54343])
            self.env.move_left_arm_to_pose(pos_wrt_robot, ori_wrt_robot)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Returning to table grasping pose")
        return self._is_callback_complete("LeftArm_follow_trajectory_callback")
    
    # Phase 3: Bin Placement States
    def _initialize_robot_for_bin_grasp_state(self):
        if self.state_machine.state_first_entry:
            self._move_joints_to_target(self.env.interface.left_arm, [2.00, -1.60, -0.60, -1.70, 0.00, -0.80, 0.00], 500)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Initializing robot for bin grasp")
        return self._is_callback_complete("LeftArm_follow_trajectory_callback")
    
    def _navigate_to_bin_side_state(self):
        if self.state_machine.state_first_entry:
            current_pos = self.env.robot.get_position()[:2]
            waypoints_1 = self.state_machine.path_planner.plan_path(current_pos, [0, 1], 30)
            waypoints_2 = self.state_machine.path_planner.plan_path([0, 1], [0.65, 1], 30)
            waypoints = waypoints_1 + waypoints_2
            self.env.move_chassis_follow_path(waypoints)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Navigating to bin side")
        return self._is_callback_complete("follow_path_callback")
    
    def _rotate_to_face_bin_state(self):
        if self.state_machine.state_first_entry:
            self.env.move_chassis_rotate(-math.pi / 2)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Rotating to face bin")
        return self._is_callback_complete("rotate_callback")
    
    def _detect_bin_pose_state(self):
        if self.state_machine.state_first_entry:
            # Simulate bin pose detection
            self.state_machine.bin_pose = self.state_machine.pose_estimator.estimate_pose("bin")
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Detecting bin pose")
        return self.state_machine.bin_pose is not None
    
    def _plan_dual_arm_pre_grasp_state(self):
        if self.state_machine.state_first_entry:
            # Simulate dual arm pre-grasp planning
            left_pos_wrt_robot = self.state_machine.bin_pose[0] + np.array([0, 0.2, 0.4])
            left_ori_wrt_robot = [0, 0, 0, 1]
            right_pos_wrt_robot = self.state_machine.bin_pose[0] + np.array([0, -0.2, 0.4])
            right_ori_wrt_robot = [0, 0, 0, 1]
            self.env.move_left_arm_to_pose(left_pos_wrt_robot, left_ori_wrt_robot)
            self.env.move_right_arm_to_pose(right_pos_wrt_robot, right_ori_wrt_robot)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Planning dual arm pre-grasp")
        return self._is_callback_complete("LeftArm_follow_trajectory_callback") and self._is_callback_complete("RightArm_follow_trajectory_callback")
    
    def _plan_dual_arm_grasp_state(self):
        if self.state_machine.state_first_entry:
            # Simulate dual arm grasp planning
            left_pos_wrt_robot = self.state_machine.bin_pose[0] + np.array([0, 0.16, 0.2])
            left_ori_wrt_robot = [0, 0, 0, 1]
            right_pos_wrt_robot = self.state_machine.bin_pose[0] + np.array([0, -0.16, 0.2])
            right_ori_wrt_robot = [0, 0, 0, 1]
            self.env.move_left_arm_to_pose(left_pos_wrt_robot, left_ori_wrt_robot)
            self.env.move_right_arm_to_pose(right_pos_wrt_robot, right_ori_wrt_robot)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Planning dual arm grasp")
        return self._is_callback_complete("LeftArm_follow_trajectory_callback") and self._is_callback_complete("RightArm_follow_trajectory_callback")
    
    def _grasp_bin_with_dual_arms_state(self):
        if self.state_machine.state_first_entry:
            # Simulate dual arm bin grasping
            self.env.interface.left_gripper.set_gripper_close()
            self.env.interface.right_gripper.set_gripper_close()
            self.state_machine.state_first_entry = False
            self.state_machine.wait_start_time = time.time()
            print(f"State: {self.state_machine.get_state_name()} - Grasping bin with dual arms")
        return time.time() - self.state_machine.wait_start_time >= 3
    
    def _lift_bin_with_dual_arms_state(self):
        if self.state_machine.state_first_entry:
            # Simulate lifting bin
            self.env.interface.leg.set_joint_positions([0.239, 0.97, 0.692, 0])
            self.state_machine.state_first_entry = False
            self.state_machine.wait_start_time = time.time()
            print(f"State: {self.state_machine.get_state_name()} - Lifting bin with dual arms")
        return time.time() - self.state_machine.wait_start_time >= 3
    
    def _rotate_to_face_shelf_state(self):
        if self.state_machine.state_first_entry:
            self.env.move_chassis_rotate(0)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Rotating to face shelf")
        return self._is_callback_complete("rotate_callback")
    
    def _navigate_to_shelf_front_state(self):
        if self.state_machine.state_first_entry:
            # Simulate navigation to shelf
            current_pos = self.env.robot.get_position()[:2]
            waypoints = self.state_machine.path_planner.plan_path(current_pos, [2.5, 4], 30)
            self.env.move_chassis_follow_path(waypoints)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Navigating to shelf front")
        return self._is_callback_complete("follow_path_callback")
    
    def _lift_legs_state(self):
        if self.state_machine.state_first_entry:
            # Simulate extending arms forward
            self.state_machine.wait_start_time = time.time()
            self.env.interface.leg.set_joint_positions([0.456, 1.3, 0.764, 0])
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Lifting legs")
        return time.time() - self.state_machine.wait_start_time >= 3
    
    def _move_forward_to_shelf_state(self):
        if self.state_machine.state_first_entry:
            # Simulate moving forward to shelf
            self.state_machine.state_first_entry = False
            current_pos = self.env.robot.get_position()[:2]
            waypoints = self.state_machine.path_planner.plan_path(current_pos, [3.4, 4], 30)
            self.env.move_chassis_follow_path(waypoints)
            print(f"State: {self.state_machine.get_state_name()} - Moving forward to shelf")
        return self._is_callback_complete("follow_path_callback")
    
    def _rotate_to_face_shelf_final_state(self):
        if self.state_machine.state_first_entry:
            # Simulate final rotation to shelf
            self.env.move_chassis_rotate(0)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Rotating to face shelf final")
        return self._is_callback_complete("rotate_callback")
    
    def _release_bin_on_shelf_state(self):
        if self.state_machine.state_first_entry:
            # Simulate releasing bin on shelf
            self.env.interface.left_gripper.set_gripper_open()
            self.env.interface.right_gripper.set_gripper_open()
            self.state_machine.state_first_entry = False
            self.state_machine.wait_start_time = time.time()
            print(f"State: {self.state_machine.get_state_name()} - Releasing bin on shelf")
        return time.time() - self.state_machine.wait_start_time >= 3
    
    def _move_backward_state(self):
        if self.state_machine.state_first_entry:
            # Simulate moving backward
            self.state_machine.state_first_entry = False
            current_pos = self.env.robot.get_position()[:2]
            waypoints = self.state_machine.path_planner.plan_path(current_pos, [2.5, 4], 30)
            self.env.move_chassis_xy(waypoints)
            print(f"State: {self.state_machine.get_state_name()} - Moving backward")
        return self._is_callback_complete("move_xy_callback")
    
    def _retract_arms_state(self):
        if self.state_machine.state_first_entry:
            # Simulate retracting arms
            self._move_joints_to_target(
                self.env.interface.left_arm,
                [2.00, -1.60, -0.60, -1.70, 0.00, -0.80, 0.00],
                500
            )
            self._move_joints_to_target(
                self.env.interface.right_arm,
                [-2.00, 1.60, 0.60, 1.70, 0.00, 0.80, 0.00],
                500
            )
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Retracting arms")
        return self._is_callback_complete("LeftArm_follow_trajectory_callback") and self._is_callback_complete("RightArm_follow_trajectory_callback")
    
    # Phase 4: Final Navigation States
    def _initialize_robot_for_exit_state(self):
        if self.state_machine.state_first_entry:
            self.state_machine.wait_start_time = time.time()
            poses = {
                self.env.interface.head: [0.0, 0.26],
                self.env.interface.leg: [0.0821758285164833, 0.6340972781181335, 0.5227039456367493, -0.00001198422432935331],
                self.env.interface.left_arm: [2.00, -1.60, -0.60, -1.70, 0.00, -0.80, 0.00],
                self.env.interface.right_arm: [-2.00, 1.60, 0.60, 1.70, 0.00, 0.80, 0.00]
            }
            for module, pose in poses.items():
                module.set_joint_positions(pose, immediate=False)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Initializing robot to safe pose")
        return time.time() - self.state_machine.wait_start_time >= 3
    
    def _rotate_to_exit_direction_state(self):
        if self.state_machine.state_first_entry:
            self.env.move_chassis_rotate(-math.pi / 2)
            self.state_machine.state_first_entry = False
            print(f"State: {self.state_machine.get_state_name()} - Rotating to exit direction")
        return self._is_callback_complete("rotate_callback")
    
    def _navigate_to_final_destination_state(self):
        if self.state_machine.state_first_entry:
            # Simulate navigation to final destination
            self.state_machine.state_first_entry = False
            current_pos = self.env.robot.get_position()[:2]
            waypoints = self.state_machine.path_planner.plan_path(current_pos, [4, 0], 30)
            self.env.move_chassis_follow_path(waypoints)
            print(f"State: {self.state_machine.get_state_name()} - Navigating to final destination")
        return self._is_callback_complete("follow_path_callback")
    
    def execute(self):
        """Main simulation loop - executes state machine each physics step"""
        # First check if this is a new state (trigger should be called first)
        if self.state_machine.trigger():
            self.state_machine.state_first_entry = True
            print(f"Current state: {self.state_machine.get_state_name()}")
        
        # Then execute current state and move to next when complete
        if self.state_machine.execute_current_state():
            # Handle special transition logic after object release
            if self.state_machine.get_state_name() == "Release Object":
                self.state_machine.objects_processed += 1
                
                if self.state_machine.objects_processed < self.state_machine.total_objects:
                    # Continue with next object - jump back to object detection
                    self.state_machine.set_state(5)  # Detect Table Objects state index
                    self.state_machine.state_first_entry = True
                    print(f"Object {self.state_machine.objects_processed}/{self.state_machine.total_objects} completed, continuing with next object")
                else:
                    # All objects processed, continue normal progression
                    print("All objects processed, continuing to bin placement phase")
                    self.state_machine.next()
                    self.state_machine.state_first_entry = True
            else:
                # Normal state progression
                if not self.state_machine.next():
                    # Task completed, reset state machine for next cycle
                    print("Task completed successfully!")
                    self.state_machine.reset()
                    self.state_machine.state_first_entry = True
                    self.state_machine.objects_processed = 0 
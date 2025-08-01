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
# Description: Advanced state machine for robot grasping task using transitions library
# Author: Chenyu Cao, Herman Ye@Galbot
#
######################################################################################

from transitions import Machine, State
from typing import Dict, Any, Optional, List
import numpy as np
import time
import math
from enum import Enum


#####################################################################################
# NOTE for Competition Participants:
#   This state machine provides a comprehensive framework for robot grasping tasks
#   with multiple phases including navigation, object manipulation, and bin placement.
#####################################################################################


class RobotState(Enum):
    """Enumeration of robot states for better type safety and state management.
    
    This enum defines all possible states in the robot grasping task workflow,
    organized into four main phases:
    1. Initial Setup and Navigation
    2. Object Grasping Loop
    3. Bin Placement
    4. Final Navigation
    
    Each state represents a specific action or decision point in the task execution.
    """
    
    # Phase 1: Initial Setup and Navigation
    INITIALIZE_ROBOT_SAFE_POSE = "initialize_robot_safe_pose"
    NAVIGATE_TO_TABLE_FRONT = "navigate_to_table_front"
    ROTATE_TO_FACE_TABLE = "rotate_to_face_table"
    DETECT_BIN_WITH_HEAD_CAMERA = "detect_bin_with_head_camera"
    ADJUST_TO_TABLE_GRASPING_POSE = "adjust_to_table_grasping_pose"
    
    # Phase 2: Object Grasping Loop
    DETECT_TABLE_OBJECTS = "detect_table_objects"
    GET_OBJECT_GRASP_POSE = "get_object_grasp_pose"
    MOVE_TO_OBJECT_PRE_GRASP = "move_to_object_pre_grasp"
    MOVE_TO_OBJECT_GRASP = "move_to_object_grasp"
    GRASP_OBJECT = "grasp_object"
    MOVE_TO_OBJECT_RETREAT = "move_to_object_retreat"
    MOVE_TO_BIN_PLACE_POSE = "move_to_bin_place_pose"
    PLACE_OBJECT_IN_BIN = "place_object_in_bin"
    RELEASE_OBJECT = "release_object"
    RETURN_TO_TABLE_GRASPING_POSE = "return_to_table_grasping_pose"
    
    # Phase 3: Bin Placement
    INITIALIZE_ROBOT_FOR_BIN_GRASP = "initialize_robot_for_bin_grasp"
    NAVIGATE_TO_BIN_SIDE = "navigate_to_bin_side"
    ROTATE_TO_FACE_BIN = "rotate_to_face_bin"
    DETECT_BIN_POSE = "detect_bin_pose"
    PLAN_DUAL_ARM_PRE_GRASP = "plan_dual_arm_pre_grasp"
    PLAN_DUAL_ARM_GRASP = "plan_dual_arm_grasp"
    GRASP_BIN_WITH_DUAL_ARMS = "grasp_bin_with_dual_arms"
    LIFT_BIN_WITH_DUAL_ARMS = "lift_bin_with_dual_arms"
    ROTATE_TO_FACE_SHELF = "rotate_to_face_shelf"
    NAVIGATE_TO_SHELF_FRONT = "navigate_to_shelf_front"
    ROTATE_TO_FACE_SHELF_FINAL = "rotate_to_face_shelf_final"
    EXTEND_ARMS_FORWARD = "extend_arms_forward"
    RELEASE_BIN_ON_SHELF = "release_bin_on_shelf"
    RETRACT_ARMS = "retract_arms"
    
    # Phase 4: Final Navigation
    INITIALIZE_ROBOT_FOR_EXIT = "initialize_robot_for_exit"
    ROTATE_TO_EXIT_DIRECTION = "rotate_to_exit_direction"
    NAVIGATE_TO_FINAL_DESTINATION = "navigate_to_final_destination"


class OfficialRobotStateMachine:
    """Advanced state machine for robot grasping task using transitions library.
    
    This class implements a comprehensive state machine for autonomous robot grasping
    tasks in the IOAI environment. The state machine manages the complete workflow
    from initial navigation to final object placement, including:
    
    - Robot navigation and positioning
    - Object detection and grasp pose estimation
    - Multi-object grasping and placement
    - Bin manipulation with dual-arm coordination
    - Task completion and exit navigation
    
    The implementation uses the transitions library for robust state management,
    providing clear state transitions, callbacks, and error handling.
    
    Coordinate convention:
        - Position: [x, y, z]
        - Orientation (quaternion): [qx, qy, qz, qw]
        - Grasp pose: [x, y, z, qx, qy, qz, qw]
    """
    
    def __init__(self, env):
        """Initialize the robot state machine with the IOAI environment.
        
        Args:
            env: IOAI environment instance providing robot control and simulation
                 capabilities.
        """
        self.env = env
        self.object_name = "cube"
        self.object_pose = None
        self.grasp_pose = None
        self.bin_pose = None
        self.state_first_entry = True
        self.wait_start_time = None
        self.objects_processed = 0
        self.total_objects = 4  # cube, power_drill, extrusion, toy
        
        # Initialize state machine
        self._setup_state_machine()
        
    def _setup_state_machine(self):
        """Setup the state machine with all states and transitions.
        
        This method configures the complete state machine including:
        - All state definitions with their respective callbacks
        - State transition logic and conditions
        - Initial state configuration
        - Error handling and invalid trigger management
        
        The state machine is organized into four main phases with clear
        progression logic and conditional branching for object processing.
        """
        
        # Define states with callbacks
        states = [
            # Phase 1: Initial Setup and Navigation
            State(RobotState.INITIALIZE_ROBOT_SAFE_POSE.value, on_enter=self._on_initialize_robot_safe_pose),
            State(RobotState.NAVIGATE_TO_TABLE_FRONT.value, on_enter=self._on_navigate_to_table_front),
            State(RobotState.ROTATE_TO_FACE_TABLE.value, on_enter=self._on_rotate_to_face_table),
            State(RobotState.DETECT_BIN_WITH_HEAD_CAMERA.value, on_enter=self._on_detect_bin_with_head_camera),
            State(RobotState.ADJUST_TO_TABLE_GRASPING_POSE.value, on_enter=self._on_adjust_to_table_grasping_pose),
            
            # Phase 2: Object Grasping Loop
            State(RobotState.DETECT_TABLE_OBJECTS.value, on_enter=self._on_detect_table_objects),
            State(RobotState.GET_OBJECT_GRASP_POSE.value, on_enter=self._on_get_object_grasp_pose),
            State(RobotState.MOVE_TO_OBJECT_PRE_GRASP.value, on_enter=self._on_move_to_object_pre_grasp),
            State(RobotState.MOVE_TO_OBJECT_GRASP.value, on_enter=self._on_move_to_object_grasp),
            State(RobotState.GRASP_OBJECT.value, on_enter=self._on_grasp_object),
            State(RobotState.MOVE_TO_OBJECT_RETREAT.value, on_enter=self._on_move_to_object_retreat),
            State(RobotState.MOVE_TO_BIN_PLACE_POSE.value, on_enter=self._on_move_to_bin_place_pose),
            State(RobotState.PLACE_OBJECT_IN_BIN.value, on_enter=self._on_place_object_in_bin),
            State(RobotState.RELEASE_OBJECT.value, on_enter=self._on_release_object),
            State(RobotState.RETURN_TO_TABLE_GRASPING_POSE.value, on_enter=self._on_return_to_table_grasping_pose),
            
            # Phase 3: Bin Placement
            State(RobotState.INITIALIZE_ROBOT_FOR_BIN_GRASP.value, on_enter=self._on_initialize_robot_for_bin_grasp),
            State(RobotState.NAVIGATE_TO_BIN_SIDE.value, on_enter=self._on_navigate_to_bin_side),
            State(RobotState.ROTATE_TO_FACE_BIN.value, on_enter=self._on_rotate_to_face_bin),
            State(RobotState.DETECT_BIN_POSE.value, on_enter=self._on_detect_bin_pose),
            State(RobotState.PLAN_DUAL_ARM_PRE_GRASP.value, on_enter=self._on_plan_dual_arm_pre_grasp),
            State(RobotState.PLAN_DUAL_ARM_GRASP.value, on_enter=self._on_plan_dual_arm_grasp),
            State(RobotState.GRASP_BIN_WITH_DUAL_ARMS.value, on_enter=self._on_grasp_bin_with_dual_arms),
            State(RobotState.LIFT_BIN_WITH_DUAL_ARMS.value, on_enter=self._on_lift_bin_with_dual_arms),
            State(RobotState.ROTATE_TO_FACE_SHELF.value, on_enter=self._on_rotate_to_face_shelf),
            State(RobotState.NAVIGATE_TO_SHELF_FRONT.value, on_enter=self._on_navigate_to_shelf_front),
            State(RobotState.ROTATE_TO_FACE_SHELF_FINAL.value, on_enter=self._on_rotate_to_face_shelf_final),
            State(RobotState.EXTEND_ARMS_FORWARD.value, on_enter=self._on_extend_arms_forward),
            State(RobotState.RELEASE_BIN_ON_SHELF.value, on_enter=self._on_release_bin_on_shelf),
            State(RobotState.RETRACT_ARMS.value, on_enter=self._on_retract_arms),
            
            # Phase 4: Final Navigation
            State(RobotState.INITIALIZE_ROBOT_FOR_EXIT.value, on_enter=self._on_initialize_robot_for_exit),
            State(RobotState.ROTATE_TO_EXIT_DIRECTION.value, on_enter=self._on_rotate_to_exit_direction),
            State(RobotState.NAVIGATE_TO_FINAL_DESTINATION.value, on_enter=self._on_navigate_to_final_destination),
        ]
        
        # Define transitions
        transitions = [
            # Phase 1: Initial Setup and Navigation
            {'trigger': 'next', 'source': RobotState.INITIALIZE_ROBOT_SAFE_POSE.value, 'dest': RobotState.NAVIGATE_TO_TABLE_FRONT.value},
            {'trigger': 'next', 'source': RobotState.NAVIGATE_TO_TABLE_FRONT.value, 'dest': RobotState.ROTATE_TO_FACE_TABLE.value},
            {'trigger': 'next', 'source': RobotState.ROTATE_TO_FACE_TABLE.value, 'dest': RobotState.DETECT_BIN_WITH_HEAD_CAMERA.value},
            {'trigger': 'next', 'source': RobotState.DETECT_BIN_WITH_HEAD_CAMERA.value, 'dest': RobotState.ADJUST_TO_TABLE_GRASPING_POSE.value},
            {'trigger': 'next', 'source': RobotState.ADJUST_TO_TABLE_GRASPING_POSE.value, 'dest': RobotState.DETECT_TABLE_OBJECTS.value},
            
            # Phase 2: Object Grasping Loop
            {'trigger': 'next', 'source': RobotState.DETECT_TABLE_OBJECTS.value, 'dest': RobotState.GET_OBJECT_GRASP_POSE.value},
            {'trigger': 'next', 'source': RobotState.GET_OBJECT_GRASP_POSE.value, 'dest': RobotState.MOVE_TO_OBJECT_PRE_GRASP.value},
            {'trigger': 'next', 'source': RobotState.MOVE_TO_OBJECT_PRE_GRASP.value, 'dest': RobotState.MOVE_TO_OBJECT_GRASP.value},
            {'trigger': 'next', 'source': RobotState.MOVE_TO_OBJECT_GRASP.value, 'dest': RobotState.GRASP_OBJECT.value},
            {'trigger': 'next', 'source': RobotState.GRASP_OBJECT.value, 'dest': RobotState.MOVE_TO_OBJECT_RETREAT.value},
            {'trigger': 'next', 'source': RobotState.MOVE_TO_OBJECT_RETREAT.value, 'dest': RobotState.MOVE_TO_BIN_PLACE_POSE.value},
            {'trigger': 'next', 'source': RobotState.MOVE_TO_BIN_PLACE_POSE.value, 'dest': RobotState.PLACE_OBJECT_IN_BIN.value},
            {'trigger': 'next', 'source': RobotState.PLACE_OBJECT_IN_BIN.value, 'dest': RobotState.RELEASE_OBJECT.value},
            {'trigger': 'next', 'source': RobotState.RELEASE_OBJECT.value, 'dest': RobotState.RETURN_TO_TABLE_GRASPING_POSE.value},
            
            # Object loop transition logic
            {'trigger': 'continue_grasping', 'source': RobotState.RETURN_TO_TABLE_GRASPING_POSE.value, 'dest': RobotState.DETECT_TABLE_OBJECTS.value},
            {'trigger': 'start_bin_placement', 'source': RobotState.RETURN_TO_TABLE_GRASPING_POSE.value, 'dest': RobotState.INITIALIZE_ROBOT_FOR_BIN_GRASP.value},
            
            # Phase 3: Bin Placement
            {'trigger': 'next', 'source': RobotState.INITIALIZE_ROBOT_FOR_BIN_GRASP.value, 'dest': RobotState.NAVIGATE_TO_BIN_SIDE.value},
            {'trigger': 'next', 'source': RobotState.NAVIGATE_TO_BIN_SIDE.value, 'dest': RobotState.ROTATE_TO_FACE_BIN.value},
            {'trigger': 'next', 'source': RobotState.ROTATE_TO_FACE_BIN.value, 'dest': RobotState.DETECT_BIN_POSE.value},
            {'trigger': 'next', 'source': RobotState.DETECT_BIN_POSE.value, 'dest': RobotState.PLAN_DUAL_ARM_PRE_GRASP.value},
            {'trigger': 'next', 'source': RobotState.PLAN_DUAL_ARM_PRE_GRASP.value, 'dest': RobotState.PLAN_DUAL_ARM_GRASP.value},
            {'trigger': 'next', 'source': RobotState.PLAN_DUAL_ARM_GRASP.value, 'dest': RobotState.GRASP_BIN_WITH_DUAL_ARMS.value},
            {'trigger': 'next', 'source': RobotState.GRASP_BIN_WITH_DUAL_ARMS.value, 'dest': RobotState.LIFT_BIN_WITH_DUAL_ARMS.value},
            {'trigger': 'next', 'source': RobotState.LIFT_BIN_WITH_DUAL_ARMS.value, 'dest': RobotState.ROTATE_TO_FACE_SHELF.value},
            {'trigger': 'next', 'source': RobotState.ROTATE_TO_FACE_SHELF.value, 'dest': RobotState.NAVIGATE_TO_SHELF_FRONT.value},
            {'trigger': 'next', 'source': RobotState.NAVIGATE_TO_SHELF_FRONT.value, 'dest': RobotState.ROTATE_TO_FACE_SHELF_FINAL.value},
            {'trigger': 'next', 'source': RobotState.ROTATE_TO_FACE_SHELF_FINAL.value, 'dest': RobotState.EXTEND_ARMS_FORWARD.value},
            {'trigger': 'next', 'source': RobotState.EXTEND_ARMS_FORWARD.value, 'dest': RobotState.RELEASE_BIN_ON_SHELF.value},
            {'trigger': 'next', 'source': RobotState.RELEASE_BIN_ON_SHELF.value, 'dest': RobotState.RETRACT_ARMS.value},
            
            # Phase 4: Final Navigation
            {'trigger': 'next', 'source': RobotState.RETRACT_ARMS.value, 'dest': RobotState.INITIALIZE_ROBOT_FOR_EXIT.value},
            {'trigger': 'next', 'source': RobotState.INITIALIZE_ROBOT_FOR_EXIT.value, 'dest': RobotState.ROTATE_TO_EXIT_DIRECTION.value},
            {'trigger': 'next', 'source': RobotState.ROTATE_TO_EXIT_DIRECTION.value, 'dest': RobotState.NAVIGATE_TO_FINAL_DESTINATION.value},
            
            # Reset to start
            {'trigger': 'reset', 'source': '*', 'dest': RobotState.INITIALIZE_ROBOT_SAFE_POSE.value},
        ]
        
        # Create state machine
        self.machine = Machine(
            model=self,
            states=states,
            transitions=transitions,
            initial=RobotState.INITIALIZE_ROBOT_SAFE_POSE.value,
            ignore_invalid_triggers=True
        )
        
    def _is_motion_complete(self, callback_name: str) -> bool:
        """Check if a motion is complete by verifying callback existence.
        
        Args:
            callback_name (str): The name of the physics callback to check.
            
        Returns:
            bool: True if the motion is complete (callback doesn't exist),
                  False if the motion is still in progress.
        """
        return not self.env.simulator.physics_callback_exists(callback_name)
    
    # Phase 1: Initial Setup and Navigation
    def _on_initialize_robot_safe_pose(self):
        """Initialize robot to safe pose for movement.
        
        This state sets the robot's arms to a safe configuration suitable for
        navigation and initial positioning. The safe pose ensures the robot
        can move without collision risks.
        """
        if self.state_first_entry:
            # Set robot to safe pose for navigation
            robot_pos = np.array([0.5, 0.1, 0.8])
            robot_ori = np.array([0, 0.7071, 0, 0.7071])
            self.env.move_left_arm_to_pose(robot_pos, robot_ori)
            self.state_first_entry = False
            print(f"State: {self.state} - Initializing robot to safe pose")
    
    def _on_navigate_to_table_front(self):
        """Navigate to front of table using path following.
        
        This state executes a series of waypoint-based navigation movements
        to position the robot in front of the target table for object
        manipulation tasks.
        """
        if self.state_first_entry:
            waypoints = [[0, 4], [0, 2], [0, -0.2]]
            self.env.move_chassis_follow_path(waypoints)
            self.state_first_entry = False
            print(f"State: {self.state} - Navigating to table front")
    
    def _on_rotate_to_face_table(self):
        """Rotate robot to face table for optimal viewing and manipulation.
        
        This state rotates the robot chassis to orient the robot towards
        the table, ensuring optimal camera angles and arm reach for
        subsequent object detection and manipulation tasks.
        """
        if self.state_first_entry:
            self.env.move_chassis_rotate(0)
            self.state_first_entry = False
            print(f"State: {self.state} - Rotating to face table")
    
    def _on_detect_bin_with_head_camera(self):
        """Detect bin position using head camera for spatial awareness.
        
        This state uses the robot's head-mounted camera to detect and
        localize the target bin in the environment. The detected bin pose
        is stored for subsequent manipulation tasks.
        """
        if self.state_first_entry:
            # Simulate bin detection with head camera
            self.bin_pose = (np.array([0.65, 0.9, 0.1]), np.array([0, 0, 0, 1]))
            self.state_first_entry = False
            print(f"State: {self.state} - Detecting bin with head camera")
    
    def _on_adjust_to_table_grasping_pose(self):
        """Adjust robot to table grasping pose for object manipulation.
        
        This state positions the robot's arms in an optimal configuration
        for grasping objects from the table. The pose is designed to
        maximize reach and dexterity for the upcoming manipulation tasks.
        """
        if self.state_first_entry:
            robot_pos = np.array([0.5, 0.1, 0.8])
            robot_ori = np.array([0, 0.7071, 0, 0.7071])
            self.env.move_left_arm_to_pose(robot_pos, robot_ori)
            self.state_first_entry = False
            print(f"State: {self.state} - Adjusting to table grasping pose")
    
    # Phase 2: Object Grasping Loop
    def _on_detect_table_objects(self):
        """Detect objects on table for sequential processing.
        
        This state identifies objects present on the table and selects
        the next object for manipulation. The system processes objects
        in a predefined sequence to ensure systematic task completion.
        """
        if self.state_first_entry:
            # Simulate object detection
            object_sequence = ["cube", "power_drill", "extrusion", "toy"]
            if self.objects_processed < len(object_sequence):
                self.object_name = object_sequence[self.objects_processed]
            self.state_first_entry = False
            print(f"State: {self.state} - Detecting table objects")
    
    def _on_get_object_grasp_pose(self):
        """Get grasp pose for detected object using pose estimation.
        
        This state calculates the optimal grasp pose for the selected object
        based on its current pose and geometric properties. The grasp pose
        includes both position and orientation for precise manipulation.
        """
        if self.state_first_entry:
            # Simulate grasp pose estimation
            self.object_pose = (np.array([0.5, 0.0, 0.1]), np.array([0, 0, 0, 1]))
            self.grasp_pose = np.concatenate([self.object_pose[0], self.object_pose[1]])
            self.state_first_entry = False
            print(f"State: {self.state} - Getting grasp pose for {self.object_name}")
    
    def _on_move_to_object_pre_grasp(self):
        """Move to pre-grasp position for object preparation.
        
        This state positions the robot arm above the target object at a
        safe distance for final approach. The pre-grasp position ensures
        collision-free movement to the final grasp pose.
        """
        if self.state_first_entry:
            robot_pos = self.grasp_pose[:3] + np.array([0, 0, 0.3])
            robot_ori = self.grasp_pose[3:]
            self.env.move_left_arm_to_pose(robot_pos, robot_ori)
            self.state_first_entry = False
            print(f"State: {self.state} - Moving to pre-grasp position")
    
    def _on_move_to_object_grasp(self):
        """Move to grasp position for object manipulation.
        
        This state moves the robot arm to the final grasp position where
        the gripper can securely grasp the target object. The position
        is calculated based on the object's pose and grasp strategy.
        """
        if self.state_first_entry:
            robot_pos = self.grasp_pose[:3] + np.array([0, 0, 0.02])
            robot_ori = self.grasp_pose[3:]
            self.env.move_left_arm_to_pose(robot_pos, robot_ori)
            self.state_first_entry = False
            print(f"State: {self.state} - Moving to grasp position")
    
    def _on_grasp_object(self):
        """Grasp the object using the robot's gripper.
        
        This state executes the actual grasping action by closing the
        gripper around the target object. The system waits for a
        specified duration to ensure secure object acquisition.
        """
        if self.state_first_entry:
            self.env.interface.left_gripper.set_gripper_close()
            self.state_first_entry = False
            self.wait_start_time = time.time()
            print(f"State: {self.state} - Grasping {self.object_name}")
    
    def _on_move_to_object_retreat(self):
        """Move to retreat position after successful grasping.
        
        This state moves the robot arm back to a safe position above
        the object after grasping. The retreat position prevents
        collisions during subsequent movements and object transport.
        """
        if self.state_first_entry:
            robot_pos = self.grasp_pose[:3] + np.array([0, 0, 0.3])
            robot_ori = self.grasp_pose[3:]
            self.env.move_left_arm_to_pose(robot_pos, robot_ori)
            self.state_first_entry = False
            print(f"State: {self.state} - Moving to retreat position")
    
    def _on_move_to_bin_place_pose(self):
        """Move to bin place position for object deposition.
        
        This state positions the robot arm above the target bin at a
        safe height for object placement. The position is calculated
        based on the bin's detected pose and placement requirements.
        """
        if self.state_first_entry:
            robot_pos = self.bin_pose[0] + np.array([0, 0, 0.4])
            robot_ori = [0, 0, 0, 1]
            self.env.move_left_arm_to_pose(robot_pos, robot_ori)
            self.state_first_entry = False
            print(f"State: {self.state} - Moving to bin place position")
    
    def _on_place_object_in_bin(self):
        """Place object in bin at the target location.
        
        This state moves the robot arm to the final placement position
        within the bin. The position is carefully calculated to ensure
        proper object placement without damage or misalignment.
        """
        if self.state_first_entry:
            robot_pos = self.bin_pose[0] + np.array([0, 0, 0.1])
            robot_ori = [0, 0, 0, 1]
            self.env.move_left_arm_to_pose(robot_pos, robot_ori)
            self.state_first_entry = False
            print(f"State: {self.state} - Placing {self.object_name} in bin")
    
    def _on_release_object(self):
        """Release object in bin using gripper control.
        
        This state opens the gripper to release the grasped object
        into the target bin. The system waits for a specified duration
        to ensure complete object release and stability.
        """
        if self.state_first_entry:
            self.env.interface.left_gripper.set_gripper_open()
            self.state_first_entry = False
            self.wait_start_time = time.time()
            print(f"State: {self.state} - Releasing {self.object_name}")
    
    def _on_return_to_table_grasping_pose(self):
        """Return to table grasping pose for next object processing.
        
        This state moves the robot arm back to the optimal position
        for detecting and grasping the next object on the table.
        The pose is maintained throughout the object processing loop.
        """
        if self.state_first_entry:
            robot_pos = np.array([0.5, 0.1, 0.8])
            robot_ori = np.array([0, 0.7071, 0, 0.7071])
            self.env.move_left_arm_to_pose(robot_pos, robot_ori)
            self.state_first_entry = False
            print(f"State: {self.state} - Returning to table grasping pose")
    
    # Phase 3: Bin Placement
    def _on_initialize_robot_for_bin_grasp(self):
        """Initialize robot for bin grasping with dual-arm coordination.
        
        This state configures both robot arms in preparation for
        dual-arm bin manipulation. The arms are positioned to
        enable coordinated grasping of the target bin.
        """
        if self.state_first_entry:
            # Initialize both arms for bin grasping
            self.env._move_joints_to_target(
                self.env.interface.left_arm, 
                [2.00, -1.60, -0.60, -1.70, 0.00, -0.80, 0.00], 
                500
            )
            self.state_first_entry = False
            print(f"State: {self.state} - Initializing robot for bin grasp")
    
    def _on_navigate_to_bin_side(self):
        """Navigate to bin side for optimal access.
        
        This state executes a complex navigation path to position
        the robot at the side of the bin. The path includes both
        linear and curved segments for optimal positioning.
        """
        if self.state_first_entry:
            waypoints_1 = np.linspace([0, 0], [0, 0.9], 30).tolist()
            waypoints_2 = np.linspace([0, 0.9], [0.65, 0.9], 30).tolist()
            waypoints = waypoints_1 + waypoints_2
            self.env.move_chassis_follow_path(waypoints)
            self.state_first_entry = False
            print(f"State: {self.state} - Navigating to bin side")
    
    def _on_rotate_to_face_bin(self):
        """Rotate to face bin for dual-arm manipulation.
        
        This state rotates the robot chassis to orient towards
        the bin, ensuring optimal positioning for dual-arm
        grasping and manipulation tasks.
        """
        if self.state_first_entry:
            self.env.move_chassis_rotate(-math.pi / 2)
            self.state_first_entry = False
            print(f"State: {self.state} - Rotating to face bin")
    
    def _on_detect_bin_pose(self):
        """Detect bin pose for grasping using sensor data.
        
        This state uses the robot's sensors to determine the
        precise pose of the bin for dual-arm grasping. The
        detected pose is used for subsequent manipulation planning.
        """
        if self.state_first_entry:
            # Simulate bin pose detection
            self.bin_pose = (np.array([0.65, 0.9, 0.1]), np.array([0, 0, 0, 1]))
            self.state_first_entry = False
            print(f"State: {self.state} - Detecting bin pose")
    
    def _on_plan_dual_arm_pre_grasp(self):
        """Plan dual arm pre-grasp for bin manipulation.
        
        This state calculates the optimal pre-grasp positions for
        both robot arms when approaching the bin. The planning
        ensures coordinated movement and collision-free approach.
        """
        if self.state_first_entry:
            # Simulate dual arm pre-grasp planning
            self.state_first_entry = False
            print(f"State: {self.state} - Planning dual arm pre-grasp")
    
    def _on_plan_dual_arm_grasp(self):
        """Plan dual arm grasp for bin manipulation.
        
        This state calculates the final grasp positions for both
        robot arms to securely grasp the bin. The planning
        ensures stable and balanced bin manipulation.
        """
        if self.state_first_entry:
            # Simulate dual arm grasp planning
            self.state_first_entry = False
            print(f"State: {self.state} - Planning dual arm grasp")
    
    def _on_grasp_bin_with_dual_arms(self):
        """Grasp bin with dual arms for coordinated manipulation.
        
        This state executes the dual-arm grasping action to securely
        hold the bin. Both arms work in coordination to ensure
        stable and balanced bin acquisition.
        """
        if self.state_first_entry:
            # Simulate dual arm bin grasping
            self.state_first_entry = False
            self.wait_start_time = time.time()
            print(f"State: {self.state} - Grasping bin with dual arms")
    
    def _on_lift_bin_with_dual_arms(self):
        """Lift bin with dual arms for transport.
        
        This state raises the grasped bin to a safe height for
        transportation. The lifting motion is coordinated between
        both arms to maintain stability and balance.
        """
        if self.state_first_entry:
            # Simulate lifting bin
            self.state_first_entry = False
            print(f"State: {self.state} - Lifting bin with dual arms")
    
    def _on_rotate_to_face_shelf(self):
        """Rotate to face shelf for bin placement.
        
        This state rotates the robot chassis to orient towards
        the target shelf where the bin will be placed. The
        rotation ensures optimal positioning for placement tasks.
        """
        if self.state_first_entry:
            self.env.move_chassis_rotate(-math.pi / 2)
            self.state_first_entry = False
            print(f"State: {self.state} - Rotating to face shelf")
    
    def _on_navigate_to_shelf_front(self):
        """Navigate to shelf front for bin placement.
        
        This state moves the robot to the front of the target shelf
        where the bin will be placed. The navigation ensures
        optimal positioning for the placement operation.
        """
        if self.state_first_entry:
            # Simulate navigation to shelf
            self.state_first_entry = False
            print(f"State: {self.state} - Navigating to shelf front")
    
    def _on_rotate_to_face_shelf_final(self):
        """Rotate to face shelf final position for precise placement.
        
        This state performs the final rotation to achieve optimal
        orientation for bin placement on the shelf. The rotation
        ensures precise and stable placement.
        """
        if self.state_first_entry:
            # Simulate final rotation to shelf
            self.state_first_entry = False
            print(f"State: {self.state} - Rotating to face shelf final")
    
    def _on_extend_arms_forward(self):
        """Extend arms forward for bin placement on shelf.
        
        This state extends both robot arms forward to position
        the bin over the target location on the shelf. The
        extension ensures proper placement without collision.
        """
        if self.state_first_entry:
            # Simulate extending arms forward
            self.state_first_entry = False
            print(f"State: {self.state} - Extending arms forward")
    
    def _on_release_bin_on_shelf(self):
        """Release bin on shelf using dual-arm coordination.
        
        This state releases the bin from both robot arms onto
        the target shelf location. The release is coordinated
        to ensure stable placement and prevent tipping.
        """
        if self.state_first_entry:
            # Simulate releasing bin on shelf
            self.state_first_entry = False
            self.wait_start_time = time.time()
            print(f"State: {self.state} - Releasing bin on shelf")
    
    def _on_retract_arms(self):
        """Retract arms after bin placement for safety.
        
        This state retracts both robot arms to a safe position
        after successfully placing the bin on the shelf. The
        retraction prevents collisions during subsequent movements.
        """
        if self.state_first_entry:
            # Simulate retracting arms
            self.state_first_entry = False
            print(f"State: {self.state} - Retracting arms")
    
    # Phase 4: Final Navigation
    def _on_initialize_robot_for_exit(self):
        """Initialize robot for exit navigation and task completion.
        
        This state configures the robot in a safe pose suitable for
        exit navigation. The pose ensures the robot can move freely
        without collision risks during the final navigation phase.
        """
        if self.state_first_entry:
            # Set robot to safe pose for exit
            robot_pos = np.array([0.5, 0.1, 0.8])
            robot_ori = np.array([0, 0.7071, 0, 0.7071])
            self.env.move_left_arm_to_pose(robot_pos, robot_ori)
            self.state_first_entry = False
            print(f"State: {self.state} - Initializing robot for exit")
    
    def _on_rotate_to_exit_direction(self):
        """Rotate to exit direction for final navigation.
        
        This state rotates the robot chassis to face the exit
        direction. The rotation prepares the robot for the
        final navigation to the destination point.
        """
        if self.state_first_entry:
            self.env.move_chassis_rotate(math.pi)
            self.state_first_entry = False
            print(f"State: {self.state} - Rotating to exit direction")
    
    def _on_navigate_to_final_destination(self):
        """Navigate to final destination to complete the task.
        
        This state executes the final navigation movement to
        reach the designated destination point, completing
        the entire robot grasping and manipulation task.
        """
        if self.state_first_entry:
            # Simulate navigation to final destination
            self.state_first_entry = False
            print(f"State: {self.state} - Navigating to final destination")
    
    def is_state_complete(self) -> bool:
        """Check if current state is complete and ready for transition.
        
        This method evaluates the completion status of the current state
        based on motion completion, timing requirements, and state-specific
        conditions. It determines when the state machine should proceed
        to the next state.
        COMPETITION TASK 4 - 
        Returns:
            bool: True if the current state is complete and ready for
                  transition, False otherwise.
        """
        if self.state == RobotState.INITIALIZE_ROBOT_SAFE_POSE.value:
            return self._is_motion_complete("LeftArm_follow_trajectory_callback")
        elif self.state == RobotState.NAVIGATE_TO_TABLE_FRONT.value:
            return self._is_motion_complete("follow_path_callback")
        elif self.state == RobotState.ROTATE_TO_FACE_TABLE.value:
            return self._is_motion_complete("rotate_callback")
        elif self.state == RobotState.DETECT_BIN_WITH_HEAD_CAMERA.value:
            return self.bin_pose is not None
        elif self.state == RobotState.ADJUST_TO_TABLE_GRASPING_POSE.value:
            return self._is_motion_complete("LeftArm_follow_trajectory_callback")
        elif self.state == RobotState.DETECT_TABLE_OBJECTS.value:
            return True
        elif self.state == RobotState.GET_OBJECT_GRASP_POSE.value:
            return self.grasp_pose is not None
        elif self.state == RobotState.MOVE_TO_OBJECT_PRE_GRASP.value:
            return self._is_motion_complete("LeftArm_follow_trajectory_callback")
        elif self.state == RobotState.MOVE_TO_OBJECT_GRASP.value:
            return self._is_motion_complete("LeftArm_follow_trajectory_callback")
        elif self.state == RobotState.GRASP_OBJECT.value:
            return time.time() - self.wait_start_time >= 3
        elif self.state == RobotState.MOVE_TO_OBJECT_RETREAT.value:
            return self._is_motion_complete("LeftArm_follow_trajectory_callback")
        elif self.state == RobotState.MOVE_TO_BIN_PLACE_POSE.value:
            return self._is_motion_complete("LeftArm_follow_trajectory_callback")
        elif self.state == RobotState.PLACE_OBJECT_IN_BIN.value:
            return self._is_motion_complete("LeftArm_follow_trajectory_callback")
        elif self.state == RobotState.RELEASE_OBJECT.value:
            return time.time() - self.wait_start_time >= 3
        elif self.state == RobotState.RETURN_TO_TABLE_GRASPING_POSE.value:
            return self._is_motion_complete("LeftArm_follow_trajectory_callback")
        elif self.state == RobotState.INITIALIZE_ROBOT_FOR_BIN_GRASP.value:
            return self._is_motion_complete("LeftArm_follow_trajectory_callback")
        elif self.state == RobotState.NAVIGATE_TO_BIN_SIDE.value:
            return self._is_motion_complete("follow_path_callback")
        elif self.state == RobotState.ROTATE_TO_FACE_BIN.value:
            return self._is_motion_complete("rotate_callback")
        elif self.state == RobotState.DETECT_BIN_POSE.value:
            return self.bin_pose is not None
        elif self.state == RobotState.PLAN_DUAL_ARM_PRE_GRASP.value:
            return True
        elif self.state == RobotState.PLAN_DUAL_ARM_GRASP.value:
            return True
        elif self.state == RobotState.GRASP_BIN_WITH_DUAL_ARMS.value:
            return time.time() - self.wait_start_time >= 3
        elif self.state == RobotState.LIFT_BIN_WITH_DUAL_ARMS.value:
            return True
        elif self.state == RobotState.ROTATE_TO_FACE_SHELF.value:
            return self._is_motion_complete("rotate_callback")
        elif self.state == RobotState.NAVIGATE_TO_SHELF_FRONT.value:
            return True
        elif self.state == RobotState.ROTATE_TO_FACE_SHELF_FINAL.value:
            return True
        elif self.state == RobotState.EXTEND_ARMS_FORWARD.value:
            return True
        elif self.state == RobotState.RELEASE_BIN_ON_SHELF.value:
            return time.time() - self.wait_start_time >= 3
        elif self.state == RobotState.RETRACT_ARMS.value:
            return True
        elif self.state == RobotState.INITIALIZE_ROBOT_FOR_EXIT.value:
            return self._is_motion_complete("LeftArm_follow_trajectory_callback")
        elif self.state == RobotState.ROTATE_TO_EXIT_DIRECTION.value:
            return self._is_motion_complete("rotate_callback")
        elif self.state == RobotState.NAVIGATE_TO_FINAL_DESTINATION.value:
            return True
        return False
    
    def execute(self):
        """Execute the current state and handle transitions.
        
        This method is the main execution loop for the state machine.
        It checks if the current state is complete and handles state
        transitions based on completion conditions and special logic
        for object processing loops.
        """
        # Check if current state is complete
        if self.is_state_complete():
            # Handle special transition logic
            if self.state == RobotState.RELEASE_OBJECT.value:
                self._handle_release_transition()
            else:
                # Normal progression
                self.next()
                self.state_first_entry = True
    
    def _handle_release_transition(self):
        """Handle special transition logic after object release.
        
        This method manages the transition logic after an object has been
        successfully released into the bin. It determines whether to continue
        with the next object in the sequence or proceed to the bin placement
        phase based on the number of objects processed.
        """
        self.objects_processed += 1
        
        if self.objects_processed < self.total_objects:
            # Continue with next object
            self.continue_grasping()
            self.state_first_entry = True
            print(f"Object {self.objects_processed}/{self.total_objects} completed, continuing with next object")
        else:
            # All objects processed, start bin placement
            self.start_bin_placement()
            self.state_first_entry = True
            print("All objects processed, starting bin placement phase")
    
    def reset_machine(self):
        """Reset the state machine to initial state.
        
        This method resets all state machine variables and returns
        the system to the initial state. It clears all object poses,
        grasp poses, and processing counters to prepare for a new
        task execution cycle.
        """
        self.reset()
        self.state_first_entry = True
        self.object_name = "cube"
        self.object_pose = None
        self.grasp_pose = None
        self.bin_pose = None
        self.objects_processed = 0
        print("State machine reset to initial state") 
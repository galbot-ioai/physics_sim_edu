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
# Description: a rough pipeline for IOAI env
# Author: Chenyu Cao@Galbot
# Date: 2025-07-28
#
#####################################################################################

from ioai_env import IOAIEnv
import numpy as np
from physics_simulator.utils.state_machine import SimpleStateMachine
from physics_simulator.utils.data_types import JointTrajectory
import math
import time
from pathlib import Path
import os
import cv2
from physics_simulator.utils import preprocess_depth

# ---------- Init env ---------
env = IOAIEnv(headless=False)

# ---------- Useful functions ----preprocess_depth --
# It is a simple toolbox to generate a grsping pose from the detected pose
# you can modify this as well
from grasp_reg import GraspRegistration
grasp_reg = GraspRegistration()

# --------- Pose Estimator ------

class BasePoseEstimator:
    def __init__(self, env: IOAIEnv):
        pass

    def estimate_pose(self, object_name):
        pass

class DummyPoseEstimator(BasePoseEstimator):
    def __init__(self, env: IOAIEnv):
        self.env = env
    
    # NOTE: You need to change the dummy pose estimator to the vision-based one
    # A state-based method, which you can only get the basic scores
    def estimate_pose(self, object_name):
        if object_name == "cube":
            cube_prim_path = "/World/Cube"
            cube_state = self.env.simulator.get_object_state(cube_prim_path)
            cube_state = self.env.world_to_robot_frame(cube_state["position"], cube_state["orientation"])
            return cube_state[0], cube_state[1]
        elif object_name == "power_drill":
            power_drill_prim_path = "/World/PowerDrill"
            power_drill_state = self.env.simulator.get_object_state(power_drill_prim_path)
            power_drill_state = self.env.world_to_robot_frame(power_drill_state["position"], power_drill_state["orientation"])
            return power_drill_state[0], power_drill_state[1]
        elif object_name == "extrusion":
            extrusion_prim_path = "/World/Extrusion"
            extrusion_state = self.env.simulator.get_object_state(extrusion_prim_path)
            extrusion_state = self.env.world_to_robot_frame(extrusion_state["position"], extrusion_state["orientation"])
            return extrusion_state[0], extrusion_state[1]
        elif object_name == "cone":
            cone_prim_path = "/World/Cone"
            cone_state = self.env.simulator.get_object_state(cone_prim_path)
            cone_state = self.env.world_to_robot_frame(cone_state["position"], cone_state["orientation"])
            return cone_state[0], cone_state[1]
        elif object_name == "bin":
            bin_prim_path = "/World/Bin"
            bin_state = self.env.simulator.get_object_state(bin_prim_path)
            bin_state = self.env.world_to_robot_frame(bin_state["position"], bin_state["orientation"])
            return bin_state[0], bin_state[1]
        elif object_name == "toy":
            toy_prim_path = "/World/Toy"
            toy_state = self.env.simulator.get_object_state(toy_prim_path)
            toy_state = self.env.world_to_robot_frame(toy_state["position"], toy_state["orientation"])
            return toy_state[0], toy_state[1]
        else:
            raise ValueError(f"Unknown object name: {object_name}")

from yolo_seg.seg import YoloSeg
from pose_est.pose_est import PoseEstimator

current_dir = Path(__file__).parent
yolo_seg = YoloSeg(
    model_path=os.path.join(current_dir, "yolo_seg/ckpts/cotrain_all_class_0731_1.pt")
)
pose_estimator = PoseEstimator(
    camera_matrix=[638.315, 637.683, 636.496, 363.410],
    depth_scale=0.001,
    model_scale_factor=None,
    visualize=False,
    log_debug=True,
)

class YoloSegPoseEstimator:
    def __init__(self, env: IOAIEnv):
        self.env = env

    def estimate_pose(self, object_name):
        # get rgb and depth images from front camera
        rgb = self.env.interface.front_head_camera.get_rgb()
        depth = self.env.interface.front_head_camera.get_depth()

        # preprocess depth
        depth = preprocess_depth(
            depth,
            scale=1000,
            min_value=0.0,
            max_value=5 * 1000,
            data_type=np.uint16,
        )
        

        seg_results = yolo_seg.segment_image(rgb)

        for result in seg_results:
            if result.masks is not None:
                print(f"Detected {len(result.masks)} masks in the image.")
            else:
                print("No masks detected.")
        
        best_mask = yolo_seg.get_best_mask(seg_results, object_name)
        if best_mask is not None:
            cv2.imwrite("/tmp/best_mask.png", best_mask * 255)
        else:
            print("No mask found.")

        # pose in camera frame
        pose = pose_estimator.estimate_pose(
            rgb_path="/tmp/rgb_image.png",
            depth_path="/tmp/depth_image.png",
            mask_path="/tmp/best_mask.png",
            cad_name=object_name,
        )

        # pose in robot frame

        return pose

pose_estimator = DummyPoseEstimator(env=env)

# ---- Init State Machine ------
state_machine = SimpleStateMachine(max_states=-1)
state_machine.state_first_entry = True
state_machine.object_name = "cube"

# ----- define your states here -----
# You need to plan a set of waypoints
def move_to_table_state():
    if state_machine.state_first_entry:
        env.move_chassis_follow_path([[0, 4], [0, 2], [0, -0.2]])
        state_machine.state_first_entry = False
    return not env.simulator.physics_callback_exists("follow_path_callback")

def init_state():
    if state_machine.state_first_entry:
        robot_pos = np.array([0.5, 0.1, 0.8])
        robot_ori = np.array([0, 0.7071, 0, 0.7071])
        env.move_left_arm_to_pose(robot_pos, robot_ori)
        state_machine.state_first_entry = False
    return not env.simulator.physics_callback_exists("LeftArm_follow_trajectory_callback")

def front_to_table_state():
    if state_machine.state_first_entry:
        env.move_chassis_rotate(0)
        state_machine.state_first_entry = False
    return not env.simulator.physics_callback_exists("rotate_callback")

def detect_object_state():
    object_name = state_machine.object_name
    pose = pose_estimator.estimate_pose(object_name)
    if pose is None:
        return False
    state_machine.object_pose = pose
    state_machine.grasp_pose = grasp_reg.predict_grasp(object_name, np.concatenate([pose[0], pose[1]]))['grasp_pose']
    # state_machine.grasp_pose = np.concatenate([pose[0], pose[1]])
    return True

def pre_grasp_state():
    if state_machine.state_first_entry:
        robot_pos = state_machine.grasp_pose[:3] + np.array([0, 0, 0.3])
        # robot_ori = [0, 0.7071, 0, 0.7071]
        robot_ori = state_machine.grasp_pose[3:]
        env.move_left_arm_to_pose(robot_pos, robot_ori)
        state_machine.state_first_entry = False
    return not env.simulator.physics_callback_exists("LeftArm_follow_trajectory_callback")

def grasp_state():
    if state_machine.state_first_entry:
        robot_pos = state_machine.grasp_pose[:3] + np.array([0, 0, 0.02])  # A small offset
        # robot_ori = [0, 0.7071, 0, 0.7071]
        robot_ori = state_machine.grasp_pose[3:]
        env.move_left_arm_to_pose(robot_pos, robot_ori)
        state_machine.state_first_entry = False

        print("object_name: ", state_machine.object_name)
        print("grasp_pose: ", state_machine.grasp_pose)
    return not env.simulator.physics_callback_exists("LeftArm_follow_trajectory_callback")

def grasp_object_state():
    if state_machine.state_first_entry:
        env.interface.left_gripper.set_gripper_close()
        state_machine.state_first_entry = False
        state_machine.wait_start_time = time.time()

    elapsed_time = time.time() - state_machine.wait_start_time
    return elapsed_time >= 3

def detect_bin_state():
    if state_machine.state_first_entry:
        pose = pose_estimator.estimate_pose("bin")
        if pose is None:
            return False
        state_machine.bin_pose = pose
        state_machine.state_first_entry = False
        return True

def pre_place_state():
    if state_machine.state_first_entry:
        robot_pos = state_machine.object_pose[0] + np.array([0, 0, 0.4])
        # robot_ori = state_machine.object_pose[1]
        robot_ori = [0, 0.7071, 0, 0.7071]
        env.move_left_arm_to_pose(robot_pos, robot_ori)
        state_machine.state_first_entry = False
    return not env.simulator.physics_callback_exists("LeftArm_follow_trajectory_callback")

def place_state():
    if state_machine.state_first_entry:
        robot_pos = state_machine.bin_pose[0] + np.array([0, 0, 0.4])
        robot_ori = [0, 0, 0, 1]
        env.move_left_arm_to_pose(robot_pos, robot_ori)
        state_machine.state_first_entry = False
    return not env.simulator.physics_callback_exists("LeftArm_follow_trajectory_callback")

def release_state():
    if state_machine.state_first_entry:
        env.interface.left_gripper.set_gripper_open()
        state_machine.state_first_entry = False
        state_machine.wait_start_time = time.time()
    elapsed_time = time.time() - state_machine.wait_start_time
    return elapsed_time >= 3

def return_to_init_state():
    if state_machine.state_first_entry:
        robot_pos = np.array([0.5, 0.1, 0.8])
        robot_ori = np.array([0, 0.7071, 0, 0.7071])
        env.move_left_arm_to_pose(robot_pos, robot_ori)
        state_machine.state_first_entry = False
    return not env.simulator.physics_callback_exists("LeftArm_follow_trajectory_callback")

def move_to_bin_state():
    if state_machine.state_first_entry:
        waypoints_1 = np.linspace([0, 0], [0, 0.9], 30).tolist()
        waypoints_2 = np.linspace([0, 0.9], [0.65, 0.9], 30).tolist()
        waypoints = waypoints_1 + waypoints_2
        env.move_chassis_follow_path(waypoints)
        state_machine.state_first_entry = False
    return not env.simulator.physics_callback_exists("follow_path_callback")

def front_to_bin_state():
    if state_machine.state_first_entry:
        env.move_chassis_rotate(-math.pi / 2)
        state_machine.state_first_entry = False
    return not env.simulator.physics_callback_exists("rotate_callback")

def init_left_arm_joints_state():
    if state_machine.state_first_entry:
        env._move_joints_to_target(env.interface.left_arm, [2.00,-1.60, -0.60, -1.70, 0.00, -0.80, 0.00], 500)
        state_machine.state_first_entry = False
    return not env.simulator.physics_callback_exists("LeftArm_follow_trajectory_callback")

def rotate_to_shelf_state():
    if state_machine.state_first_entry:
        env.move_chassis_rotate(-math.pi / 2)
        state_machine.state_first_entry = False
    return not env.simulator.physics_callback_exists("rotate_callback")

def navigate_to_shelf_state():
    return True

def front_to_shelf_state():
    return True

def place_bin_state():
    return True

# ----- Initialize state machine -----
# States are added in execution order, indices are automatically assigned
MOVE_TO_TABLE_IDX = state_machine.add_state("Move to table", move_to_table_state)
INIT_IDX = state_machine.add_state("Init", init_state)
FRONT_TO_TABLE_IDX = state_machine.add_state("Front to table", front_to_table_state)
DETECT_OBJECT_IDX = state_machine.add_state("Detect Object", detect_object_state)
PRE_GRASP_IDX = state_machine.add_state("Move to Pre Grasp", pre_grasp_state)
GRASP_IDX = state_machine.add_state("Move to Grasp State", grasp_state)
GRASP_OBJECT_IDX = state_machine.add_state("Grasp the object", grasp_object_state)
DETECT_BIN_IDX = state_machine.add_state("Detect bin", detect_bin_state)
PRE_PLACE_IDX = state_machine.add_state("Move to Pre Place State", pre_place_state)
PLACE_IDX = state_machine.add_state("Move to Place State", place_state)
RELEASE_IDX = state_machine.add_state("Release the object", release_state)
RETURN_TO_INIT_IDX = state_machine.add_state("Return to Init State", return_to_init_state)
MOVE_TO_BIN_IDX = state_machine.add_state("Move to bin", move_to_bin_state)
FRONT_TO_BIN_IDX = state_machine.add_state("Front to bin", front_to_bin_state)
INIT_LEFT_ARM_JOINTS_IDX = state_machine.add_state("Init left arm joints", init_left_arm_joints_state)
NAVIGATE_TO_SHELF_IDX = state_machine.add_state("Navigate to shelf", navigate_to_shelf_state)
FRONT_TO_SHELF_IDX = state_machine.add_state("Front to shelf", front_to_shelf_state)
PLACE_BIN_IDX = state_machine.add_state("Place bin", place_bin_state)

# ----- please define your callbacks here -----

def ioai_main_callback():
    # First check if this is a new state (trigger should be called first)
    if state_machine.trigger():
        state_machine.state_first_entry = True
        print(f"Current state: {state_machine.get_state_name()}")
    
    # Then execute current state and move to next when complete
    if state_machine.execute_current_state():
        # TODO: define your custom state transition logic
        if state_machine.state_idx == RELEASE_IDX:
            if state_machine.object_name == "cube":
                state_machine.object_name = "power_drill"
                state_machine.set_state(DETECT_OBJECT_IDX)
                state_machine.state_first_entry = True
            elif state_machine.object_name == "power_drill":
                state_machine.object_name = "extrusion"
                state_machine.set_state(DETECT_OBJECT_IDX)
                state_machine.state_first_entry = True
            elif state_machine.object_name == "extrusion":
                state_machine.object_name = "toy"
                state_machine.set_state(DETECT_OBJECT_IDX)
                state_machine.state_first_entry = True
            elif state_machine.object_name == "toy":
                # Move on
                state_machine.set_state(MOVE_TO_BIN_IDX)
                state_machine.state_first_entry = True
            
        # Default: normal state progression
        else:
            # Normal state progression
            if not state_machine.next():
                # Task completed, reset state machine for next cycle
                print("Task completed!")
                state_machine.reset()


env.simulator.add_physics_callback("ioai_main_callback", ioai_main_callback)

# -----------
env.run()
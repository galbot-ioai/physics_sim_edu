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
# Description: main pipeline for IOAI env
# Author: Chenyu Cao@Galbot
# Date: 2025-07-28
#
#####################################################################################

from ioai_env import IOAIEnv
import numpy as np
from physics_simulator.utils.state_machine import SimpleStateMachine
from physics_simulator.utils.data_types import JointTrajectory

# ---------- Init env ---------
env = IOAIEnv(headless=False)

# ---------- Init Grasp Reg ---------
from grasp_reg import GraspRegistration
grasp_reg = GraspRegistration()

# ---------- Useful functions ---------
def interpolate_joint_positions(start_positions, end_positions, steps):
    return np.linspace(start_positions, end_positions, steps).tolist()

# --------- Dummy Pose Estimator ------
class DummyPoseEstimator:
    def __init__(self):
        pass
    
    def estimate_pose(self, object_name):
        return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])

# ---- Init State Machine ------
state_machine = SimpleStateMachine(max_states=9)
state_machine.state_first_entry = True
state_machine.object_name = "cube"
# ----- define your states here -----
def init_state():
    if state_machine.state_first_entry:
        robot_pos = np.array([0.5, 0.3, 0.7])
        robot_ori = np.array([0, 0.7071, 0, 0.7071])
        env.move_left_arm_to_pose(robot_pos, robot_ori)
        state_machine.state_first_entry = False
    return not env.simulator.physics_callback_exists("LeftArm_follow_trajectory_callback")

# You need to plan a set of waypoints
def move_to_table_state():
    if state_machine.state_first_entry:
        env.move_chassis_follow_path([[0, 4], [0, 2], [0, 0]])
        state_machine.state_first_entry = False
    return np.allclose(env.robot.get_position(), [0, 0, 0], 0.01)


def pre_grasp_state():
    if state_machine.state_first_entry:
        pass
    return True

def grasp_state():
    return True

def grasp_object_state():
    return True

def wait_state():
    return True

def pre_place_state():
    return True

def place_state():
    return True

def release_state():
    return True

def retrun_to_init_state():
    return True

# ----- Initialize state machine -----
state_machine.add_state(0, "Init", init_state)
state_machine.add_state(1, "Move to table", move_to_table_state)
state_machine.add_state(2, "Move to Pre Grasp State", pre_grasp_state)
state_machine.add_state(3, "Move to Grasp State", grasp_state)
state_machine.add_state(4, "Grasp the object", grasp_object_state)
state_machine.add_state(5, "Wait for 3 seconds", wait_state)
state_machine.add_state(6, "Move to Pre Place State", pre_place_state)
state_machine.add_state(7, "Move to Place State", place_state)
state_machine.add_state(8, "Release the object", release_state)
state_machine.add_state(9, "Move to Init State", retrun_to_init_state)

# ----- please define your callbacks here -----

def ioai_main_callback():
    # First check if this is a new state (trigger should be called first)
    if state_machine.trigger():
        state_machine.state_first_entry = True
        print(f"Current state: {state_machine.get_state_name()}")
    
    # Then execute current state and move to next when complete
    if state_machine.execute_current_state():
        # TODO: define your custom state transition logic
        if state_machine.state_idx == 8:
            if state_machine.object_name == "cube":
                state_machine.object_name = "power_drill"
                state_machine.state_idx = 1
            elif state_machine.object_name == "power_drill":
                state_machine.object_name = "extrusion"
                state_machine.state_idx = 1
            elif state_machine.object_name == "extrusion":
                # Move on
                pass
            
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
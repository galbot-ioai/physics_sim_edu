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
# Module: Mujoco Robot
# Description: Wrapper class for robots in MuJoCo
# Author: Herman Ye@Galbot
# Date: 2025-04-07
# Note: This file is a modified version of robosuite
#
#####################################################################################

import mujoco
from physics_simulator.robot.robot_model import MujocoRobotModel
# from physics_simulator import PhysicsSimulator
import numpy as np
from synthnova_config import RobotConfig
from typing import Dict, List, Tuple, Optional
from auro_utils import wxyz_to_xyzw

class MujocoRobot(object):
    """A wrapper class for manipulating robots in MuJoCo simulations.
    
    Provides methods for controlling joint positions, getting state information,
    and performing forward kinematics.
    """

    def __init__(self,
                 sim,
                 robot_config: RobotConfig,
                 namespace: str = ""):
        """Initialize a MuJoCo robot.
        
        Args:
            sim: The physics simulator instance
            robot_config: Configuration for the robot
            namespace: Optional namespace for the robot
        """
        self.sim = sim
        self.config = robot_config
        self.prim_path = robot_config.prim_path
        self.namespace = namespace
        self.name = robot_config.name
        self.joint_names = None
        self.joint_indexes = None
        self.model = None
        self.data = None
        # Store initial world pose until MuJoCo is initialized
        self.world_pose = [0, 0, 0, 0, 0, 0, 1]
        # Store velocities until MuJoCo is initialized
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        
        translation = robot_config.translation
        rotation = robot_config.rotation

        # Global pose
        position = robot_config.position
        orientation = robot_config.orientation

        self.robot_model = MujocoRobotModel(self.config.mjcf_path, namespace)
        if position is not None and orientation is not None:
            self.set_world_pose(np.concatenate([position, orientation]))
        elif translation is not None and rotation is not None:
            self.set_world_pose(np.concatenate([translation, rotation]))
        else:
            raise ValueError("Invalid pose input")
        
        # Store target positions for non-immediate joint control
        self.joint_position_targets = {}
        self.position_control_mode = {}

    def initialize(self):
        """Initialize the robot with MuJoCo model and data references.
        Must be called after the simulator is initialized."""
        self.model = self.sim.model
        self.data = self.sim.data
        # Store direct references to the underlying MuJoCo model and data
        self.mujoco_model = self.model._model
        self.mujoco_data = self.data._data
        
        # Initialize robot model with mujoco model and data
        self.robot_model.initialize(self.mujoco_model, self.mujoco_data)
        
        # Get joint names and indexes
        self.joint_names = self.robot_model.joint_names
        self.joint_indexes = self.get_joint_indexes(self.joint_names)
        
        # Find the robot's root/base body (first body that's not 'world' or 'ground')
        self.root_body_id = None
        self.root_body_name = None
        
        # Try to find the root body using the robot's name as a prefix
        for i in range(self.mujoco_model.nbody):
            body_name = mujoco.mj_id2name(self.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and body_name.startswith(self.name) and body_name != 'world' and body_name != 'ground':
                self.root_body_id = i
                self.root_body_name = body_name
                break
        
        # If we didn't find a body with the robot name, try to find the first non-world, non-ground body
        if self.root_body_name is None:
            for i in range(self.mujoco_model.nbody):
                body_name = mujoco.mj_id2name(self.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and body_name != 'world' and body_name != 'ground':
                    self.root_body_id = i
                    self.root_body_name = body_name
                    break
                    
        # If we still didn't find a valid body, log a warning
        if self.root_body_name is None:
            self.sim.logger.log_warning(f"Could not find a valid root body for robot {self.name}. Using body ID 1.")
            self.root_body_id = 1
            self.root_body_name = mujoco.mj_id2name(self.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, 1) or "unknown"

    def set_world_pose(self, pose):
        """Set the position and orientation of the robot in world coordinates.
        
        Args:
            pose: Array containing [x, y, z, qx, qy, qz, qw] position and orientation
        """
        position, orientation = pose[:3], pose[3:]
        # Store pose for use before MuJoCo initialization
        self.world_pose = pose.copy() if isinstance(pose, np.ndarray) else np.array(pose)
        
        # Update MuJoCo mocap data if available
        if hasattr(self, 'data') and self.data is not None:
            self.data.set_mocap_pos(self.root_body_name, position)
            self.data.set_mocap_quat(self.root_body_name, orientation)
        
        # Update robot model if initialized
        if hasattr(self, 'robot_model'):
            self.robot_model.set_base_position(position)
            self.robot_model.set_base_orientation(orientation)
        else:
            self.sim.logger.log_warning(f"WARNING: Robot model not initialized. Cannot set world pose for robot {self.name}.")

    #NOTE[DEPRECATED]: This method is deprecated and will be removed in the future.
    def set_world_velocity(self, velocity):
        """Set the velocity of the robot in world coordinates.
        
        Args:
            velocity: Array containing [vx, vy, vz, wx, wy, wz] velocity
        """
        if not hasattr(self, 'data') or self.data is None:
            return
            
        # Store velocities
        self.linear_velocity = velocity[:3]
        self.angular_velocity = velocity[3:]

        joint_velocities = {
            self.chassis: velocity[0],  # vx
            'mobile_side_joint': velocity[1],     # vy
            'mobile_yaw_joint': velocity[5]       # wz
        }
        
        # Set joint velocities
        for joint_name, joint_vel in joint_velocities.items():
            joint_id = self.model.body_name2id(joint_name)
            if joint_id >= 0:
                dof_idx = self.model.jnt_dofadr[joint_id]
                self.data.qvel[dof_idx] = joint_vel

    def get_world_pose(self):
        """Get the position and orientation of the robot in world coordinates.

        Returns:
            tuple: (position, orientation) where:
                - position: 3D position [x, y, z]
                - orientation: Quaternion [qx, qy, qz, qw]
        """
            
        # Get position and orientation from the root body
        position = self.data.xpos[self.root_body_id].copy()
        orientation = self.data.xquat[self.root_body_id].copy()
        
        # Convert from wxyz to xyzw quaternion format
        orientation = wxyz_to_xyzw(orientation)
            
        return position, orientation

    def get_joint_dof_index(self, joint_name):
        """Get the DOF index for a joint by name for velocity operations.
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            int: DOF index in qvel array, or -1 if not found
        """
        joint_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            return self.mujoco_model.jnt_dofadr[joint_id]
        return -1

    def get_joint_index(self, joint_name):
        """Get the index for a joint by name in MuJoCo's qpos array.
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            int: Joint index in qpos array
        """
        joint_id = self.model.joint_name2id(joint_name)
        return self.model.jnt_qposadr[joint_id]

    def get_joint_indexes(self, joint_names):
        """Get the indexes for a list of joints by name."""
        return [
            self.get_joint_index(joint_name) for joint_name in joint_names
        ]

    def get_dof_index(self, joint_name):
        """Get the DOF index for a joint by name (alias for get_joint_dof_index)."""
        return self.get_joint_dof_index(joint_name)

    def get_joint_names(self):
        """Get all joint names from the robot.
        
        Returns:
            list: List of joint names defined in the robot model
        """
        return self.joint_names

    def get_joint_positions(self, joint_names=None):
        """Get current positions of specified joints.
        
        Args:
            joint_names: List of joint names to query (defaults to all joints)
            
        Returns:
            numpy.ndarray: Array of joint positions in radians/meters
        """
        if joint_names is None:
            joint_names = self.joint_names
            
        joint_positions = []
        for joint_name in joint_names:
            joint_idx = self.get_joint_index(joint_name)
            if joint_idx >= 0:
                joint_positions.append(self.data.qpos[joint_idx])
            else:
                raise ValueError(f"Joint {joint_name} not found")
                
        return np.array(joint_positions)

    def get_joint_velocities(self, joint_names=None):
        """Get current velocities of specified joints.
        
        Args:
            joint_names: List of joint names to query (defaults to all joints)
            
        Returns:
            numpy.ndarray: Array of joint velocities
        """
        if joint_names is None:
            joint_names = self.joint_names
            
        joint_velocities = []
        for joint_name in joint_names:
            dof_idx = self.get_joint_dof_index(joint_name)
            if dof_idx >= 0:
                joint_velocities.append(self.data.qvel[dof_idx])
            else:
                joint_velocities.append(0.0)
                
        return np.array(joint_velocities)

    def set_joint_positions(self, positions, joint_names=None, immediate=False):
        """Set positions of specified joints.
        
        Args:
            positions: List of target positions for each joint
            joint_names: List of joint names to set positions for
            immediate: If True, set positions immediately without interpolation
        """

        if joint_names is None:
            joint_names = self.joint_names

        # Process each joint and its target position
        for joint_name, position in zip(joint_names, positions):
            if joint_name not in self.joint_names:
                raise ValueError(f"Joint {joint_name} not found")
                
            if immediate:
                # Direct position setting + sync actuator control
                joint_idx = self.get_joint_index(joint_name)
                self.data.qpos[joint_idx] = position
                # Also update actuator to maintain the position
                actuator_id = self.model.actuator_name2id(joint_name)
                if actuator_id >= 0:
                    self.data.ctrl[actuator_id] = position
            else:
                # Try actuator control first, fallback to direct setting
                actuator_id = self.model.actuator_name2id(joint_name)
                if actuator_id >= 0:
                    self.data.ctrl[actuator_id] = position
                else:
                    joint_idx = self.get_joint_index(joint_name)
                    self.data.qpos[joint_idx] = position

        if immediate:
            mujoco.mj_forward(self.mujoco_model, self.mujoco_data)

    def set_joint_velocities(self, velocities, joint_names=None, immediate=False):
        """Set velocities for specified joints.
        
        Args:
            velocities: List of velocity values to set
            joint_names: List of joint names to set velocities for
            immediate: If True, update velocities immediately
            
        Raises:
            ValueError: If a joint is not found
        """
        if joint_names is None:
            joint_names = self.joint_names

        for joint_name, velocity in zip(joint_names, velocities):
            if joint_name not in self.joint_names:
                raise ValueError(f"Joint {joint_name} not found")
                
            if immediate:
                # Direct velocity setting + sync actuator control
                dof_idx = self.get_joint_dof_index(joint_name)
                if dof_idx >= 0:
                    self.data.qvel[dof_idx] = velocity
                # Also update actuator to maintain the velocity
                actuator_id = self.model.actuator_name2id(joint_name)
                if actuator_id >= 0:
                    self.data.ctrl[actuator_id] = velocity
            else:
                # Try actuator control first, fallback to direct setting
                actuator_id = self.model.actuator_name2id(joint_name)
                if actuator_id >= 0:
                    self.data.ctrl[actuator_id] = velocity
                else:
                    dof_idx = self.get_joint_dof_index(joint_name)
                    if dof_idx >= 0:
                        self.data.qvel[dof_idx] = velocity

        if immediate:
            mujoco.mj_forward(self.mujoco_model, self.mujoco_data)

    def get_state(self):
        """Get the complete state of the robot.
        
        Returns:
            dict: Robot state including position, orientation, joint positions, and velocities
        """
        pose = self.get_world_pose()

        # Build state dictionary
        state = {
            "position": pose[0],
            "orientation": pose[1],
            "translation": pose[0],  # Local translation is same as position for simplicity
            "rotation": pose[1],
            "joint_positions": self.get_joint_positions(),
            "joint_velocities": self.get_joint_velocities(),
        }

        return state
        
    def get_position(self):
        """Get the position of the robot in world frame.
        
        Returns:
            np.ndarray: 3D position vector [x, y, z]
        """
        return self.get_world_pose()[0]
        
    def get_orientation(self):
        """Get the orientation of the robot in world frame.
        
        Returns:
            np.ndarray: Quaternion in xyzw format [qx, qy, qz, qw]
        """
        return self.get_world_pose()[1]
        
    def get_translation(self):
        """Get the translation of the robot in local frame.
        
        Returns:
            np.ndarray: 3D position vector [x, y, z]
        """
        return self.get_position()
        
    def get_rotation(self):
        """Get the rotation of the robot in local frame.
        
        Returns:
            np.ndarray: Quaternion in xyzw format [qx, qy, qz, qw]
        """
        return self.get_orientation()
        
    def get_all_joint_positions(self):
        """Get positions of all joints defined in the robot model.
        
        Returns:
            np.ndarray: Array of joint positions in radians or meters
        """
        return self.get_joint_positions()
        
    def get_all_joint_velocities(self):
        """Get velocities of all joints defined in the robot model.
        
        Returns:
            np.ndarray: Array of joint velocities in radians/sec or meters/sec
        """
        return self.get_joint_velocities()

    def get_articulation_controller(self):
        """Get the controller for this robot.
        
        This is a compatibility method for API consistency with other simulators.
        
        Returns:
            MujocoRobot: Self-reference to this robot instance
        """
        return self

    def apply_action(self, action):
        """Apply an articulation action to control the robot.
        
        Applies joint positions and/or velocities from the action object.
        
        Args:
            action: Action object containing joint_positions, joint_velocities,
                   and joint_names attributes
        """
        # Extract joint positions and velocities from action
        joint_positions = getattr(action, "joint_positions", None)
        joint_velocities = getattr(action, "joint_velocities", None)
        joint_names = getattr(action, "joint_names", None)

        # Apply positions if specified
        if joint_positions is not None and joint_names is not None:
            self.set_joint_positions(joint_positions, joint_names, immediate=False)

        # Apply velocities if specified
        if joint_velocities is not None and joint_names is not None:
            self.set_joint_velocities(joint_velocities, joint_names, immediate=False)
            
    # Forward kinematics methods
    
    def fk_link(self, q: np.ndarray, link: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics for a specific link.
        
        Calculates the position and orientation of a link given a set of joint angles.
        
        Args:
            q: Joint angle configuration array
            link: Name of the target link
            
        Returns:
            tuple: (position, rotation_matrix) where:
                - position: 3D position vector of the link
                - rotation_matrix: 3x3 rotation matrix of the link
        """
        return self.robot_model.fk_link(q, link)
    
    def fk_all_link(self, q: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compute forward kinematics for all links in the robot.
        
        Calculates the position and orientation of all links given a set of joint angles.
        
        Args:
            q: Joint angle configuration array
            
        Returns:
            tuple: (positions, rotation_matrices) where:
                - positions: Dictionary mapping link names to 3D position vectors
                - rotation_matrices: Dictionary mapping link names to 3x3 rotation matrices
        """
        return self.robot_model.fk_all_link(q)
    
    def fk_jacobian(self, q: np.ndarray, link: str) -> np.ndarray:
        """Compute the Jacobian matrix for a specific link.
        
        The Jacobian maps joint velocities to end-effector velocities (linear and angular).
        
        Args:
            q: Joint angle configuration array
            link: Name of the target link
            
        Returns:
            np.ndarray: 6xN Jacobian matrix where N is the number of joints
                        (first 3 rows for linear velocity, last 3 for angular velocity)
        """
        return self.robot_model.fk_jacobian(q, link)
    
    def generate_random_qpos(self) -> np.ndarray:
        """Generate a random joint configuration within joint limits.
        
        Returns:
            np.ndarray: Random joint configuration array
        """
        return self.robot_model.generate_random_qpos()
    
    def clip_qpos(self, q: np.ndarray) -> np.ndarray:
        """Clip a joint configuration to respect joint limits.
        
        Args:
            q: Joint configuration to clip
            
        Returns:
            np.ndarray: Clipped joint configuration that respects joint limits
        """
        return self.robot_model.clip_qpos(q)

    def get_linear_velocities(self):
        """Get the linear velocities of the robot bodies.
        
        Returns:
            numpy.ndarray: Array of linear velocity vectors for each body
        """
        body_id = self.root_body_id
        
        self.linear_velocity = self.data.cvel[body_id][3:].copy()
        return [self.linear_velocity]

    def get_angular_velocities(self):
        """Get the angular velocities of the robot bodies.
        
        Returns:
            numpy.ndarray: Array of angular velocity vectors for each body
        """
        body_id = self.root_body_id
        
        self.angular_velocity = self.data.cvel[body_id][:3].copy()
        return [self.angular_velocity]
    
    # TODO@Chenyu: Add articulation controller
    def get_articulation_controller(self):
        """Get the articulation controller for this robot.
        
        Returns:
            MujocoRobot: Self-reference to this robot instance
        """
        return self
    
    def get_body_world_pose(self, link_name: str):
        """Get the world pose of a specific robot body.
        
        Args:
            link_name: Name of the link to get the world pose for
            
        Returns:
            numpy.ndarray: Array containing the world pose [x, y, z, qw, qx, qy, qz]
                where [x, y, z] is position and [qw, qx, qy, qz] is quaternion orientation.
        """
        return self.robot_model.get_body_world_pose(link_name)
    
    def get_all_body_world_poses(self):
        """Get the world poses of all robot bodies.
        
        Returns:
            Dict[str, numpy.ndarray]: Dictionary mapping body names to their world poses.
                Each pose is a 7-element array [x, y, z, qw, qx, qy, qz] 
                where [x, y, z] is position and [qw, qx, qy, qz] is quaternion orientation.
        """
        return self.robot_model.get_all_body_world_poses()

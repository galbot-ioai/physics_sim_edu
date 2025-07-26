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
# Module: Mujoco Robot Model
# Description: Wrapper class for robot model in MuJoCo
# Author: Herman Ye@Galbot
# Date: 2025-04-07
# Note: This file is a modified version of robosuite
#
#####################################################################################


import mujoco
from auro_utils.math.transform import xyzw_to_wxyz, wxyz_to_xyzw
from physics_simulator.object import MujocoXMLModel
# from physics_simulator.robot.gripper_model import GripperModel
from physics_simulator.utils.mjcf_utils import (
    MOUNT_COLLISION_COLOR,
    array_to_string
)
import numpy as np
from typing import Dict, List, Tuple, Optional


class MujocoRobotModel(MujocoXMLModel):
    """Wrapper class for robot model in MuJoCo."""

    def __init__(self, mjcf_file_path: str, namespace: str):
        self.namespace = namespace
        super().__init__(mjcf_file_path, namespace)
        
        # Robot properties
        self.joint_names = []
        self.joints_lower_limit = None
        self.joints_upper_limit = None
        self.links_name = []
        self._actuators = []  # Use private attribute with property
        self._mujoco_model = None
        self._mujoco_data = None
        self._root_body = None  # Use private attribute with property
        self._bottom_offset = np.array([0.0, 0.0, 0.0])  # Default bottom offset
    
    @property
    def actuators(self):
        """Get actuator names"""
        return self._actuators
    
    @property
    def root_body(self):
        """Get root body name"""
        return self._root_body
    
    @root_body.setter
    def root_body(self, value):
        """Set root body name"""
        self._root_body = value
    
    @property
    def bottom_offset(self):
        """Get bottom offset for robot base positioning"""
        return self._bottom_offset
    
    @bottom_offset.setter
    def bottom_offset(self, value):
        """Set bottom offset for robot base positioning"""
        self._bottom_offset = np.array(value)
        
    def initialize(self, model, data):
        """
        Initialize model with MuJoCo model and data.
        
        Parameters
        ----------
        model: mujoco.MjModel
            MuJoCo model
        data: mujoco.MjData
            MuJoCo data
        """
        self._mujoco_model = model
        self._mujoco_data = data
        
        # Get all joint information, filtering by namespace
        self.joint_names = []
        for i in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name and (self.namespace == "" or joint_name.startswith(self.namespace)):
                self.joint_names.append(joint_name)
        
        # Get actuator information (these are separate from joints)
        self._actuators = []
        for i in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name and (self.namespace == "" or actuator_name.startswith(self.namespace)):
                self._actuators.append(actuator_name)
        
        # Get joint limits for all joints
        if len(self.joint_names) > 0:
            self.joints_lower_limit = np.zeros(len(self.joint_names))
            self.joints_upper_limit = np.zeros(len(self.joint_names))
            
            for i, joint_name in enumerate(self.joint_names):
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0 and joint_id < model.njnt:
                    # Check if joint has limits
                    if model.jnt_limited[joint_id]:
                        self.joints_lower_limit[i] = model.jnt_range[joint_id, 0]
                        self.joints_upper_limit[i] = model.jnt_range[joint_id, 1]
                    else:
                        # For unlimited joints, use a large range
                        self.joints_lower_limit[i] = -10.0
                        self.joints_upper_limit[i] = 10.0
        
        # Get link information (bodies)
        self.links_name = []
        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and body_name != "world" and (self.namespace == "" or body_name.startswith(self.namespace)):
                self.links_name.append(body_name)
                
        # Root body is the first link (after world)
        if len(self.links_name) > 0:
            self._root_body = self.links_name[0]

    @property
    def contact_geom_rgba(self) -> np.ndarray:
        """Get contact geometry RGBA color as numpy array"""
        return np.array(MOUNT_COLLISION_COLOR)

    @property
    def naming_prefix(self) -> str:
        # Use the provided namespace as the naming prefix to avoid name collisions
        return self.namespace

    def set_base_position(self, pos: np.ndarray):
        """
        Places the robot on position @pos.

        Args:
            pos (3-array): (x,y,z) position to place robot base
        """
        self._elements["root_body"].set(
            "pos", array_to_string(pos - self.bottom_offset)
        )

    def set_base_orientation(self, rot: np.ndarray):
        """
        Rotates robot by rotation @rot from its original orientation.

        Args:
            rot (4-array): (x,y,z,w) quaternion specifying the orientation for the robot base
        """
        # xml quat assumes w,x,y,z so we need to convert to this format from outputted x,y,z,w format from fcn
        rot = xyzw_to_wxyz(rot)
        self._elements["root_body"].set(
            "quat", array_to_string(rot)
        )

    def get_body_world_pose(self, link_name: str):
        """Get the world pose of a specific robot body.
        
        Args:
            link_name: Name of the body/link
            
        Returns:
            dict: World pose dictionary with:
                - position: numpy.ndarray [x, y, z] - 3D position
                - orientation: numpy.ndarray [qx, qy, qz, qw] - quaternion orientation
        """

        body_id = mujoco.mj_name2id(self._mujoco_model, mujoco.mjtObj.mjOBJ_BODY, self.naming_prefix + link_name)
        if body_id < 0:
            raise ValueError(f"Link {link_name} not found in model")
            
        link_pos = self._mujoco_data.xpos[body_id].copy()
        link_quat = self._mujoco_data.xquat[body_id].copy()
        link_quat = wxyz_to_xyzw(link_quat)
        
        return {
            "position": link_pos,
            "orientation": link_quat
        }
    
    def get_all_body_world_poses(self):
        """Get the world poses of all robot bodies.
        
        Returns:
            Dict[str, dict]: Dictionary mapping body names to their world pose dictionaries.
                Each pose dictionary contains:
                - position: numpy.ndarray [x, y, z] - 3D position
                - orientation: numpy.ndarray [qx, qy, qz, qw] - quaternion orientation
                Body names in the dictionary keys have the naming prefix removed.
        """
        body_poses = {}
        for link_name in self.links_name:
            # Remove naming prefix from the key while keeping full name for pose retrieval
            clean_link_name = link_name
            if self.naming_prefix and link_name.startswith(self.naming_prefix):
                clean_link_name = link_name[len(self.naming_prefix):]
            body_poses[clean_link_name] = self.get_body_world_pose(link_name)
        
        return body_poses
    
    def fk_link(self, q: np.ndarray, link: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the forward kinematics for a specific link.

        Parameters
        ----------
        q: np.ndarray (J,)
            The joint angles.
        link: str
            The link name.

        Returns
        -------
        link_trans: np.ndarray (3,)
            The translation of the link.
        link_rot: np.ndarray (3, 3)
            The rotation of the link.
        """
        if self._mujoco_model is None or self._mujoco_data is None:
            raise ValueError("MuJoCo model and data not initialized")
            
        # Save original positions
        original_qpos = self._mujoco_data.qpos.copy()
        
        # Set joint positions
        for i, joint_name in enumerate(self.joint_names):
            if i < len(q):
                joint_id = mujoco.mj_name2id(self._mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    self._mujoco_data.qpos[self._mujoco_model.jnt_qposadr[joint_id]] = q[i]
        
        # Forward kinematics
        mujoco.mj_forward(self._mujoco_model, self._mujoco_data)
        
        # Get link position and orientation
        body_id = mujoco.mj_name2id(self._mujoco_model, mujoco.mjtObj.mjOBJ_BODY, link)
        if body_id < 0:
            raise ValueError(f"Link {link} not found in model")
            
        link_trans = self._mujoco_data.xpos[body_id].copy()
        link_rot = self._mujoco_data.xmat[body_id].reshape(3, 3).copy()
        
        # Restore original positions
        self._mujoco_data.qpos[:] = original_qpos
        mujoco.mj_forward(self._mujoco_model, self._mujoco_data)
        
        return link_trans, link_rot
    
    def fk_all_link(self, q: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Compute the forward kinematics for all links.

        Parameters
        ----------
        q: np.ndarray (J,)
            The joint angles.

        Returns
        -------
        link_trans: Dict[str, np.ndarray (3,)]
            The translation of each link.
        link_rot: Dict[str, np.ndarray (3, 3)]
            The rotation of each link.
        """
        if self._mujoco_model is None or self._mujoco_data is None:
            raise ValueError("MuJoCo model and data not initialized")
            
        # Save original positions
        original_qpos = self._mujoco_data.qpos.copy()
        
        # Set joint positions
        for i, joint_name in enumerate(self.joint_names):
            if i < len(q):
                joint_id = mujoco.mj_name2id(self._mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    self._mujoco_data.qpos[self._mujoco_model.jnt_qposadr[joint_id]] = q[i]
        
        # Forward kinematics
        mujoco.mj_forward(self._mujoco_model, self._mujoco_data)
        
        # Get all link positions and orientations
        link_trans = {}
        link_rot = {}
        
        for link in self.links_name:
            body_id = mujoco.mj_name2id(self._mujoco_model, mujoco.mjtObj.mjOBJ_BODY, link)
            if body_id >= 0:
                link_trans[link] = self._mujoco_data.xpos[body_id].copy()
                link_rot[link] = self._mujoco_data.xmat[body_id].reshape(3, 3).copy()
        
        # Restore original positions
        self._mujoco_data.qpos[:] = original_qpos
        mujoco.mj_forward(self._mujoco_model, self._mujoco_data)
        
        return link_trans, link_rot
    
    def fk_jacobian(self, q: np.ndarray, link: str) -> np.ndarray:
        """
        Compute the forward kinematics jacobian for a specific link.

        Parameters
        ----------
        q: np.ndarray (J,)
            The joint angles.
        link: str
            The link name.

        Returns
        -------
        link_jacobian: np.ndarray (6, J)
            The jacobian of the link.
        """
        if self._mujoco_model is None or self._mujoco_data is None:
            raise ValueError("MuJoCo model and data not initialized")
            
        # Save original positions
        original_qpos = self._mujoco_data.qpos.copy()
        
        # Set joint positions
        for i, joint_name in enumerate(self.joint_names):
            if i < len(q):
                joint_id = mujoco.mj_name2id(self._mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    self._mujoco_data.qpos[self._mujoco_model.jnt_qposadr[joint_id]] = q[i]
        
        # Forward kinematics
        mujoco.mj_forward(self._mujoco_model, self._mujoco_data)
        
        # Get body ID for the link
        body_id = mujoco.mj_name2id(self._mujoco_model, mujoco.mjtObj.mjOBJ_BODY, link)
        if body_id < 0:
            raise ValueError(f"Link {link} not found in model")
            
        # Compute Jacobian
        jacp = np.zeros((3, self._mujoco_model.nv))
        jacr = np.zeros((3, self._mujoco_model.nv))
        
        mujoco.mj_jacBody(self._mujoco_model, self._mujoco_data, jacp, jacr, body_id)
        
        # Combine translational and rotational Jacobians
        jacobian = np.vstack((jacp, jacr))
        
        # Restore original positions
        self._mujoco_data.qpos[:] = original_qpos
        mujoco.mj_forward(self._mujoco_model, self._mujoco_data)
        
        return jacobian
    
    def generate_random_qpos(self) -> np.ndarray:
        """
        Generate a random joint configuration.

        Returns
        -------
        qpos: np.ndarray (J,)
            The random joint configuration.
        """
        if self.joints_lower_limit is None or self.joints_upper_limit is None:
            raise ValueError("Joint limits not initialized")
            
        return (
            np.random.rand(len(self.joints_lower_limit))
            * (self.joints_upper_limit - self.joints_lower_limit)
            + self.joints_lower_limit
        )
    
    def clip_qpos(self, q: np.ndarray) -> np.ndarray:
        """
        Clip the joint configuration to the joint limits.

        Parameters
        ----------
        q: np.ndarray (J,)
            The joint configuration.

        Returns
        -------
        clipped_qpos: np.ndarray (J,)
            The clipped joint configuration.
        """
        if self.joints_lower_limit is None or self.joints_upper_limit is None:
            raise ValueError("Joint limits not initialized")
            
        return np.clip(q, self.joints_lower_limit, self.joints_upper_limit)

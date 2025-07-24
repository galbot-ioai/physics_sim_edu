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
from auro_utils.math.transform import xyzw_to_wxyz
from auro_utils.math.transform import wxyz_to_xyzw
from physics_simulator.utils.mjcf_utils import (
    MOUNT_COLLISION_COLOR,
    array_to_string,
    find_elements,
    find_parent,
    new_body,
    new_site,
)
import numpy as np
import xml.etree.ElementTree as ET
from physics_simulator.object import MujocoObject
from auro_utils import Logger


class MujocoSensorModel(MujocoObject):
    """Wrapper class for sensor model in MuJoCo.
    
    This class represents a sensor (e.g. camera) that can be mounted on a robot link.
    It inherits from MujocoObject to better integrate with the existing framework,
    but maintains a lightweight representation with just position and orientation.
    """

    def __init__(self, name: str, parent_entity_name: str, camera_name: str = None, position: np.ndarray = None, orientation: np.ndarray = None, translation: np.ndarray = None, rotation: np.ndarray = None, fov: float = None, width: int = None, height: int = None):
        """
        Initialize a sensor model.

        Args:
            name (str): Name of the sensor model
            parent_entity_name (str): Name of the parent link to mount the sensor on
            camera_name (str, optional): Name of the camera element. If None, will use the sensor model name. Defaults to None.
            position (np.ndarray, optional): Global position of the sensor. Defaults to None.
            orientation (np.ndarray, optional): Global orientation of the sensor (xyzw quaternion). Defaults to None.
            translation (np.ndarray, optional): Local position of the sensor relative to parent. Defaults to None.
            rotation (np.ndarray, optional): Local orientation of the sensor relative to parent (xyzw quaternion). Defaults to None.
            fov (float, optional): Field of view in degrees. Defaults to None.
            width (int, optional): Image width in pixels. Defaults to None.
            height (int, optional): Image height in pixels. Defaults to None.
        """
        super().__init__(obj_type="visual")  # Camera is purely visual
        self._name = name
        self.camera_name = camera_name if camera_name is not None else name
        self.parent_entity_name = parent_entity_name
        
        # Handle position and orientation
        if position is not None and orientation is not None:
            self.position = position
            self.orientation = orientation
        elif translation is not None and rotation is not None:
            self.position = translation
            self.orientation = rotation
        else:
            self.position = np.zeros(3)
            self.orientation = np.array([1, 0, 0, 0])  # Identity quaternion
            
        # Import constants for default values
        from physics_simulator.utils.constants import (
            DEFAULT_CAMERA_FOV, 
            DEFAULT_CAMERA_WIDTH, 
            DEFAULT_CAMERA_HEIGHT
        )
        
        self.fov = fov if fov is not None else DEFAULT_CAMERA_FOV
        self.width = width if width is not None else DEFAULT_CAMERA_WIDTH
        self.height = height if height is not None else DEFAULT_CAMERA_HEIGHT
        
        # Create a minimal object representation
        self._obj = self._get_object_subtree()

    def _get_object_subtree(self):
        """
        Creates a minimal object representation with just a camera.
        """
        # Create a body element with prefixed name
        body = new_body(name=f"{self.naming_prefix}main")
        
        # Create a camera element
        camera = ET.Element("camera")
        camera.set("name", self.camera_name)
        camera.set("pos", array_to_string(self.position))
        camera.set("quat", array_to_string(self.orientation))
        camera.set("fovy", str(self.fov))
        camera.set("resolution", f"{self.width} {self.height}")
        
        # Add camera to body
        body.append(camera)
        
        # Add a default site with prefixed name
        site = new_site(name=f"{self.naming_prefix}default_site")
        body.append(site)
        
        return body

    def exclude_from_prefixing(self, inp):
        """
        By default, don't exclude any from being prefixed
        """
        return False

    def mount_to_parent(self, worldbody):
        """
        Mounts the sensor to its parent entity (usually a robot link).
        This method should be called after the parent entity is loaded into the simulation.
        
        Args:
            worldbody: The worldbody element of the MuJoCo XML model
        """
        # Find the parent body (link) in the worldbody
        parent_body = find_elements(
            root=worldbody,
            tags="body",
            attribs={"name": self.parent_entity_name},
            return_first=True
        )
        
        if parent_body is None:
            raise ValueError(f"Parent body (link) {self.parent_entity_name} not found in the worldbody")
        
        # Add the sensor body as a child of the parent body
        parent_body.append(self._obj)

    def set_position(self, position: np.ndarray):
        """
        Set the position of the sensor relative to its parent link.

        Args:
            position (np.ndarray): 3D position vector
        """
        self.position = position
        camera = self._obj.find("./camera")
        if camera is not None:
            camera.set("pos", array_to_string(position))

    def set_orientation(self, orientation: np.ndarray):
        """
        Set the orientation of the sensor relative to its parent link.

        Args:
            orientation (np.ndarray): xyzw quaternion
        """
        self.orientation = orientation
        camera = self._obj.find("./camera")
        if camera is not None:
            camera.set("quat", array_to_string(xyzw_to_wxyz(orientation)))

    @property
    def bottom_offset(self):
        """
        Returns vector from model root body to model bottom.
        For camera, this is just the position.
        """
        return self.position

    @property
    def top_offset(self):
        """
        Returns vector from model root body to model top.
        For camera, this is just the position.
        """
        return self.position

    @property
    def horizontal_radius(self):
        """
        Returns maximum distance from model root body to any radial point of the model.
        For camera, this is 0 since it's a point.
        """
        return 0.0

    @property
    def contact_geom_rgba(self) -> np.array:
        return MOUNT_COLLISION_COLOR

    @property
    def naming_prefix(self) -> str:
        return self._name + "_"

    def set_base_position(self, pos: np.ndarray):
        """
        Places the robot on position @pos.

        Args:
            pos (3-array): (x,y,z) position to place robot base
        """
        self.position = pos - self.bottom_offset

    def set_base_orientation(self, rot: np.ndarray):
        """
        Rotates robot by rotation @rot from its original orientation.

        Args:
            rot (4-array): (x,y,z,w) quaternion specifying the orientation for the robot base
        """
        # xml quat assumes w,x,y,z so we need to convert to this format from outputted x,y,z,w format from fcn
        self.orientation = xyzw_to_wxyz(rot)
        
    def get_position(self):
        """Get the position of the object in world frame.
        
        Note: This returns stored position. For runtime position queries,
        use simulator.get_sensor_state() instead.
        
        Returns:
            np.ndarray: 3D position [x, y, z]
        """
        # Return stored position - actual position should be queried through simulator
        # This method is kept for API compatibility but should not be used for runtime queries
        
        # Create a temporary logger for deprecation warning
        logger = Logger()
        logger.log_warning(
            "get_position() called on sensor - this returns stored position only. "
            "Use simulator.get_sensor_state() for runtime position."
        )
        return self.position
        
    def get_orientation(self):
        """Get the orientation of the object in world frame.
        
        Note: This returns stored orientation. For runtime orientation queries,
        use simulator.get_sensor_state() instead.
        
        Returns:
            np.ndarray: Quaternion in xyzw format
        """
        # Return stored orientation - actual orientation should be queried through simulator  
        # This method is kept for API compatibility but should not be used for runtime queries
        
        # Create a temporary logger for deprecation warning
        logger = Logger()
        logger.log_warning(
            "get_orientation() called on sensor - this returns stored orientation only. "
            "Use simulator.get_sensor_state() for runtime orientation."
        )
        return wxyz_to_xyzw(self.orientation) if self.orientation is not None else np.array([0.0, 0.0, 0.0, 1.0])

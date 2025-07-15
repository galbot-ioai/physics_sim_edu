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
# Description: Factory class for creating Mujoco objects from ObjectConfig
# Author: Chenyu Cao@Galbot
# Date: 2025-04-02
#
#####################################################################################

import numpy as np
from typing import Union, Optional, List, Any

# Config
from synthnova_config import (
    ObjectConfig,
    CuboidConfig,
    MeshConfig
)

from physics_simulator.object import MujocoXMLObject
from physics_simulator.object.primitive.box import BoxObject


class MujocoObjectFactory:
    """
    Factory class for creating Mujoco objects from ObjectConfig.
    
    This class is responsible for converting high-level configuration objects into 
    low-level Mujoco object instances, handling all necessary parameter mappings and conversions.
    """
    
    @staticmethod
    def create_from_config(config: Union[ObjectConfig, MeshConfig, CuboidConfig], 
                          obj_type: str = "all", 
                          duplicate_collision_geoms: bool = True) -> Any:
        """
        Create a Mujoco object instance from ObjectConfig.
        
        Args:
            config: Object configuration
            obj_type: Geometry element type, can be 'collision', 'visual', or 'all'
            duplicate_collision_geoms: Whether to create visual geometry duplicates for each collision geometry
            
        Returns:
            Any: The created object instance (MujocoXMLObject, BoxObject, etc.)
        """
        # Check for CuboidConfig
        if hasattr(config, "type") and config.type == "cuboid":
            return MujocoObjectFactory._create_box_object(config, obj_type, duplicate_collision_geoms)
        # Check for XML-based objects (MeshConfig with mjcf_path, or other XML-based configs)
        elif hasattr(config, "mjcf_path") and config.mjcf_path:
            return MujocoObjectFactory._create_xml_object(config, obj_type, duplicate_collision_geoms)
        else:
            # No valid configuration matches
            raise ValueError(f"Unable to create object from config: {config}. No suitable factory method found.")
    
    @staticmethod
    def _create_xml_object(config: Union[ObjectConfig, MeshConfig],
                          obj_type: str = "all",
                          duplicate_collision_geoms: bool = True) -> MujocoXMLObject:
        """
        Create a MujocoXMLObject instance from XML-based configuration.
        
        Args:
            config: Object configuration with mjcf_path
            obj_type: Geometry element type
            duplicate_collision_geoms: Whether to create visual geometry duplicates
            
        Returns:
            MujocoXMLObject: The created XML-based object instance
        """
        # Determine object name
        name = config.name if hasattr(config, "name") and config.name else config.prim_path.split("/")[-1]
        
        # Set joints configuration based on interaction_type
        joints = MujocoObjectFactory._get_joints_from_interaction_type(config)
        
        # Create MujocoXMLObject instance
        obj = MujocoXMLObject(
            fname=config.mjcf_path,
            name=name,
            joints=joints,
            obj_type=obj_type,
            duplicate_collision_geoms=duplicate_collision_geoms,
            scale=config.scale if hasattr(config, "scale") else None
        )
        
        # Set object position and orientation
        MujocoObjectFactory._set_object_pose(obj, config)
            
        return obj
    
    @staticmethod
    def _create_box_object(config: CuboidConfig,
                          obj_type: str = "all",
                          duplicate_collision_geoms: bool = True) -> BoxObject:
        """
        Create a BoxObject instance from CuboidConfig.
        
        Args:
            config: Box configuration
            obj_type: Geometry element type
            duplicate_collision_geoms: Whether to create visual geometry duplicates
            
        Returns:
            BoxObject: The created box object instance
        """
        # Determine object name
        name = config.name if hasattr(config, "name") and config.name else config.prim_path.split("/")[-1]
        
        # Set joints configuration based on interaction_type
        joints = MujocoObjectFactory._get_joints_from_interaction_type(config)
        
        # Map color to rgba (adding alpha channel)
        rgba = None
        if hasattr(config, "color") and config.color is not None:
            rgba = list(config.color) + [1.0]  # Add alpha=1.0
        
        # Create BoxObject instance
        obj = BoxObject(
            name=name,
            size=config.scale if hasattr(config, "scale") else [0.05, 0.05, 0.05],  # Default size if not provided
            rgba=rgba,
            joints=joints,
            obj_type=obj_type,
            duplicate_collision_geoms=duplicate_collision_geoms
        )
        
        # Set object position and orientation
        MujocoObjectFactory._set_object_pose(obj, config)
        
        return obj
    
    @staticmethod
    def _get_joints_from_interaction_type(config):
        """
        Determine joints configuration based on interaction_type.
        
        Args:
            config: Object configuration containing interaction_type
            
        Returns:
            str or None: Joints configuration ("default" for movable, None for fixed)
        """
        # Check if config has joints attribute already set
        if hasattr(config, "joints"):
            return getattr(config, "joints")
        
        # TODO@Chenyu Cao: implement more iteraction type
        # Set joints configuration based on interaction_type
        if hasattr(config, 'interaction_type'):
            if config.interaction_type.value == "dynamic":
                # Dynamic objects can move freely, use default free joint
                return "default"
            elif config.interaction_type.value == "static":
                # Static objects are fixed in space, no joints
                return None
            elif config.interaction_type.value == "kinematic":
                # Kinematic objects are controlled externally, use default for now
                return "default"
            elif config.interaction_type.value == "ghost":
                # Ghost objects are visible but no collision, use default joint
                return "default"
            elif config.interaction_type.value == "invisible":
                # Invisible objects, use default joint for physics consistency
                return "default"
            else:
                # Fallback to default for unknown interaction types
                return "default"
        else:
            # If no interaction_type specified, use default (backward compatibility)
            return "default"
    
    @staticmethod
    def _set_object_pose(obj, config):
        """
        Set position and orientation for an object based on config.
        
        Args:
            obj: Object instance to modify
            config: Configuration containing position and orientation
        """
        # TODO@Chenyu: implement `translation` and `rotation` case
        # Set object position
        if hasattr(config, "position") and config.position is not None and len(config.position) == 3:
            obj.set_pos(config.position)
        # elif hasattr(config, "translation") and config.translation is not None and len(config.translation) == 3:
        #     obj.set_pos(config.translation)
            
        # Set object orientation
        if hasattr(config, "orientation") and config.orientation is not None and len(config.orientation) == 4:
            # Convert quaternion from xyzw to wxyz format (MuJoCo XML expects wxyz)
            from auro_utils import xyzw_to_wxyz
            quat_wxyz = xyzw_to_wxyz(config.orientation)
            obj.set_quat(quat_wxyz)
        # elif hasattr(config, "rotation") and config.rotation is not None and len(config.rotation) == 4:
        #     # Convert quaternion from xyzw to wxyz format (MuJoCo XML expects wxyz)
        #     from auro_utils import xyzw_to_wxyz
        #     quat_wxyz = xyzw_to_wxyz(config.rotation)
        #     obj.set_quat(quat_wxyz)
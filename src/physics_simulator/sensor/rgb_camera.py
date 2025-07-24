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
# Description: RGB camera for SynthNova Physics Simulator
# Author: Chenyu Cao, Herman Ye@Galbot
# Date: 2025-03-06
#
#####################################################################################

from typing import TYPE_CHECKING

from physics_simulator import PhysicsSimulator as GalbotEduSim
from synthnova_config import RgbCameraConfig
import numpy as np

from auro_utils.math.transform import wxyz_to_xyzw
from auro_utils.math.transform import xyzw_to_wxyz
from auro_utils.math.transform import position_and_orientation_to_pose
from auro_utils.math.transform import pose_to_position_and_orientation
import auro_utils.manager as au
from auro_utils import Logger
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

# Coordinate transformation matrices

# Transformation from ROS camera frame to MuJoCo camera frame
ROS_TO_MUJOCO_TRANSFORM = np.array([
    [1, 0, 0, 0], 
    [0, -1, 0, 0], 
    [0, 0, -1, 0], 
    [0, 0, 0, 1]
])

# Transformation from MuJoCo camera frame to ROS camera frame
MUJOCO_TO_ROS_TRANSFORM = np.array([
    [1, 0, 0, 0], 
    [0, -1, 0, 0], 
    [0, 0, -1, 0], 
    [0, 0, 0, 1]
])

# Transformation from World frame to MuJoCo camera frame
WORLD_TO_MUJOCO_TRANSFORM = np.array([
    [0, 0, -1, 0], 
    [-1, 0, 0, 0], 
    [0, 1, 0, 0], 
    [0, 0, 0, 1]
])

# Transformation from MuJoCo camera frame to World frame
MUJOCO_TO_WORLD_TRANSFORM = np.array([
    [0, -1, 0, 0], 
    [0, 0, 1, 0], 
    [-1, 0, 0, 0], 
    [0, 0, 0, 1]
])

class MujocoRgbCamera:
    """RGB camera sensor for MuJoCo simulation.
    
    This class provides functionality for rendering and retrieving RGB camera data 
    from a MuJoCo simulation. It handles camera positioning, orientation, intrinsic 
    parameters, and coordinate transformations between different conventions.
    """

    def __init__(self, simulator: GalbotEduSim, camera_config: RgbCameraConfig):
        """Initialize an RGB camera in the MuJoCo simulation.
        
        Args:
            simulator: The physics simulator instance
            camera_config: Configuration object for the RGB camera
            
        Raises:
            ValueError: If the camera pose configuration is invalid
        """
        self.simulator = simulator
        self.logger = self.simulator.logger
        # Get basic configs
        self.name = camera_config.name
        self.prim_path = camera_config.prim_path
        self.parent_entity_name = camera_config.parent_entity_name
        self.camera_axes = camera_config.camera_axes

        # Get pose
        self.position = camera_config.position
        self.orientation = camera_config.orientation
        self.translation = camera_config.translation
        self.rotation = camera_config.rotation
        # Input check for pose
        use_local_pose = (
            self.translation is not None
            and self.rotation is not None
            and self.position is None
            and self.orientation is None
        )
        use_global_pose = (
            self.position is not None
            and self.orientation is not None
            and self.translation is None
            and self.rotation is None
        )

        if not (use_local_pose or use_global_pose):
            raise ValueError(
                "Invalid pose input, must be either local(translation, rotation) or global(position, orientation)"
            )

        # Convert orientation to Mujoco format (wxyz)
        if self.orientation is not None:
            self.orientation = xyzw_to_wxyz(self.orientation)
        elif self.rotation is not None:
            self.rotation = xyzw_to_wxyz(self.rotation)
        else:
            self.logger.log_error("Invalid pose input")
            raise ValueError(
                "Invalid pose input, must be either local(translation, rotation) or global(position, orientation)"
            )
        # Convert camera axes to MuJoCo format
        transform_matrix = {
            "world": WORLD_TO_MUJOCO_TRANSFORM,
            "ros": ROS_TO_MUJOCO_TRANSFORM
        }.get(self.camera_axes)
        
        if transform_matrix is None:
            raise ValueError(f"Unsupported camera_axes: {self.camera_axes}")
            
        # Apply coordinate transformation
        for attr in ["orientation", "rotation"]:
            value = getattr(self, attr)
            if value is not None:
                rotation_matrix = self._quaternion_to_rotation_matrix(value)
                setattr(self, attr, self._apply_coordinate_transform(rotation_matrix, transform_matrix))

        # Get sensor config
        sensor_config = camera_config.sensor_config
        # Set camera parameters
        self.width = sensor_config.width
        self.height = sensor_config.height
        self.frequency = sensor_config.frequency
        self.fx = sensor_config.fx
        self.fy = sensor_config.fy
        self.cx = sensor_config.cx
        self.cy = sensor_config.cy
        self.pixel_size = sensor_config.pixel_size
        self.f_stop = sensor_config.f_stop
        self.focus_distance = sensor_config.focus_distance
        self.projection_type = sensor_config.projection_type
        self.clipping_range = sensor_config.clipping_range
        self.distortion_model = sensor_config.distortion_model
        self.distortion_coefficients = sensor_config.distortion_coefficients
        self.horizontal_fov = sensor_config.horizontal_fov
        self.vertical_fov = sensor_config.vertical_fov
        self.diagonal_fov = sensor_config.diagonal_fov

        # Calculate the HFOV, VFOV, and DFOV in degrees for pinhole projection
        # Warning: For fisheye projection, please use the FOV data provided by the manufacturer
        if self.horizontal_fov is None and self.projection_type == "pinhole":
            self.horizontal_fov = 2 * math.degrees(
                math.atan(self.width / (2 * self.fx))
            )
        if self.vertical_fov is None and self.projection_type == "pinhole":
            self.vertical_fov = 2 * math.degrees(math.atan(self.height / (2 * self.fy)))
        if self.diagonal_fov is None and self.projection_type == "pinhole":
            self.diagonal_fov = 2 * math.degrees(
                math.atan(
                    math.sqrt(self.width**2 + self.height**2)
                    / (2 * math.sqrt(self.fx**2 + self.fy**2))
                )
            )

        # Calculate the intrinsic matrix
        self.intrinsic_matrix = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
        )

        # The aperture size in mm
        self.horizontal_aperture = self.pixel_size * self.width
        self.vertical_aperture = self.pixel_size * self.height
        self.focal_length_x = self.fx * self.pixel_size
        self.focal_length_y = self.fy * self.pixel_size
        # The focal length in mm
        self.focal_length = (self.focal_length_x + self.focal_length_y) / 2

        # Create sensor model
        from physics_simulator.sensor.sensor_model import MujocoSensorModel

        self.sensor_model = MujocoSensorModel(
            name=self.name,
            camera_name=self.name,
            parent_entity_name=self.parent_entity_name,
            position=self.position if use_global_pose else None,
            orientation=self.orientation if use_global_pose else None,
            translation=self.translation if use_local_pose else None,
            rotation=self.rotation if use_local_pose else None,
            fov=self.horizontal_fov,  # Use horizontal FOV as default
            width=self.width,
            height=self.height,
        )

        # Mount sensor to parent in the world
        self.sensor_model.mount_to_parent(self.simulator.world.worldbody)

        # Store sensor reference in simulator
        self.simulator._sensors[self.prim_path] = self.sensor_model

        self.params = {
            "position": self.position,
            "orientation": self.orientation,
            "translation": self.translation,
            "rotation": self.rotation,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "horizontal_fov": self.horizontal_fov,
            "vertical_fov": self.vertical_fov,
            "diagonal_fov": self.diagonal_fov,
            "intrinsic_matrix": self.intrinsic_matrix,
            "projection_type": self.projection_type,
            "distortion_model": self.distortion_model,
            "distortion_coefficients": self.distortion_coefficients,
            "clipping_range": self.clipping_range,
            "focal_length": self.focal_length,
            "focus_distance": self.focus_distance,
            "name": self.name,
            "prim_path": self.prim_path,
            "frequency": self.frequency,
        }

    def rotation_matrix_to_quaternion(self, R):
        """Convert a 3x3 rotation matrix to a quaternion using scipy.
        
        Args:
            R: 3x3 rotation matrix (numpy.ndarray)
            
        Returns:
            np.ndarray: Quaternion in [w, x, y, z] format
        """
        from scipy.spatial.transform import Rotation as SciRot
        rot = SciRot.from_matrix(R)
        # scipy返回的四元数格式为[x, y, z, w]，需转换为[w, x, y, z]
        quat_xyzw = rot.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return quat_wxyz


    def _quaternion_to_rotation_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix using scipy."""
        from scipy.spatial.transform import Rotation as R
        # scipy的from_quat格式为[x, y, z, w]
        q = np.asarray(quaternion)
        if q.shape[0] != 4:
            raise ValueError("Quaternion must be of shape (4,), format [w, x, y, z]")
        # 转换为[x, y, z, w]格式
        quat_xyzw = np.array([q[1], q[2], q[3], q[0]])
        rot = R.from_quat(quat_xyzw)
        return rot.as_matrix()

    def _apply_coordinate_transform(self, rotation_matrix: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Apply coordinate system transformation and return quaternion [w, x, y, z]."""
        # transformed = transform[:3, :3] @ rotation_matrix @ transform[:3, :3].T
        transformed = np.matmul(rotation_matrix, transform[:3, :3])
        return self.rotation_matrix_to_quaternion(transformed)

    def get_parameters(self) -> dict:
        """Retrieves the camera's parameters.

        This method returns a dictionary containing all the camera's parameters,
        including intrinsic parameters, distortion coefficients, and other configuration
        settings.

        Returns:
            dict: A dictionary containing all camera parameters.
        """
        return self.params

    def render(self, depth=False, segmentation=False):
        """Render an image from the camera's current view.
        
        Args:
            depth: Whether to include depth information
            segmentation: Whether to include segmentation information
            
        Returns:
            np.ndarray: The rendered image (and depth/segmentation if requested)
        """
        return self.simulator.render(
            width=self.width,
            height=self.height,
            camera_name=self.name,
            depth=depth,
            segmentation=segmentation,
        )

    def get_data(self) -> np.ndarray:
        """Get the latest rendered RGB image data.
        
        Returns:
            np.ndarray: RGB image as a numpy array with shape (height, width, 3)
        """
        return self.get_rgb()

    def get_rgb(self):
        """Get the latest RGB image data (alias for get_data).
        
        Returns:
            np.ndarray: RGB image as a numpy array
        """
        rgb = self.render(depth=False, segmentation=False)
        data = rgb
        # Check if data is valid
        if data is None or data.size == 0:
            return None
        

        try:
            # Reshape the data to height x width x 3 format
            height = self.height
            width = self.width
            reshaped_data = data.reshape(height, width, -1)[..., :3]

            # Additional validation
            if reshaped_data.size == 0:
                return None
            
            # Flip rgb img vertically
            # Reason: the image returned by `read_pixels` is upside-down compared to the usual image coordinates (origin at top-left).
            reshaped_data = np.flipud(reshaped_data)

            return reshaped_data

        except (ValueError, AttributeError) as e:
            self.logger.log_error(f"Error processing camera data: {e}")
            return None

    def get_segmentation(self):
        """Get the segmentation mask for the current view.
        
        Returns:
            np.ndarray: Segmentation mask as a numpy array
        """
        segmentation = self.render(depth=False, segmentation=True)
        if segmentation is None:
            self.logger.log_warning(
                f"[get_segmentation][{self.prim_path}] WARNING: Annotator 'instance_segmentation' contains no data. Returning None"
            )
            return None

        return segmentation

    def get_world_pose(self, camera_axes="world"):
        """Get the camera's pose in world coordinates.
        
        Args:
            camera_axes: Coordinate system for the returned pose ("world" or "camera")
            
        Returns:
            tuple: (position, orientation) where position is a 3D vector and
                  orientation is a quaternion in [w, x, y, z] format
        """
        from physics_simulator.utils.camera_utils import get_camera_transform_matrix
        from scipy.spatial.transform import Rotation as R

        with self.simulator.lock:
            world_to_camera_matrix = get_camera_transform_matrix(
                sim=self.simulator,
                camera_name=self.name,
                camera_height=self.height,
                camera_width=self.width,
            )
            camera_to_world_matrix = np.linalg.inv(world_to_camera_matrix)
            # Extract position from the last column of the matrix
            position = camera_to_world_matrix[:3, 3]
            # Extract rotation matrix and convert to quaternion
            rotation_matrix = camera_to_world_matrix[:3, :3]
            quaternion = R.from_matrix(rotation_matrix).as_quat()

            inverse_transform = {
                "world": MUJOCO_TO_WORLD_TRANSFORM,
                "ros": MUJOCO_TO_ROS_TRANSFORM
            }.get(camera_axes, np.eye(4))

            # Transform position
            homog_pos = np.append(position, 1)
            position = (inverse_transform @ homog_pos)[:3]

            # Transform rotation
            rotation_matrix = R.from_quat(quaternion).as_matrix()
            transformed_rot = rotation_matrix @ inverse_transform[:3, :3].T
            quaternion = R.from_matrix(transformed_rot).as_quat()

        return np.concatenate([position, quaternion])

    def get_pose_wrt_robot(self, robot):
        """Get the camera's pose relative to a specific robot.
        
        Args:
            robot: The robot to calculate the relative pose to
            
        Returns:
            tuple: (position, orientation) relative to the robot's frame
        """
        with self.simulator.lock:
            camera_pose_wrt_world = self.get_world_pose(camera_axes="world")

            robot_position, robot_orientation = robot.get_world_pose()
            robot_pose_wrt_world = position_and_orientation_to_pose(
                robot_position, robot_orientation
            )
        # Extract camera's position and orientation (as a quaternion)
        camera_position = camera_pose_wrt_world[:3]
        camera_orientation = camera_pose_wrt_world[3:]

        # Create transformation matrices
        robot_rotation_matrix = R.from_quat(robot_orientation).as_matrix()
        camera_rotation_matrix = R.from_quat(camera_orientation).as_matrix()

        # Create the transformation matrix for the robot in world frame
        robot_transform = np.eye(4)
        robot_transform[:3, :3] = robot_rotation_matrix
        robot_transform[:3, 3] = robot_position

        # Create the transformation matrix for the camera in world frame
        camera_transform = np.eye(4)
        camera_transform[:3, :3] = camera_rotation_matrix
        camera_transform[:3, 3] = camera_position

        # Calculate the inverse of the robot transform to get world-to-robot transform
        world_to_robot_transform = np.linalg.inv(robot_transform)

        # Calculate the camera pose relative to the robot
        camera_pose2robot_transform = np.dot(world_to_robot_transform, camera_transform)

        # Extract the position and rotation from the resulting transformation matrix
        camera_position2robot = camera_pose2robot_transform[:3, 3]
        camera_rotation2robot = R.from_matrix(
            camera_pose2robot_transform[:3, :3]
        ).as_quat()

        # Combine position and quaternion into a single array
        camera_pose2robot = np.concatenate(
            [camera_position2robot, camera_rotation2robot]
        )
        return camera_pose2robot

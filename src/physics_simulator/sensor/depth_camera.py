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
# Description: Depth camera for SynthNova Physics Simulator
# Author: Chenyu Cao, Herman Ye@Galbot
# Date: 2025-03-06
#
#####################################################################################

import numpy as np
from .rgb_camera import MujocoRgbCamera
from physics_simulator.simulator import MujocoSimulator as PhysicsSimulator
from synthnova_config import DepthCameraConfig

class MujocoDepthCamera(MujocoRgbCamera):
    """Depth camera sensor for MuJoCo simulation.
    
    This class extends the RGB camera to provide depth sensing capabilities.
    It inherits all functionality from MujocoRgbCamera and adds depth-specific
    features such as depth map retrieval and point cloud generation.
    """
    
    def __init__(self, simulator: PhysicsSimulator, camera_config: DepthCameraConfig):
        """Initialize a depth camera in the MuJoCo simulation.
        
        Args:
            simulator: The physics simulator instance
            camera_config: Configuration object for the depth camera
        """
        super().__init__(simulator, camera_config)

    def get_data(self) -> np.ndarray:
        """Get the latest rendered depth data.
        
        Returns:
            np.ndarray: Depth image as a numpy array with shape (height, width, 1)
                        where each value represents the distance from the camera
        """
        return self.get_depth()

    def get_depth(self):
        """Get the depth map from the current camera view.
        
        Returns:
            np.ndarray: Depth map as a numpy array, with values representing
                       distance from the camera in meters
        """
        # Get depth map from MuJoCo
        _, depth = self.render(depth=True, segmentation=False)
        
        # Convert MuJoCo depth to real depth
        from physics_simulator.utils.camera_utils import get_real_depth_map
        depth = get_real_depth_map(self.simulator, depth)
        
        # Flip vertically to match standard image coordinates
        depth = np.flipud(depth)
            
        return depth
    
    def _calculate_intrinsics_from_fovy(self):
        """Calculate camera intrinsics from MuJoCo fovy parameter.
        
        MuJoCo cameras only have fovy (field of view in y direction).
        We calculate fx, fy, cx, cy from fovy and image dimensions.
        """
        # Get camera ID and fovy from MuJoCo
        cam_id = self.simulator.model.camera_name2id(self.name)
        fovy_rad = self.simulator.model.cam_fovy[cam_id] * np.pi / 180.0  # Convert to radians
        
        # Calculate focal length from fovy
        # f = height / (2 * tan(fovy/2))
        focal_length = self.height / (2 * np.tan(fovy_rad / 2))
        
        # For square pixels, fx = fy = focal_length
        fx = fy = focal_length
        
        # Principal point at image center
        cx = self.width / 2
        cy = self.height / 2
        
        return fx, fy, cx, cy
    
    def get_point_cloud(self) -> np.ndarray:
        """Generate a 3D point cloud from the depth data in world coordinates.
        
        Converts the depth map to a set of 3D points in the world coordinate frame.
        
        Returns:
            np.ndarray: Point cloud as an Nx3 array of (X, Y, Z) points in world frame
        """
        from physics_simulator.utils.camera_utils import (
            get_point_cloud_from_depth, 
            filter_point_cloud,
            get_camera_extrinsic_matrix
        )
        
        # Get depth map
        depth_map = self.get_depth()
        
        # Calculate intrinsics from MuJoCo fovy
        fx, fy, cx, cy = self._calculate_intrinsics_from_fovy()
        
        # Generate point cloud in camera frame using calculated intrinsic parameters
        point_cloud_camera = get_point_cloud_from_depth(
            depth_image=depth_map,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )
        
        # Filter the point cloud to remove invalid points
        point_cloud_camera = filter_point_cloud(
            point_cloud_camera,
            min_distance=0.01,
            max_distance=10.0,
            remove_outliers=True,
            outlier_std_factor=2.0
        )
        
        if len(point_cloud_camera) == 0:
            return np.array([]).reshape(0, 3)
        
        # Transform from camera frame to world frame
        with self.simulator.lock:
            # Get camera extrinsic matrix (camera to world transform)
            camera_extrinsic = get_camera_extrinsic_matrix(
                sim=self.simulator,
                camera_name=self.name
            )
        
        # Convert points to homogeneous coordinates
        ones = np.ones((point_cloud_camera.shape[0], 1))
        point_cloud_homogeneous = np.hstack([point_cloud_camera, ones])
        
        # Transform to world coordinates
        point_cloud_world = (camera_extrinsic @ point_cloud_homogeneous.T).T
        
        return point_cloud_world[:, :3]

    def get_point_cloud_wrt_robot(self, robot, downsample_factor=2, max_distance=2.0, 
                                  skip_outlier_removal=True):
        """Transform the depth point cloud to the robot's coordinate frame.
        
        Converts the depth data to a 3D point cloud and transforms it directly from
        camera coordinates to the robot's local coordinate frame (optimized version).
        
        Args:
            robot: The robot object to transform the point cloud relative to
            downsample_factor (int): Factor to downsample the depth image for performance 
                                   (2 = half resolution, 4x speed improvement)
            max_distance (float): Maximum distance to keep points in meters (default: 2.0m)
            skip_outlier_removal (bool): Skip expensive statistical outlier removal for speed
            
        Returns:
            np.ndarray: Point cloud as an Nx3 array of (X, Y, Z) points in robot frame
        """
        from physics_simulator.utils.camera_utils import get_camera_extrinsic_matrix
        from scipy.spatial.transform import Rotation as R
        
        # Get depth map once
        _, depth_map = self.render(depth=True, segmentation=False)
        from physics_simulator.utils.camera_utils import get_real_depth_map
        depth_map = get_real_depth_map(self.simulator, depth_map)
        
        # Flip vertically to match standard image coordinates (same as get_depth method)
        depth_map = np.flipud(depth_map)
        
        # Calculate intrinsics from MuJoCo fovy first
        fx, fy, cx, cy = self._calculate_intrinsics_from_fovy()
        
        # Downsample for performance if requested
        if downsample_factor > 1:
            depth_map = depth_map[::downsample_factor, ::downsample_factor]
            fx_scaled = fx / downsample_factor
            fy_scaled = fy / downsample_factor
            cx_scaled = cx / downsample_factor
            cy_scaled = cy / downsample_factor
        else:
            fx_scaled, fy_scaled, cx_scaled, cy_scaled = fx, fy, cx, cy
        
        # Generate point cloud in camera frame (optimized)
        height, width = depth_map.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Simple distance filtering instead of complex outlier removal for speed
        valid_mask = (depth_map > 0.01) & (depth_map < max_distance) & np.isfinite(depth_map)
        
        if not np.any(valid_mask):
            return np.array([]).reshape(0, 3)
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth_map[valid_mask]
        
        # Convert to camera coordinates using pinhole camera model
        x_camera = (u_valid - cx_scaled) * z_valid / fx_scaled
        y_camera = (v_valid - cy_scaled) * z_valid / fy_scaled
        z_camera = z_valid
        
        point_cloud_camera = np.stack([x_camera, y_camera, z_camera], axis=1)
        
        # Apply statistical outlier removal if not skipped (slower but better quality)
        if not skip_outlier_removal:
            from physics_simulator.utils.camera_utils import filter_point_cloud
            point_cloud_camera = filter_point_cloud(
                point_cloud_camera,
                min_distance=0.01,
                max_distance=max_distance,
                remove_outliers=True,
                outlier_std_factor=2.0
            )
            
            if len(point_cloud_camera) == 0:
                return np.array([]).reshape(0, 3)
        
        # Get transformation matrices in one lock for efficiency
        with self.simulator.lock:
            # Get camera extrinsic matrix (camera to world transform)
            camera_extrinsic = get_camera_extrinsic_matrix(
                sim=self.simulator,
                camera_name=self.name
            )
            
            # Get robot pose in world frame
            robot_position, robot_orientation = robot.get_world_pose()
        
        # Compute direct camera-to-robot transformation (skip intermediate world transform)
        robot_rotation_matrix = R.from_quat(robot_orientation).as_matrix()
        world_to_robot = np.eye(4)
        world_to_robot[:3, :3] = robot_rotation_matrix.T
        world_to_robot[:3, 3] = -robot_rotation_matrix.T @ robot_position
        
        # Direct camera-to-robot transformation matrix
        camera_to_robot = world_to_robot @ camera_extrinsic
        
        # Transform points directly from camera to robot frame
        ones = np.ones((len(point_cloud_camera), 1))
        point_cloud_homogeneous = np.hstack([point_cloud_camera, ones])
        point_cloud_robot = (camera_to_robot @ point_cloud_homogeneous.T).T
        
        return point_cloud_robot[:, :3]
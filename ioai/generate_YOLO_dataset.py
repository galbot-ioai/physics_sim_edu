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
# Description: YOLO Dataset Generation for Objects
# Author: Galbot
# Date: 2025-07-30
#
#####################################################################################

from physics_simulator import PhysicsSimulator
from synthnova_config import (
    MujocoConfig,
    PhysicsSimulatorConfig,
    RobotConfig,
    MeshConfig,
    CuboidConfig,
    RgbCameraConfig,
    RealsenseD436RgbSensorConfig,
    DepthCameraConfig,
    RealsenseD436DepthSensorConfig
)
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
from physics_simulator.utils import preprocess_depth
import os
import numpy as np
import cv2

from pathlib import Path

from PIL import Image
from auro_utils import xyzw_to_wxyz
from scipy.spatial.transform import Rotation as R
import random
import argparse

# Object classes for YOLO dataset
CLASSES = ["power_drill", "cube", "mug", "bin", "extrusion", "toy"]
SPLITS = {"train": 21, "test": 6, "val": 3}
ROOT = "/home/galbot/project/physics_sim_edu/dataset"

def make_dirs(object_name=None):
    """Create dataset directory structure and clean existing files"""
    for split in SPLITS:
        for sub in ["images", "labels"]:
            dir_path = f"{ROOT}/{sub}/{split}"
            os.makedirs(dir_path, exist_ok=True)
            if object_name:
                # Only delete files related to the specified object
                for f in os.listdir(dir_path):
                    if f.startswith(f"{object_name}_"):
                        os.remove(os.path.join(dir_path, f))
            else:
                # Clear entire directory for full dataset collection
                for f in os.listdir(dir_path):
                    os.remove(os.path.join(dir_path, f))

def write_classes():
    """Write class names to classes.txt file"""
    with open(f"{ROOT}/classes.txt", "w") as f:
        for cls in CLASSES:
            f.write(f"{cls}\n")

def random_pose(x_range, y_range, z_value, object_name=None):
    """Generate random pose within specified ranges"""
    if object_name == "bin":
        # Bin: restrict to bottom-left corner with small movement range
        table_x, table_y, table_z = [0.65, 0.0, 0.0]
        table_width, table_length = [0.55, 0.90]
        
        # Bottom-left corner base position (with small margin from edge)
        base_x = table_x - table_width/2 + 0.3  # 18cm from left edge
        base_y = table_y + table_length/2 - 0.2  # 10cm from bottom edge
        
        # Small random movement within 5cm range
        x = base_x + random.uniform(-0.05, 0.05)
        y = base_y + random.uniform(-0.05, 0.05)
        z = z_value
        
        # Initial rotation 90 degrees + small random rotation (within 10 degrees)
        base_angle_deg = 0  # Initial 90 degree rotation
        random_angle_deg = random.uniform(-10, 10)  # Small random variation
        total_angle_deg = base_angle_deg + random_angle_deg
        angle_rad = np.radians(total_angle_deg)
        quat = R.from_euler('z', angle_rad).as_quat()  # [x, y, z, w]
    else:
        # Other objects: normal random positioning
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        z = z_value
        quat = R.random().as_quat()  # [x, y, z, w]
    
    position = [x, y, z]
    return position, quat

def check_collision(pos1, pos2, radius1, radius2, min_distance=0.05):
    """Check if two objects collide based on their positions and radii"""
    center_distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    edge_distance = center_distance - radius1 - radius2
    return edge_distance < min_distance

def get_object_radius(object_name):
    """Get radius for collision detection"""
    if object_name == "bin":
        return 0.17  # Actual: 0.169m, +1% safety margin
    elif object_name == "cube":
        return 0.03  # Estimated for cube
    elif object_name == "mug":
        return 0.06  # Actual: 0.059m, +2% safety margin
    elif object_name == "power_drill":
        return 0.07  # Actual: 0.069m, +1% safety margin
    elif object_name == "extrusion":
        return 0.05  # Actual: 0.050m, +0% safety margin
    elif object_name == "toy":
        return 0.05  # Actual: 0.048m, +4% safety margin
    else:
        return 0.06  # Default radius for other objects

def generate_non_colliding_positions(object_list, x_range, y_range, z_value, max_attempts=50):
    """Generate positions for multiple objects without collision"""
    positions = []
    orientations = []
    
    for i, object_name in enumerate(object_list):
        radius = get_object_radius(object_name)
        attempts = 0
        position = None
        
        while attempts < max_attempts:
            # Generate random position
            pos, quat = random_pose(x_range, y_range, z_value, object_name)
            
            # Check collision with existing objects
            collision = False
            for j, existing_pos in enumerate(positions):
                existing_radius = get_object_radius(object_list[j])
                if check_collision(pos, existing_pos, radius, existing_radius):
                    collision = True
                    break
            
            if not collision:
                position = pos
                break
            
            attempts += 1
        
        if position is None:
            # If can't find non-colliding position, use random position
            position, quat = random_pose(x_range, y_range, z_value, object_name)
        
        positions.append(position)
        orientations.append(quat)
    
    return positions, orientations

def generate_random_object_list(min_objects=2, max_objects=6):
    """Generate random list of objects without duplicates"""
    num_objects = random.randint(min_objects, min(max_objects, len(CLASSES)))
    # Randomly select objects, no duplicates allowed
    object_list = random.sample(CLASSES, num_objects)
    return object_list

def check_objects_on_table(synthnova_physics_simulator, object_geoms_dict, object_poses_dict, table_z=0.55, tolerance=0.1):
    """Check if all objects are on the table surface"""
    for object_name, geom_ids in object_geoms_dict.items():
        # Get actual position of the object
        if object_name in object_poses_dict:
            actual_position = object_poses_dict[object_name][0]
            z_pos = actual_position[2]
        
            # Table surface height is 0.55, if object z coordinate is below 0.45, consider it fell off
            table_surface_z = table_z  # Table surface height
            min_z_threshold = table_surface_z - tolerance
            
            if z_pos < min_z_threshold:
                print(f"[WARNING] Object {object_name} fell off table (z: {z_pos:.3f})")
                return False
    
    return True

def check_objects_intersection(synthnova_physics_simulator, object_geoms_dict, object_poses_dict, min_distance=0.01):
    """Check if objects intersect with each other, check edge distance"""
    object_names = list(object_poses_dict.keys())
    
    for i, object_name1 in enumerate(object_names):
        for j, object_name2 in enumerate(object_names):
            if i >= j:  # Avoid checking the same pair twice
                continue
            
            pos1 = object_poses_dict[object_name1][0]
            pos2 = object_poses_dict[object_name2][0]
            
            # Calculate distance between centers of two objects
            center_distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            
            # Get radii using the existing function to avoid code duplication
            obj_name1 = object_name1.split('_')[0]  # Remove index suffix
            obj_name2 = object_name2.split('_')[0]  # Remove index suffix
            radius1 = get_object_radius(obj_name1)
            radius2 = get_object_radius(obj_name2)
            
            # Calculate edge distance (center distance minus two radii)
            edge_distance = center_distance - radius1 - radius2
            
            # If edge distance is less than minimum distance, consider it intersection
            if edge_distance < min_distance:
                print(f"[WARNING] Intersection: {object_name1} and {object_name2} (distance: {edge_distance:.3f}m)")
                return True
    
    return False

def collect_and_save_sample_yolo_single(img_name, label_name, save_split_dir, 
                                       galbot_interface, geom_ids, object_pose, class_id):
    """Collect and save single object data for YOLO format"""
    rgb = galbot_interface.front_head_camera.get_rgb()
    mask = galbot_interface.front_head_camera.get_segmentation()
   
    # Flip mask to correspond with rgb
    mask = np.flipud(mask)

    images_dir = os.path.join(ROOT, "images", save_split_dir)
    labels_dir = os.path.join(ROOT, "labels", save_split_dir)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    image_path = os.path.join(images_dir, img_name)
    label_path = os.path.join(labels_dir, label_name)
    # Save as RGB format 
    Image.fromarray(rgb).save(image_path)
    height, width = rgb.shape[:2]

    # Calculate segmentation labels based on mask - target object is 255 white, background is 0
    obj_mask_bool = np.isin(mask[..., 1], geom_ids) 
    if np.any(obj_mask_bool):
        # Extract mask contour
        mask_uint8 = obj_mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Take the largest contour
            contour = max(contours, key=cv2.contourArea)
            # Normalize polygon points
            poly = contour.squeeze()
            if poly.ndim == 1:
                poly = poly[None, :]
            poly_norm = []
            for x, y in poly:
                poly_norm.append(f"{x/width:.6f}")
                poly_norm.append(f"{y/height:.6f}")
            with open(label_path, "w") as f:
                f.write(f"{class_id} " + " ".join(poly_norm) + "\n")
        else:
            with open(label_path, "w") as f:
                pass
    else:
        with open(label_path, "w") as f:
            pass

def collect_and_save_sample_yolo_multi(img_name, label_name, save_split_dir, 
                                      galbot_interface, object_geoms_dict, object_poses_dict):
    """Collect and save multiple objects data for YOLO format"""
    rgb = galbot_interface.front_head_camera.get_rgb()
    mask = galbot_interface.front_head_camera.get_segmentation()

    # Flip mask to correspond with rgb
    mask = np.flipud(mask)
    
    images_dir = os.path.join(ROOT, "images", save_split_dir)
    labels_dir = os.path.join(ROOT, "labels", save_split_dir)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    image_path = os.path.join(images_dir, img_name)
    label_path = os.path.join(labels_dir, label_name)
    
    # Save as RGB format 
    Image.fromarray(rgb).save(image_path)
    height, width = rgb.shape[:2]
    
    # Generate labels for each object
    with open(label_path, "w") as f:
        for object_key, geom_ids in object_geoms_dict.items():
            # Extract object name from key (remove _idx suffix)
            # Handle cases where object name contains underscores (e.g., power_drill)
            parts = object_key.split('_')
            if len(parts) >= 2:
                # Try to find the object name by checking if the first part(s) match a class
                for i in range(len(parts) - 1, 0, -1):
                    potential_name = '_'.join(parts[:i])
                    if potential_name in CLASSES:
                        object_name = potential_name
                        break
                else:
                    # If no match found, use the first part (fallback)
                    object_name = parts[0]
            else:
                object_name = parts[0]
            class_id = CLASSES.index(object_name)
            
            # Calculate segmentation labels based on mask - target object is 255 white, background is 0
            obj_mask_bool = np.isin(mask[..., 1], geom_ids) 
            if np.any(obj_mask_bool):
                # Extract mask contour
                mask_uint8 = obj_mask_bool.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                    # Take the largest contour
                    contour = max(contours, key=cv2.contourArea)
                    # Normalize polygon points
                    poly = contour.squeeze()
                    if poly.ndim == 1:
                        poly = poly[None, :]
                    poly_norm = []
                    for x, y in poly:
                        poly_norm.append(f"{x/width:.6f}")
                        poly_norm.append(f"{y/height:.6f}")
                    f.write(f"{class_id} " + " ".join(poly_norm) + "\n")

def setup_simulator():
    """Setup simulator and robot configuration"""
    # Initialize simulator
    sim_config = PhysicsSimulatorConfig(
        mujoco_config=MujocoConfig(headless=False)
    )
    synthnova_physics_simulator = PhysicsSimulator(sim_config)
    synthnova_physics_simulator.add_default_scene()
    
    # # Add robot
    # robot_config = RobotConfig(
    #     prim_path="/World/Galbot",
    #     name="galbot_one_foxtrot",
    #     mjcf_path=Path()
    #         .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
    #         .joinpath("synthnova_assets")
    #         .joinpath("robots")
    #         .joinpath("galbot_one_foxtrot_description_simplified")
    #         .joinpath("galbot_one_foxtrot.xml"),
    #     position=[0, 0, 0],
    #     orientation=[0, 0, 0, 1]
    # )
    # synthnova_physics_simulator.add_robot(robot_config)
    # robot = synthnova_physics_simulator.get_robot("/World/Galbot")


    robot_config = RobotConfig(
        prim_path="/World/Galbot",
        name="galbot_one_foxtrot",
        mjcf_path=Path()
        .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
        .joinpath("synthnova_assets")
        .joinpath("robots")
        .joinpath("galbot_one_foxtrot_description_simplified")
        .joinpath("galbot_one_foxtrot.xml"),
        position=[0.65, 0.9, 0],
        orientation=[0, 0, 0.70711, -0.70711]
    )
    synthnova_physics_simulator.add_robot(robot_config)
    robot = synthnova_physics_simulator.get_robot("/World/Galbot")

    
    # Add front head RGB camera (RealSense D436)
    front_head_rgb_camera_config = RgbCameraConfig(
        name="front_head_rgb_camera",
        prim_path=os.path.join(
            robot.prim_path,
            "head_link2",
            "head_end_effector_mount_link",
            "front_head_rgb_camera",
        ),
        translation=[
            0.10084319533055261,
            -0.059042081352783105,
            0.03184978861787491
        ],
        rotation=[
            -0.1654571792421115, 
            0.6935589352367344,
            0.16457378953789606,
            0.6815536611211676
        ],
        camera_axes="ros",
        sensor_config=RealsenseD436RgbSensorConfig(),
        parent_entity_name="galbot_one_foxtrot/head_end_effector_mount_link"
    )
    front_head_rgb_camera_path = synthnova_physics_simulator.add_sensor(front_head_rgb_camera_config)

    # Add front head depth camera (RealSense D436)
    front_head_depth_camera_config = DepthCameraConfig(
        name="front_head_depth_camera",
        prim_path=os.path.join(
            robot.prim_path,
            "head_link2",
            "head_end_effector_mount_link",
            "front_head_depth_camera",
        ),
        translation=[
            0.10084319533055261,
            -0.059042081352783105,
            0.03184978861787491
        ],
        rotation=[
            -0.1654571792421115, 
            0.6935589352367344,
            0.16457378953789606,
            0.6815536611211676
        ],
        camera_axes="ros",
        sensor_config=RealsenseD436DepthSensorConfig(),
        parent_entity_name="galbot_one_foxtrot/head_end_effector_mount_link"
    )
    front_head_depth_camera_path = synthnova_physics_simulator.add_sensor(front_head_depth_camera_config)
    
    # Add table
    table_config = MeshConfig(
        prim_path="/World/Table",
        name="table",
        mjcf_path=Path()
            .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("objects")
            .joinpath("table")
            .joinpath("table.xml"),
        position=[0.65, 0.0, 0.0],  
        orientation=[0, 0, 0.70711, 0.70711],  
    )
    synthnova_physics_simulator.add_object(table_config)

    center_x = 2
    center_y = 2
    wall_width = 5.5
    wall_height = 0.5
    wall_depth = 0.05

    # Add walls
    wall_1_config = CuboidConfig(
        prim_path="/World/Wall1",
        position=[center_x, center_y+wall_width/2, wall_height / 2],
        orientation=[0, 0, 0, 1],
        scale=[wall_width, wall_depth, wall_height],
        color=[0.6, 0.8, 1.0],  # Light blue color
        interaction_type="static"
    )
    synthnova_physics_simulator.add_object(wall_1_config)

    wall_2_config = CuboidConfig(
        prim_path="/World/Wall2",
        position=[center_x, center_y-wall_width/2, wall_height / 2],
        orientation=[0, 0, 0, 1],
        scale=[wall_width, wall_depth, wall_height],
        color=[0.6, 0.8, 1.0],  # Light blue color
        interaction_type="static"
    )
    synthnova_physics_simulator.add_object(wall_2_config)

    wall_3_config = CuboidConfig(
        prim_path="/World/Wall3",
        position=[center_x+wall_width/2, center_y, wall_height / 2],
        orientation=[0, 0, 0, 1],
        scale=[wall_depth, wall_width, wall_height],
        color=[0.6, 0.8, 1.0],  # Light blue color
        interaction_type="static"
    )
    synthnova_physics_simulator.add_object(wall_3_config)

    wall_4_config = CuboidConfig(
        prim_path="/World/Wall4",
        position=[center_x-wall_width/2, center_y, wall_height / 2],
        orientation=[0, 0, 0, 1],
        scale=[wall_depth, wall_width, wall_height],
        color=[0.6, 0.8, 1.0],  # Light blue color
        interaction_type="static"
    )
    synthnova_physics_simulator.add_object(wall_4_config)

    # Add shelf
    shelf_config = MeshConfig(
        prim_path="/World/Shelf",
        mjcf_path=Path()
        .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
        .joinpath("synthnova_assets")
        .joinpath("objects")
        .joinpath("shelf")
        .joinpath("shelf.xml"),
        position=[4, 4, 0.55],
        orientation=[0, 0, 0.70711, 0.70711],
    )
    synthnova_physics_simulator.add_object(shelf_config)

    # Add cones
    cone_1_config = MeshConfig(
        prim_path="/World/Cone1",
        mjcf_path=Path()
        .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
        .joinpath("synthnova_assets")
        .joinpath("objects")
        .joinpath("cone")
        .joinpath("cone.xml"),
        position=[3, 3, 0.55],
        orientation=[0, 0, 0.70711, 0.70711],
    )
    synthnova_physics_simulator.add_object(cone_1_config)

    cone_2_config = MeshConfig(
        prim_path="/World/Cone2",
        mjcf_path=Path()
        .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
        .joinpath("synthnova_assets")
        .joinpath("objects")
        .joinpath("cone")
        .joinpath("cone.xml"),
        position=[2, 2, 0.55],
        orientation=[0, 0, 0.70711, 0.70711],
    )
    synthnova_physics_simulator.add_object(cone_2_config)

    cone_3_config = MeshConfig(
        prim_path="/World/Cone3",
        mjcf_path=Path()
        .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
        .joinpath("synthnova_assets")
        .joinpath("objects")
        .joinpath("cone")
        .joinpath("cone.xml"),
        position=[1, 1, 0.55],
        orientation=[0, 0, 0.70711, 0.70711],
    )
    synthnova_physics_simulator.add_object(cone_3_config)

    return synthnova_physics_simulator, robot, front_head_rgb_camera_path, front_head_depth_camera_path

def setup_interface(synthnova_physics_simulator, robot, front_head_rgb_camera_path, front_head_depth_camera_path):
    """Setup Galbot interface configuration"""
    galbot_interface_config = GalbotInterfaceConfig()
    galbot_interface_config.robot.prim_path = "/World/Galbot"
    robot_name = robot.name
    
    # Enable modules 
    galbot_interface_config.modules_manager.enabled_modules.extend([
        "front_head_camera", "right_arm", "left_arm", "leg", "head", 
        "chassis", "left_gripper", "right_gripper"
    ])

    galbot_interface_config.right_arm.joint_names = [
        f"{robot_name}/right_arm_joint1",
        f"{robot_name}/right_arm_joint2",
        f"{robot_name}/right_arm_joint3",
        f"{robot_name}/right_arm_joint4",
        f"{robot_name}/right_arm_joint5",
        f"{robot_name}/right_arm_joint6",
        f"{robot_name}/right_arm_joint7",
    ]

    galbot_interface_config.left_arm.joint_names = [
        f"{robot_name}/left_arm_joint1",
        f"{robot_name}/left_arm_joint2",
        f"{robot_name}/left_arm_joint3",
        f"{robot_name}/left_arm_joint4",
        f"{robot_name}/left_arm_joint5",
        f"{robot_name}/left_arm_joint6",
        f"{robot_name}/left_arm_joint7",
    ]

    galbot_interface_config.leg.joint_names = [
        f"{robot_name}/leg_joint1",
        f"{robot_name}/leg_joint2",
        f"{robot_name}/leg_joint3",
        f"{robot_name}/leg_joint4",
    ]
    
    galbot_interface_config.head.joint_names = [
        f"{robot_name}/head_joint1",
        f"{robot_name}/head_joint2"
    ]

    galbot_interface_config.chassis.joint_names = [
        f"{robot_name}/mobile_forward_joint",
        f"{robot_name}/mobile_side_joint",
        f"{robot_name}/mobile_yaw_joint",
    ]

    galbot_interface_config.left_gripper.joint_names = [
        f"{robot_name}/left_gripper_r_knuckle_joint",
    ]

    galbot_interface_config.right_gripper.joint_names = [
        f"{robot_name}/right_gripper_r_knuckle_joint",
    ]

    galbot_interface_config.front_head_camera.prim_path_rgb = front_head_rgb_camera_path
    galbot_interface_config.front_head_camera.prim_path_depth = front_head_depth_camera_path
    galbot_interface = GalbotInterface(
        galbot_interface_config=galbot_interface_config,
        simulator=synthnova_physics_simulator
    )

    synthnova_physics_simulator.play()
    synthnova_physics_simulator.step(10)  # Let simulator stabilize

    galbot_interface.initialize()
    
    galbot_interface.head.set_joint_positions([0.0, 0.26], immediate=True)
    galbot_interface.leg.set_joint_positions([0.0821758285164833, 0.6340972781181335, 0.5227039456367493, -0.00001198422432935331], immediate=True)
    # galbot_interface.left_arm.set_joint_positions([2.0020599365234375, -1.5977126359939575, -0.5948255658149719, -1.694089651107788, -0.0002879792882595211, -0.7909831404685974, -0.00016755158139858395], immediate=True)
    # galbot_interface.right_arm.set_joint_positions([-2.001628875732422, 1.6029852628707886, 0.6024474501609802, 1.6955766677856445, -0.0002391100861132145, 0.7967827916145325, -0.00014311698032543063], immediate=True)
    galbot_interface.left_arm.set_joint_positions([-1.303036521691927, -0.849731362082964, -1.5498274064724777, 1.7160621326982866, -0.5459467998122908, 0.73477846982291, -1.530963497943708], immediate=True)
    galbot_interface.right_arm.set_joint_positions([0.47050871457159077, -1.4784330820163598, 0.6231806137214511, -1.873117291806456, -1.3999728445850048, 0.010039015860751377, -1.0994585195925979], immediate=True)


    # Wait for joint settings to take effect
    synthnova_physics_simulator.step(1000)
    return galbot_interface

def main():
    """Main function for YOLO dataset generation"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate YOLO segmentation dataset')
    parser.add_argument('--object', type=str, default=None, 
                       help='Specify object name to collect (power_drill, cube, mug, bin, extrusion, toy, multi)')
    parser.add_argument('--start_idx', type=int, default=1, 
                       help='Start index (default: 1)')
    parser.add_argument('--end_idx', type=int, default=31, 
                       help='End index (default: 31)')
    parser.add_argument('--split', type=str, default=None, 
                       help='Specify dataset split (train, test, val, default: auto-assign by index)')
    parser.add_argument('--min_objects', type=int, default=2, 
                       help='Minimum number of objects (default: 2, only for multi mode)')
    parser.add_argument('--max_objects', type=int, default=6, 
                       help='Maximum number of objects (default: 6, only for multi mode)')
    args = parser.parse_args()

    # Validate object name
    valid_objects = CLASSES + ["multi"]
    if args.object and args.object not in valid_objects:
        print(f"Error: Object name '{args.object}' not in supported list: {valid_objects}")
        return

    # Determine target object list
    if args.object:
        if args.object == "multi":
            target_classes = []
            print("Specified to collect multi-object data")
        else:
            target_classes = [(CLASSES.index(args.object), args.object)]
            print(f"Specified to collect object: {args.object}")
    else:
        target_classes = list(enumerate(CLASSES))
        print("Collect all single objects")

    # Determine index range
    start_idx = args.start_idx
    end_idx = args.end_idx
    print(f"Index range: {start_idx} - {end_idx}")

    # Determine dataset split
    if args.split:
        if args.split not in ["train", "test", "val"]:
            print(f"Error: Dataset split '{args.split}' invalid, should be train, test, val")
            return
        target_splits = [args.split]
        print(f"Specified dataset split: {args.split}")
    else:
        target_splits = ["train", "test", "val"]
        print("Auto-assign dataset split by index")

    # Setup directories
    if args.object:
        make_dirs(args.object)
    else:
        make_dirs()
    write_classes()

    table_position = [0.65, 0.0, 0.0] 
    table_size = [0.55, 0.90]
    table_x, table_y, table_z = table_position
    table_width, table_length = table_size

    # Set up safety margin
    safety_margin1 = 0.12
    safety_margin2 = 0.35
    x_range = (table_x - table_width/2 + safety_margin1, table_x + table_width/2 - safety_margin1)
    y_range = (table_y - table_length/2 + safety_margin1, table_y + table_length/2 - safety_margin2)
    z_value = 0.55

    # Process single object data collection
    for class_id, object_name in target_classes:
        if args.split:
            # If split is specified, use the specified range directly
            indices = list(range(start_idx, end_idx))
            split_indices = {args.split: indices}
        else:
            # Auto-assign indices to train/test/val
            all_indices = list(range(start_idx, end_idx))
            split_indices = {
                "train": [i for i in all_indices if 1 <= i < 22],
                "test": [i for i in all_indices if 22 <= i < 28],
                "val": [i for i in all_indices if 28 <= i < 31]
            }
        
        for split in target_splits:
            if split not in split_indices or not split_indices[split]:
                continue
            indices = split_indices[split]
            for i in indices:
                img_name = f"{object_name}_{i:02d}.jpg"
                label_name = f"{object_name}_{i:02d}.txt"
                
                print(f"\n{'='*50}")
                print(f"[INFO] {object_name} {split} frame {i}")
                print(f"{'='*50}")
                
                # Retry mechanism: if object falls off table, recollect
                retry_count = 0
                success = False
                
                while not success:
                    if retry_count > 0:
                        print(f"\n[RETRY] {object_name} {split} frame {i} - attempt {retry_count}")
                    
                    # Setup simulator and robot
                    synthnova_physics_simulator, robot, front_head_rgb_camera_path, front_head_depth_camera_path = setup_simulator()
                    
                    # Randomize object
                    position, quat = random_pose(x_range, y_range, z_value, object_name)
                    unique_prim_path = f"/World/{object_name[0].upper() + object_name[1:]}" if object_name != "cube" else f"/World/cube"
                    
                    if object_name == "cube":
                        cube_config = CuboidConfig(
                            prim_path=unique_prim_path,
                            position=position,
                            orientation=quat,
                            scale=[0.05, 0.05, 0.05],  
                            color=[0.5, 0.5, 0.5], 
                        )
                        synthnova_physics_simulator.add_object(cube_config)
                    else:
                        object_config = MeshConfig(
                            prim_path=unique_prim_path,
                            name=object_name,
                            mjcf_path=Path()
                                .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
                                .joinpath("synthnova_assets")
                                .joinpath("objects")
                                .joinpath(object_name)
                                .joinpath(f"{object_name}.xml"),
                            position=position,
                            orientation=quat,
                        )
                        synthnova_physics_simulator.add_object(object_config)

                    # Setup Galbot interface
                    galbot_interface = setup_interface(synthnova_physics_simulator, robot, front_head_rgb_camera_path, front_head_depth_camera_path)
                    
                    # Wait for object to stabilize after placement
                    synthnova_physics_simulator.step(1000)
                    
                    obj = synthnova_physics_simulator.get_object(unique_prim_path)
                    geom_names = obj.visual_geoms
                    geom_ids = [synthnova_physics_simulator.model.geom_name2id(name) for name in geom_names]
                    obj_state = synthnova_physics_simulator.get_object_state(unique_prim_path)
                    actual_position = obj_state['position']
                    actual_orientation = obj_state['orientation']
                    object_pose = (actual_position, actual_orientation)

                    # Check if object is on table
                    object_geoms_dict = {object_name: geom_ids}
                    object_poses_dict = {object_name: object_pose}
                    
                    if check_objects_on_table(synthnova_physics_simulator, object_geoms_dict, object_poses_dict, table_z=0.55):
                        # Object is on table, collect data
                        collect_and_save_sample_yolo_single(img_name, label_name, split, galbot_interface, geom_ids, object_pose, class_id)
                        print(f"[INFO] {object_name} {split} collected {i}/{len(indices)} frames")
                        print("\n\n")  # Add clear separation between frames
                        success = True
                    else:
                        # Object fell off table, close simulator and prepare to retry
                        print(f"[WARNING] {object_name} {split} frame {i} - object fell off table, preparing to retry")
                        synthnova_physics_simulator.close()
                        retry_count += 1
                        continue
                    
                    synthnova_physics_simulator.close()


    # Process multi-object data collection
    if args.object == "multi" or args.object is None:
        print(f"Object count range: {args.min_objects} - {args.max_objects}")
        
        if args.split:
            # If split is specified, use the specified range directly
            indices = list(range(start_idx, end_idx))
            split_indices = {args.split: indices}
        else:
            # Auto-assign indices to train/test/val
            all_indices = list(range(start_idx, end_idx))
            split_indices = {
                "train": [i for i in all_indices if 1 <= i < 22],
                "test": [i for i in all_indices if 22 <= i < 28],
                "val": [i for i in all_indices if 28 <= i < 31]
            }
        
        for split in target_splits:
            if split not in split_indices or not split_indices[split]:
                continue
            indices = split_indices[split]
            for i in indices:
                # Retry mechanism: if objects fall off table or intersect, recollect
                retry_count = 0
                success = False
                
                print(f"\n{'='*50}")
                print(f"[INFO] multi {split} frame {i}")
                print(f"{'='*50}")
                
                # Retry mechanism: if objects fall off table or intersect, recollect
                retry_count = 0
                success = False
                
                while not success:
                    if retry_count > 0:
                        print(f"\n[RETRY] multi {split} frame {i} - attempt {retry_count}")
                    
                    # Generate random object list
                    object_list = generate_random_object_list(args.min_objects, args.max_objects)
                    combination_name = f"multi_{i:02d}"
                    
                    img_name = f"{combination_name}.jpg"
                    label_name = f"{combination_name}.txt"
                    
                    print(f"[INFO] {combination_name} {split} frame {i} - random objects: {object_list}")
                    
                    # Setup simulator 
                    synthnova_physics_simulator, robot, front_head_rgb_camera_path, front_head_depth_camera_path = setup_simulator()
                    
                    # Generate non-colliding positions for all objects
                    positions, orientations = generate_non_colliding_positions(object_list, x_range, y_range, z_value)
                    
                    # Add objects with pre-generated non-colliding poses
                    for idx, object_name in enumerate(object_list):
                        unique_prim_path = f"/World/{object_name[0].upper() + object_name[1:]}_{idx}" if object_name != "cube" else f"/World/cube_{idx}"
                        
                        position = positions[idx]
                        quat = orientations[idx]
                        
                        if object_name == "cube":
                            cube_config = CuboidConfig(
                                prim_path=unique_prim_path,
                                position=position,
                                orientation=quat,
                                scale=[0.05, 0.05, 0.05],
                                color=[0.5, 0.5, 0.5],
                            )
                            synthnova_physics_simulator.add_object(cube_config)
                        else:
                            object_config = MeshConfig(
                                prim_path=unique_prim_path,
                                name=f"{object_name}_{idx}",  # Create unique name for each instance
                                mjcf_path=Path()
                                    .joinpath(synthnova_physics_simulator.synthnova_assets_directory)
                                    .joinpath("synthnova_assets")
                                    .joinpath("objects")
                                    .joinpath(object_name)
                                    .joinpath(f"{object_name}.xml"),
                                position=position,
                                orientation=quat,
                            )
                            synthnova_physics_simulator.add_object(object_config)

                    
                    # Setup Galbot interface
                    galbot_interface = setup_interface(synthnova_physics_simulator, robot, front_head_rgb_camera_path, front_head_depth_camera_path)
                    
                    # Wait for joint settings to take effect and let objects stabilize
                    synthnova_physics_simulator.step(10000)  
                    
                    # Get geometric information and state for all objects
                    object_geoms_dict = {}
                    object_poses_dict = {}
                    
                    for idx, object_name in enumerate(object_list):
                        unique_prim_path = f"/World/{object_name[0].upper() + object_name[1:]}_{idx}" if object_name != "cube" else f"/World/cube_{idx}"
                        obj = synthnova_physics_simulator.get_object(unique_prim_path)
                        geom_names = obj.visual_geoms
                        geom_ids = [synthnova_physics_simulator.model.geom_name2id(name) for name in geom_names]
                        # Use unique key to avoid conflicts when same object type appears multiple times
                        object_key = f"{object_name}_{idx}"
                        object_geoms_dict[object_key] = geom_ids
                        
                        obj_state = synthnova_physics_simulator.get_object_state(unique_prim_path)
                        actual_position = obj_state['position']
                        actual_orientation = obj_state['orientation']
                        object_poses_dict[object_key] = (actual_position, actual_orientation)
                    
                    # Check if all objects are on table
                    if check_objects_on_table(synthnova_physics_simulator, object_geoms_dict, object_poses_dict, table_z=0.55):
                        # Special check for bin in multi mode
                        bin_outside = False
                        for object_name, geom_ids in object_geoms_dict.items():
                            if 'bin' in object_name and object_name in object_poses_dict:
                                pos = object_poses_dict[object_name][0]
                                # Check if bin is within table boundaries (more lenient range)
                                if pos[0] < 0.375 or pos[0] > 0.825 or pos[1] < -0.35 or pos[1] > 0.35:
                                    print(f"[WARNING] Bin {object_name} is outside expected range (x: {pos[0]:.3f}, y: {pos[1]:.3f})")
                                    bin_outside = True
                                    break
                        
                        if bin_outside:
                            synthnova_physics_simulator.close()
                            retry_count += 1
                            continue
                        
                        # Check if objects intersect with each other
                        if check_objects_intersection(synthnova_physics_simulator, object_geoms_dict, object_poses_dict):
                            # Intersection detected, close simulator and prepare to retry
                            print(f"[WARNING] {combination_name} {split} frame {i} - object intersection detected, preparing to retry")
                            synthnova_physics_simulator.close()
                            retry_count += 1
                            continue
                        
                        # All objects are on table and no intersection, collect data
                        collect_and_save_sample_yolo_multi(img_name, label_name, split, galbot_interface, object_geoms_dict, object_poses_dict)
                        print(f"[INFO] {combination_name} {split} collected {i}/{len(indices)} frames")
                        success = True
                    else:
                        # Some objects fell off table, close simulator and prepare to retry
                        print(f"[WARNING] {combination_name} {split} frame {i} - some objects fell off table, preparing to retry")
                        synthnova_physics_simulator.close()
                        retry_count += 1
                        continue
                    
                    synthnova_physics_simulator.close()

    
    print(f"\n\n[INFO] Data collection completed")

if __name__ == "__main__":
    main() 
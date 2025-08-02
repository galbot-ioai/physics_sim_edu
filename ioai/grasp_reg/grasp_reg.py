import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, Any, Tuple
import copy


class GraspRegistration:
    """Register the grasp labels.

    Note:
        Assume that the object is placed on the table.
        x-axis is the longer side of the object.
        y-axis is the shorter side of the object.
        z-axis is the height of the object.
    """

    def __init__(self):
        x_p = np.array([1.0, 0, 0])
        self.x_p = x_p / np.linalg.norm(x_p)

        z_n = np.array([0, 0, -1.0])
        self.z_n = z_n / np.linalg.norm(z_n)

    def move_matrix_along_x_axis(self, matrix, distance):
        """Move the matrix along the x-axis.
        Args:
            matrix (np.ndarray): The input matrix.
            distance (float): The distance to move in meters.
        Returns:
            np.ndarray: The moved matrix.
        """
        transform_matrix = np.array(
            [
                [1, 0, 0, distance],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        new_matrix = matrix @ transform_matrix
        return new_matrix

    def move_matrix_along_y_axis(self, matrix, distance):
        """Move the matrix along the y-axis.
        Args:
            matrix (np.ndarray): The input matrix.
            distance (float): The distance to move in meters.
        Returns:
            np.ndarray: The moved matrix.
        """
        transform_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, distance],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        new_matrix = matrix @ transform_matrix
        return new_matrix

    def move_matrix_along_z_axis(self, matrix, distance):
        """Move the matrix along the z-axis.
        Args:
            matrix (np.ndarray): The input matrix.
            distance (float): The distance to move in meters.
        Returns:
            np.ndarray: The moved matrix.
        """
        transform_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, distance],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        new_matrix = matrix @ transform_matrix
        return new_matrix

    def rotate_matrix_along_x_axis(self, matrix, angle):
        """Rotate the matrix along the z-axis.
        Args:
            matrix (np.ndarray): The input matrix.
            angle (float): The angle to rotate in degrees.
        Returns:
            np.ndarray: The rotated matrix.
        """
        theta = np.deg2rad(angle)
        transform_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(theta), -np.sin(theta), 0],
                [0, np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        new_matrix = matrix @ transform_matrix
        return new_matrix

    def rotate_matrix_along_y_axis(self, matrix, angle):
        """Rotate the matrix along the y-axis.
        Args:
            matrix (np.ndarray): The input matrix.
            angle (float): The angle to rotate in degrees.
        Returns:
            np.ndarray: The rotated matrix.
        """
        theta = np.deg2rad(angle)
        transform_matrix = np.array(
            [
                [np.cos(theta), 0, np.sin(theta), 0],
                [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        new_matrix = matrix @ transform_matrix
        return new_matrix

    def rotate_matrix_along_z_axis(self, matrix, angle):
        """Rotate the matrix along the z-axis.
        Args:
            matrix (np.ndarray): The input matrix.
            angle (float): The angle to rotate in degrees.
        Returns:
            np.ndarray: The rotated matrix.
        """
        theta = np.deg2rad(angle)
        transform_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        new_matrix = matrix @ transform_matrix
        return new_matrix

    @staticmethod
    def quat_pose_to_se3(quat_pose: np.ndarray) -> np.ndarray:
        """Convert a pose in quaternion representation to SE(3) representation."""
        if not isinstance(quat_pose, np.ndarray) or quat_pose.shape != (7,):
            raise ValueError("Input must be a numpy array of shape (7,).")

        position, quat = quat_pose[:3], quat_pose[3:]
        quat = quat / np.linalg.norm(quat)  # Normalize quaternion
        rot_mat = R.from_quat(quat).as_matrix()

        se3 = np.eye(4)
        se3[:3, :3] = rot_mat
        se3[:3, 3] = position
        return se3

    @staticmethod
    def se3_to_quat_pose(se3: np.ndarray) -> np.ndarray:
        """Convert a pose in SE(3) representation to quaternion representation."""
        if not isinstance(se3, np.ndarray) or se3.shape != (4, 4):
            raise ValueError("Input must be a numpy array of shape (4, 4).")

        position = se3[:3, 3]
        quat = R.from_matrix(se3[:3, :3]).as_quat()
        return np.hstack([position, quat])

    def _modify_se3(self, se3: np.ndarray) -> np.ndarray:
        """Check if the pose is valid and modify it."""
        # make sure the z-axis is pointing to the front
        grasp_z_p = se3[:3, 2]
        dot = np.dot(grasp_z_p, self.x_p)
        if dot < 0:
            se3[:3, 2] = -se3[:3, 2]
        z = copy.deepcopy(se3[:3, 2])

        # select the vector that is closer to the reference vector as the x-axis
        dot1 = np.dot(se3[:3, 0], self.z_n)
        dot2 = np.dot(se3[:3, 1], self.z_n)
        if dot1 > dot2:
            x = copy.deepcopy(se3[:3, 0])
            y = np.cross(z, x)
        else:
            x = copy.deepcopy(se3[:3, 1])
            y = np.cross(z, x)
        se3[:3, 0] = x
        se3[:3, 1] = y
        se3[:3, 2] = z
        return se3

    def _rotate_sample_se3(self, se3, angle_range, angle_step, axis: str = "z"):
        """Sample SE(3) in rotation mode"""
        grasp_se3_list = [se3]
        sample_num = int((angle_range[1] - angle_range[0]) / angle_step)
        angle_list = np.linspace(angle_range[0], angle_range[1], sample_num)
        if axis == "x":
            rotate_func = self.rotate_matrix_along_x_axis
        elif axis == "y":   
            rotate_func = self.rotate_matrix_along_y_axis
        elif axis == "z":
            rotate_func = self.rotate_matrix_along_z_axis
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")
        for angle in angle_list:
            sampled_se3 = rotate_func(se3, angle)
            grasp_se3_list.append(sampled_se3)
        return grasp_se3_list

    def _adjust_se3(self, se3: np.ndarray) -> np.ndarray:
        """Adjust the se3 to make sure the z-axis is pointing to the front."""
        grasp_z_p = se3[:3, 2]
        dot = np.dot(grasp_z_p, self.x_p)
        if dot < 0:
            grasp_se3 = self.rotate_matrix_along_x_axis(se3, 180)
        else:
            grasp_se3 = se3.copy()
        return grasp_se3

    def register_grasp(
        self, part_id: str, object_pose: np.ndarray
    ) -> Tuple[list, float]:
        """Register the grasp pose based on the object pose.
        Args:
            object_pose (np.ndarray): The object pose in quaternion representation or SE(3) representation.

        Returns:
            List[np.ndarray]: A list of grasp poses in SE(3) representation.
            float: The gripper width.
        """
        if object_pose.shape == (7,):
            object_se3 = self.quat_pose_to_se3(object_pose)
        elif object_pose.shape == (4, 4):
            object_se3 = object_pose.copy()
        else:
            raise ValueError("Input must be a numpy array of shape (7,) or (4, 4).")

        assert part_id in ["power_drill", "extrusion", "toy", "cube", "mug"], \
            f"Unsupported part_id: {part_id}"
            
        grasp_se3_list = []
        gripper_width = 0.5
        if part_id == "power_drill":
            se3 = object_se3.copy()
            se3 = self.move_matrix_along_z_axis(se3, 0.055)
            se3 = self.move_matrix_along_y_axis(se3, -0.02)
            se3 = self.rotate_matrix_along_y_axis(se3, 180)
            grasp_se3_list.append(se3)

            se3 = object_se3.copy()
            se3 = self.move_matrix_along_z_axis(se3, 0.055)
            se3 = self.move_matrix_along_y_axis(se3, -0.02)
            grasp_se3_list.append(se3)

            se3 = object_se3.copy()
            se3 = self.move_matrix_along_z_axis(se3, 0.12)
            se3 = self.rotate_matrix_along_y_axis(se3, 90)
            se3 = self.rotate_matrix_along_x_axis(se3, 90)
            grasp_se3_list.append(se3)

            gripper_width = 0.7
        elif part_id == "extrusion":
            se3 = object_se3.copy()
            se3 = self.move_matrix_along_z_axis(se3, 0.015)
            se3 = self.rotate_matrix_along_z_axis(se3, 90)
            se3 = self.rotate_matrix_along_x_axis(se3, 90)
            grasp_se3_list = self._rotate_sample_se3(
                se3, angle_range=(-180, 180), angle_step=10
            )
            gripper_width = 0.6

        elif part_id == "toy":
            se3 = object_se3.copy()
            se3 = self.move_matrix_along_z_axis(se3, 0.05)
            grasp_se3_list = self._rotate_sample_se3(
                se3, angle_range=(-180, 180), angle_step=10
            )
            gripper_width = 0.6

        elif part_id == "cube":
            se3 = object_se3.copy()
            grasp_se3_list1 = self._rotate_sample_se3(
                se3, angle_range=(-180, 180), angle_step=10
            )
            grasp_se3_list += grasp_se3_list1

            se3 = object_se3.copy()
            se3 = self.rotate_matrix_along_x_axis(se3, 90)
            grasp_se3_list2 = self._rotate_sample_se3(
                se3, angle_range=(-180, 180), angle_step=10
            )
            grasp_se3_list += grasp_se3_list2

            gripper_width = 0.6
        elif part_id == "mug":
            se3 = object_se3.copy()
            se3 = self.move_matrix_along_z_axis(se3, 0.038)
            grasp_se3_list1 = self._rotate_sample_se3(
                se3, angle_range=(-180, 180), angle_step=10
            )
            grasp_se3_list += grasp_se3_list1

            se3 = object_se3.copy()
            se3 = self.rotate_matrix_along_y_axis(se3, -90)
            se3 = self.move_matrix_along_x_axis(se3, 0.025)
            grasp_se3_list2 = self._rotate_sample_se3(
                se3, angle_range=(30, 150), angle_step=10, axis="x"
            )
            grasp_se3_list += grasp_se3_list2

            se3 = object_se3.copy()
            se3 = self.move_matrix_along_z_axis(se3, 0.055)
            se3 = self.rotate_matrix_along_y_axis(se3, 90)
            grasp_se3_list3 = self._rotate_sample_se3(
                se3, angle_range=(30, 150), angle_step=10, axis="x"
            )
            grasp_se3_list += grasp_se3_list3

            gripper_width = 0.6
        

        return grasp_se3_list, gripper_width

    def predict_grasp(self, part_id: str, object_pose: np.ndarray) -> Dict[str, Any]:
        """Predict the grasp pose from the grasp SE(3) list.
        Args:
            part_id (str): The part ID.
            object_pose (np.ndarray): The object pose in quaternion representation or SE(3) representation.
        Returns:
            Dict[str, Any]: A dictionary containing the part ID, object pose, grasp SE(3), grasp pose, and gripper width.
        """
        grasp_se3_list, gripper_width = self.register_grasp(part_id, object_pose)
        grasp_se3_list = sorted(
            grasp_se3_list, key=lambda x: np.dot(x[:3, 0], self.z_n), reverse=True
        )
        grasp_se3 = self._adjust_se3(grasp_se3_list[0])
        grasp_pose = self.se3_to_quat_pose(grasp_se3)

        grasp = {
            "part_id": part_id,
            "object_pose": object_pose,
            "grasp_se3": grasp_se3,
            "grasp_pose": grasp_pose,
            "gripper_width": gripper_width,
        }

        return grasp

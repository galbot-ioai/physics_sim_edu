import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Union


class VIS3D:

    def __init__(self):
        self.geometry_list = []

    def visualize(self):
        o3d.visualization.draw_geometries(self.geometry_list)

    def load_point_cloud(self, file_path: str):
        return o3d.io.read_point_cloud(file_path)

    def add_point_cloud(
        self, pcd: o3d.geometry.PointCloud, color: list = [0.5, 0.5, 0.5]
    ):
        pcd.paint_uniform_color(color)
        self.geometry_list.append(pcd)
        return pcd

    def down_sample(self, pcd: o3d.geometry.PointCloud, voxel_size: float = 0.05):
        return pcd.voxel_down_sample(voxel_size=voxel_size)

    def add_obb(
        self, pcd: o3d.geometry.PointCloud, color: list = [1, 0, 0], scale: float = 1.0
    ):
        obb = pcd.get_oriented_bounding_box()
        print("OBB Center:", obb.center)
        print("OBB Extent (Width, Height, Depth):", obb.extent)
        print("OBB Rotation Matrix:\n", obb.R)
        obb.color = color
        obb.scale(scale, center=obb.center)
        self.geometry_list.append(obb)
        return obb

    def add_cylinder(self, center: list, radius: float, height: float, color: list):
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=height
        )
        cylinder.compute_vertex_normals()
        cylinder.paint_uniform_color(color)
        cylinder.translate(center)
        self.geometry_list.append(cylinder)
        return cylinder

    def add_sphere(self, center: list, radius: float, color: list):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        sphere.translate(center)
        self.geometry_list.append(sphere)
        return sphere

    def add_ros_map(self, yaml_file: str, pgm_file: str):
        # TODO: confirm the correctness of the map
        import yaml
        from PIL import Image

        with open(yaml_file, "r") as file:
            map_metadata = yaml.safe_load(file)
        image_path = pgm_file
        resolution = map_metadata["resolution"]
        origin = map_metadata["origin"]
        image = Image.open(image_path)
        image = image.convert("L")
        map_data = np.array(image)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            [
                [i * resolution + origin[0], j * resolution + origin[1], 0]
                for i in range(map_data.shape[0])
                for j in range(map_data.shape[1])
                if map_data[i, j] == 0
            ]
        )
        self.geometry_list.append(pcd)
        map_pcd = pcd
        return map_pcd

    def add_frame(self, pose: Union[np.ndarray, list], size: float) -> o3d.geometry.TriangleMesh:
        if isinstance(pose, list):
            pose = np.array(pose)
        if pose.shape == (6,):
            position = pose[:3]
            euler = pose[3:]
            quaternion = R.from_euler("xyz", euler).as_quat()
        elif pose.shape == (7,):
            position = pose[:3]
            quaternion = pose[3:]
        elif pose.shape == (3,):
            position = pose
            quaternion = [0, 0, 0, 1]
        elif pose.shape == (4, 4):
            position = pose[:3, 3]
            quaternion = R.from_matrix(pose[:3, :3]).as_quat()
        else:
            raise ValueError("Invalid pose shape")
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        frame.rotate(
            R=R.from_quat(quaternion).as_matrix(),
            center=(0, 0, 0),
        )
        frame.translate(position)
        self.geometry_list.append(frame)
        return frame

    def fit_plane(self, pcd: o3d.geometry.PointCloud, threshold: float, max_n: int):
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=threshold, ransac_n=max_n, num_iterations=1000
        )
        [a, b, c, d] = plane_model
        print(f"Plane model: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        return plane_model, inlier_cloud, outlier_cloud
    
if __name__ == "__main__":

    pass
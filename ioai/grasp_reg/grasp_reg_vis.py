import pprint
from vis_3d import VIS3D
import numpy as np
from grasp_reg import GraspRegistration
import argparse
from pprint import pprint

if __name__ == "__main__":
    vis = VIS3D()
    parser = argparse.ArgumentParser(description="Grasp Registration Visualization")
    parser.add_argument("--object_name", type=str, default="extrusion",
                        choices=["power_drill", "extrusion", "toy", "cube", "mug"],
                        help="Object name to use for grasp registration")
    args = parser.parse_args()

    if False:  # Debug

        log_dir = "data/20250716212935-28210-2JTA0"
        part_id = np.load(f"{log_dir}/part_id.npy")
        print("part_id:", part_id)

        # target_pcd_path = f"{log_dir}/target_pointcloud.ply"
        scene_pcd_path = f"{log_dir}/scene_pointcloud.ply"
        # target_pcd = vis.load_point_cloud(target_pcd_path)
        scene_pcd = vis.load_point_cloud(scene_pcd_path)
        vis.add_point_cloud(scene_pcd, color=[0.5, 0.5, 0.5])
        # vis.add_point_cloud(target_pcd, color=[0.2, 0.7, 0.7])
        
        origin_se3 = np.eye(4)
        vis.add_frame(origin_se3, size=0.2)

        object_se3 = np.load(f"{log_dir}/object_se3.npy")
        vis.add_frame(object_se3, size=0.2)

        rgp = RegisterGraspPose()
        grasp_se3_list, grasp_width = rgp.predict(part_id, object_se3)
        for grasp_se3 in grasp_se3_list[:10]:
            vis.add_frame(grasp_se3, size=0.1)

    else:  # Registration
        object_name = args.object_name
        pcd_path = f"models/{object_name}.pcd"
        pcd = vis.load_point_cloud(pcd_path)
        vis.add_point_cloud(pcd, color=[0.5, 0.5, 0.5])
        
        obb = vis.add_obb(pcd, color=[0.2, 0.7, 0.7])
        pprint(obb.extent)

        object_se3 = np.eye(4)
        vis.add_frame(object_se3, size=0.05)

        rgp = GraspRegistration()
        grasp_se3_list, gripper_width = rgp.register_grasp(object_name, object_se3)
        for grasp_se3 in grasp_se3_list:
            vis.add_frame(grasp_se3, size=0.1)

        grasp = rgp.predict_grasp(object_name, object_se3)
        print("Grasp SE3:", grasp["grasp_se3"])
        vis.add_frame(grasp["grasp_se3"], size=0.3)

    vis.visualize()

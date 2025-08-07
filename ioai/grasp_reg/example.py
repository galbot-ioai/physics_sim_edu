from grasp_reg import GraspRegistration
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grasp Registration")
    parser.add_argument(
        "--object_name",
        type=str,
        default="extrusion",
        choices=["power_drill", "extrusion", "toy", "cube"],
        help="Object name to use for grasp registration",
    )
    args = parser.parse_args()

    object_se3 = np.eye(4)
    rgp = GraspRegistration()
    grasp = rgp.predict_grasp(args.object_name, object_se3)
    print("Grasp SE3:", grasp["grasp_se3"])
    print("Grasp Quat Pose:", grasp["grasp_pose"])

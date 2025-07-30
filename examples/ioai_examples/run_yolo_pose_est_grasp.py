"""
Example of using YOLO + pose estimation model in IoaiGraspEnv
"""

import os
from pathlib import Path
from ioai_grasp_env import IoaiGraspEnv
from yolo_pose_estimation_model import YoloPoseEstimationModel

def main():
    """Main function to run IoaiGraspEnv with YOLO pose estimation"""
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Configuration paths
    # yolo_model_path = script_dir / ".." / "yolo_seg_examples" / "real_all_class_0730.pt"
    yolo_model_path = script_dir / ".." / "yolo_seg_examples" / "best.pt"
    cad_models_dir = script_dir / ".." / "pose_est_examples" / "models"
    
    # Camera intrinsic parameters (adjust based on your camera)
    camera_matrix = [637.7254326533274, 637.7254326533274, 640.0, 360.0]
    
    # Check if required files exist
    if not yolo_model_path.exists():
        print(f"Error: YOLO model not found at {yolo_model_path}")
        return
    
    if not cad_models_dir.exists():
        print(f"Error: CAD models directory not found at {cad_models_dir}")
        return
    
    # Create YOLO pose estimation model
    pose_estimation_model = YoloPoseEstimationModel(
        yolo_model_path=str(yolo_model_path),
        cad_models_dir=str(cad_models_dir),
        camera_matrix=camera_matrix
    )
    
    # Create environment with pose estimation model
    env = IoaiGraspEnv(
        headless=False,
        pose_estimation_model=pose_estimation_model
    )
    
    print("Starting IoaiGraspEnv with YOLO pose estimation...")
    print("Press Ctrl+C to stop")
    
    try:
        # Add physics callback and run
        env.simulator.add_physics_callback("pick_and_place", env.pick_and_place_callback)
        env.simulator.loop()
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        env.simulator.close()

if __name__ == "__main__":
    main() 
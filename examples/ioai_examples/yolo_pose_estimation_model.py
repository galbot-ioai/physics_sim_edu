"""
YOLO segmentation + point cloud registration pose estimation model
"""

import os
import cv2
import numpy as np
from typing import List, Optional, Tuple
from ultralytics import YOLO
from pathlib import Path

# Import pose estimation module
from physics_simulator.utils.pose_estimation import PoseEstimator
from physics_simulator.utils.pose_estimation import PoseEstimationResult, PoseEstimationModel

from physics_simulator.utils import preprocess_depth

class YoloPoseEstimationModel(PoseEstimationModel):
    """YOLO segmentation + point cloud registration pose estimation model"""
    
    def __init__(self, yolo_model_path: str, cad_models_dir: str, camera_matrix: List[float]):
        """
        Initialize YOLO + pose estimation model
        
        Args:
            yolo_model_path: Path to YOLO segmentation model
            cad_models_dir: Directory containing CAD models (.obj files)
            camera_matrix: Camera intrinsic parameters [fx, fy, cx, cy]
        """
        self.yolo_model_path = yolo_model_path
        self.cad_models_dir = Path(cad_models_dir)
        self.camera_matrix = camera_matrix
        
        # Load YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        print(f"Loaded YOLO model: {yolo_model_path}")
        
        # Initialize pose estimator
        self.pose_estimator = PoseEstimator(
            camera_matrix=camera_matrix,
            depth_scale=0.001,
            model_scale_factor=None,
            visualize=False,  # Disable visualization for real-time use
        )
        
        # Get available CAD models
        self.available_models = self._get_available_cad_models()
        print(f"Available CAD models: {list(self.available_models.keys())}")
    
    def _get_available_cad_models(self) -> dict:
        """Get available CAD models from directory"""
        models = {}
        for obj_file in self.cad_models_dir.glob("*.obj"):
            model_name = obj_file.stem.lower()
            models[model_name] = str(obj_file)
        return models
    
    def estimate_poses(self, rgb_image: np.ndarray, depth_image: Optional[np.ndarray] = None) -> List[PoseEstimationResult]:
        """
        Estimate object poses using YOLO segmentation + point cloud registration
        
        Args:
            rgb_image: RGB image from camera
            depth_image: Depth image from camera (required for pose estimation)
            
        Returns:
            List of pose estimation results
        """
        pose_results = []
        
        if depth_image is None:
            print("Warning: Depth image required for pose estimation")
            return pose_results
        
        # Step 1: YOLO segmentation to get object masks
        yolo_results = self.yolo_model(rgb_image, verbose=False)
        
        if not yolo_results or yolo_results[0].masks is None:
            return pose_results
        
        # Step 2: Process each detected object
        for i, (mask, box, conf, cls_id) in enumerate(zip(
            yolo_results[0].masks.data.cpu().numpy(),
            yolo_results[0].boxes.xyxy.cpu().numpy(),
            yolo_results[0].boxes.conf.cpu().numpy(),
            yolo_results[0].boxes.cls.cpu().numpy()
        )):
            class_name = self.yolo_model.names[int(cls_id)]
            
            # Check if CAD model exists for this class
            if class_name.lower() not in self.available_models:
                print(f"No CAD model found for class: {class_name}")
                continue
            
            # Step 3: Estimate pose using point cloud registration
            pose_result = self._estimate_single_pose(
                rgb_image, depth_image, mask, class_name, conf
            )
            
            if pose_result is not None:
                pose_results.append(pose_result)
        
        return pose_results
    
    def _estimate_single_pose(self, rgb_image: np.ndarray, depth_image: np.ndarray, 
                            mask: np.ndarray, class_name: str, confidence: float) -> Optional[PoseEstimationResult]:
        """Estimate pose for a single object using point cloud registration"""
        try:
            # Prepare mask for pose estimation
            mask_resized = cv2.resize(
                mask.astype(np.float32),
                (rgb_image.shape[1], rgb_image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
            
            # Save temporary files for pose estimation
            temp_dir = Path("pose_est_temp")
            temp_dir.mkdir(exist_ok=True)
            
            rgb_temp = temp_dir / "temp_rgb.jpg"
            depth_temp = temp_dir / "temp_depth.png"
            mask_temp = temp_dir / "temp_mask.png"

            depth_data = preprocess_depth(
                depth_image,
                scale=1000,  # m to mm
                min_value=0.0,
                max_value=5 * 1000,  # 5m to mm
                data_type=np.uint16,
            )
            
            # Convert RGB to BGR for OpenCV imwrite
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(rgb_temp), bgr_image)
            cv2.imwrite(str(depth_temp), depth_data)
            cv2.imwrite(str(mask_temp), mask_binary)
            
            # Get CAD model path
            cad_name = class_name.lower()
            
            # Run pose estimation
            pose_matrix = self.pose_estimator.estimate_pose(
                rgb_path=str(rgb_temp),
                depth_path=str(depth_temp),
                mask_path=str(mask_temp),
                cad_name=cad_name
            )
            
            # Clean up temporary files
            for temp_file in [rgb_temp, depth_temp, mask_temp]:
                if temp_file.exists():
                    temp_file.unlink()
            
            if pose_matrix is None:
                return None
            
            # Extract position and orientation from pose matrix
            position = pose_matrix[:3, 3]  # Translation
            rotation_matrix = pose_matrix[:3, :3]  # Rotation matrix
            
            # Convert rotation matrix to quaternion
            from scipy.spatial.transform import Rotation
            quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
            
            # Create pose estimation result
            return PoseEstimationResult(
                class_name=class_name,
                position=position,
                orientation=quaternion,
                confidence=confidence,
                segmentation_mask=mask_binary,
                bbox=None  # Could extract from mask if needed
            )
            
        except Exception as e:
            print(f"Error estimating pose for {class_name}: {e}")
            return None

def create_yolo_pose_estimation_model(
    yolo_model_path: str = "examples/yolo_seg_examples/best.pt",
    cad_models_dir: str = "examples/pose_est_examples/models",
    camera_matrix: List[float] = None
) -> YoloPoseEstimationModel:
    """
    Factory function to create YOLO pose estimation model with default parameters
    
    Args:
        yolo_model_path: Path to YOLO segmentation model
        cad_models_dir: Directory containing CAD models (.obj files)
        camera_matrix: Camera intrinsic parameters [fx, fy, cx, cy]
        
    Returns:
        YoloPoseEstimationModel instance
    """
    # Default camera matrix for RealSense D436
    if camera_matrix is None:
        camera_matrix = [637.7254326533274, 637.7254326533274, 640.0, 360.0]  # [fx, fy, cx, cy]
    
    return YoloPoseEstimationModel(
        yolo_model_path=yolo_model_path,
        cad_models_dir=cad_models_dir,
        camera_matrix=camera_matrix
    )

if __name__ == "__main__":
    pass

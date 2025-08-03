import numpy as np
import cv2
import open3d as o3d
import copy
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import trimesh
from pprint import pprint
from sklearn.decomposition import PCA
import os


class PoseEstimator:
    def __init__(
        self,
        camera_matrix,
        depth_scale=0.001,
        model_scale_factor=None,
        visualize=False,
        log_debug=True,
    ):
        """
        Initialize pose estimator

        Parameters:
            camera_matrix: Camera intrinsic parameters [fx, fy, cx, cy]
            depth_scale: Depth scaling factor (default 0.001, mm to m)
            model_scale_factor: Model scaling factor (optional)
            visualize: Whether to visualize intermediate results (default False)
            debug: Whether to enable debug mode (default False)
        """
        self.camera_matrix = camera_matrix
        self.depth_scale = depth_scale
        self.model_scale_factor = model_scale_factor
        self.visualize = visualize
        self.output_dir = None
        self.debug = log_debug
        self.cad_name = None
        self.debug_dir = None

    def estimate_pose(self, rgb_path, depth_path, mask_path, cad_name):
        """
        Estimate 6D pose of object

        Parameters:
            rgb_path: RGB image path
            depth_path: Depth image path
            mask_path: Mask image path
            cad_name: CAD model name ('power_drill', 'bin', 'cube', 'mug', 'extrusion')

        Returns:
            (pose_matrix | None): 4x4 pose transformation matrix or None if estimation fails
        """
        cur_time = time.strftime("%Y%m%d%H%M%S")
        self.output_dir = f"pose_logs/pose_est_res_{cur_time}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Debug mode
        if self.debug:
            self.debug_dir = f"pose_logs/pose_est_dbg_{cur_time}"
            os.makedirs(self.debug_dir, exist_ok=True)
            savez_kwargs = {
                "camera_matrix": self.camera_matrix,
                "depth_scale": self.depth_scale,
                "visualize": self.visualize,
                "output_dir": self.output_dir,
            }
            if self.model_scale_factor is not None:
                savez_kwargs["model_scale_factor"] = self.model_scale_factor
            else:
                savez_kwargs["model_scale_factor"] = "None"
            np.savez(f"{self.debug_dir}/params_init.npz", **savez_kwargs)

        self.cad_name = cad_name

        try:
            # Load and process data
            scene_pcd, model_pcd = self.load_and_process_data(
                rgb_path, depth_path, mask_path, cad_name
            )

            # Estimate pose
            pose = self.estimate_pose_from_point_clouds(scene_pcd, model_pcd)
            if pose is None:
                pprint("Pose estimation failed, returning None")
                return None

            # Save pose matrix
            pose_path = os.path.join(self.output_dir, "pose_matrix.txt")
            self.save_pose_matrix(pose, pose_path)

            if self.visualize:
                # Visualize final registration result
                self.visualize_registration(
                    model_pcd,
                    scene_pcd,
                    pose,
                    os.path.join(self.output_dir, "final_registration.png"),
                )

            return pose

        except Exception as e:
            pprint(f"Pose estimation failed: {str(e)}")
            import traceback

            pprint(traceback.format_exc())
            return None

    def _enhance_depth_map(self, depth, max_depth=2000):
        """Enhanced depth map processing to reduce artifacts and improve quality"""
        # Apply morphological operations to clean up the depth map
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Close small holes
        depth_closed = cv2.morphologyEx(
            depth.astype(np.uint16), cv2.MORPH_CLOSE, kernel, iterations=1
        )

        # Remove small isolated regions
        depth_opened = cv2.morphologyEx(
            depth_closed, cv2.MORPH_OPEN, kernel, iterations=1
        )

        # Fill remaining holes using inpainting
        mask = np.uint8(depth_opened == 0) * 255
        if np.sum(mask) > 0:
            depth_filled = cv2.inpaint(
                depth_opened.astype(np.uint16),
                mask,
                inpaintRadius=3,
                flags=cv2.INPAINT_NS,
            )
        else:
            depth_filled = depth_opened

        # Apply Gaussian blur for smoothing (safer than edge-preserving filter)
        depth_filtered = cv2.GaussianBlur(depth_filled, (3, 3), 0.8)

        # Limit maximum depth
        depth_filtered[depth_filtered > max_depth] = max_depth

        return depth_filtered.astype(np.uint16)

    def _refine_mask(self, mask, kernel_size=5, iterations=1):
        """Optimize mask: fill small holes and smooth edges"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Close operation: fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        # Open operation: remove isolated points
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        return mask

    def _filter_depth_artifacts(self, depth, mask_binary, max_depth_jump=50):
        """
        Filter depth artifacts including flying pixels and unrealistic depth jumps
        """
        depth_filtered = depth.copy().astype(np.float32)

        # Apply Gaussian blur for initial smoothing (safer than bilateral filter)
        depth_filtered = cv2.GaussianBlur(depth_filtered, (5, 5), 1.0)

        # Apply median filter to remove salt-and-pepper noise
        depth_filtered = cv2.medianBlur(depth_filtered.astype(np.uint16), 5).astype(
            np.float32
        )

        # Detect and remove depth discontinuities (flying pixels)
        kernel = np.ones((3, 3), np.uint8)
        depth_eroded = cv2.erode(depth_filtered.astype(np.uint16), kernel, iterations=1)

        # Remove pixels where depth differs significantly from neighbors
        depth_diff = np.abs(depth_filtered - depth_eroded.astype(np.float32))
        outlier_mask = depth_diff > max_depth_jump

        # Only apply within the object mask to avoid affecting background
        outlier_mask = outlier_mask & mask_binary
        depth_filtered[outlier_mask] = 0

        return depth_filtered.astype(np.uint16)

    def _filter_depth_discontinuities(
        self, z_values, y_idxs, x_idxs, depth_image, threshold_factor=0.1
    ):
        """
        Filter points based on local depth discontinuities to remove flying pixels
        """
        if len(z_values) < 10:
            return z_values, y_idxs, x_idxs

        # Calculate local depth statistics
        depth_median = np.median(z_values)
        depth_std = np.std(z_values)

        # Define reasonable depth range
        min_depth = max(0.1, depth_median - 3 * depth_std)
        max_depth = depth_median + 3 * depth_std

        # Filter based on global depth range
        valid_global = (z_values >= min_depth) & (z_values <= max_depth)

        # Local neighborhood analysis for remaining points
        valid_local = np.ones(len(z_values), dtype=bool)

        for i, (y, x, z) in enumerate(zip(y_idxs, x_idxs, z_values)):
            if not valid_global[i]:
                continue

            # Check 3x3 neighborhood
            y_min, y_max = max(0, y - 1), min(depth_image.shape[0], y + 2)
            x_min, x_max = max(0, x - 1), min(depth_image.shape[1], x + 2)

            neighborhood = depth_image[y_min:y_max, x_min:x_max]
            valid_neighbors = neighborhood[neighborhood > 0] * self.depth_scale

            if len(valid_neighbors) < 3:
                valid_local[i] = False
                continue

            # Check if current depth is consistent with neighbors
            neighbor_median = np.median(valid_neighbors)
            depth_diff = abs(z - neighbor_median)

            # Adaptive threshold based on depth value
            adaptive_threshold = max(0.01, neighbor_median * threshold_factor)

            if depth_diff > adaptive_threshold:
                valid_local[i] = False

        # Combine filters
        valid_mask = valid_global & valid_local

        filtered_z = z_values[valid_mask]
        filtered_y = y_idxs[valid_mask]
        filtered_x = x_idxs[valid_mask]

        pprint(f"Depth filtering: {len(z_values)} -> {len(filtered_z)} points")

        return filtered_z, filtered_y, filtered_x
 
    def load_and_process_data(self, rgb_path, depth_path, mask_path, cad_name):
        """
        Load and process input data
        """
        try:
            pprint("Starting to load and process data...")

            # Load image data
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                raise FileNotFoundError(f"Cannot load RGB image: {rgb_path}")
            if self.debug:
                cv2.imwrite(os.path.join(self.debug_dir, "rgb_image.png"), rgb)

            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth is None:
                raise FileNotFoundError(f"Cannot load depth image: {depth_path}")
            if self.debug:
                cv2.imwrite(os.path.join(self.debug_dir, "depth_image.png"), depth)
            depth = self._enhance_depth_map(depth)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Cannot load mask image: {mask_path}")
            if self.debug:
                cv2.imwrite(os.path.join(self.debug_dir, "mask_image.png"), mask)
            mask = self._refine_mask(mask)

            pprint(
                f"Image dimensions: RGB={rgb.shape}, Depth={depth.shape}, Mask={mask.shape}"
            )

            if self.debug:
                savez_kwargs = {
                    "rgb_path": f"{self.debug_dir}/rgb_image.png",
                    "depth_path": f"{self.debug_dir}/depth_image.png",
                    "mask_path": f"{self.debug_dir}/mask_image.png",
                    "cad_name": cad_name,
                }
                np.savez(f"{self.debug_dir}/params_est.npz", **savez_kwargs)

            # Visualize input data
            if self.visualize:
                self.visualize_input_data(rgb, depth, mask)

            # Adaptive mask threshold processing
            mask_binary = self.process_mask(mask)

            # Generate scene point cloud
            scene_cloud = self.generate_scene_point_cloud(rgb, depth, mask_binary)

            # Load CAD model
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            cad_path = os.path.join(cur_dir, "models", f"{cad_name}.obj")
            pprint(f"Loading CAD model: {cad_path}")
            cad_mesh = self.load_mesh_with_fallback(cad_path)

            # Apply scaling
            cad_mesh = self.scale_cad_model(cad_mesh)

            # Uniform point cloud sampling with higher density for better features
            model_cloud = cad_mesh.sample_points_uniformly(number_of_points=10000)

            model_voxel = 0.004 if self.cad_name == "bin" else 0.002
            model_cloud = self.preprocess_point_cloud(
                model_cloud, voxel_size=model_voxel
            )
            if len(model_cloud.points) > 0:
                ratio = len(scene_cloud.points) / len(model_cloud.points)
                scene_voxel = model_voxel * ratio ** (1 / 3)
                scene_cloud = self.preprocess_point_cloud(
                    scene_cloud, voxel_size=scene_voxel
                )
            else:
                scene_voxel = model_voxel * 2
                scene_cloud = self.preprocess_point_cloud(
                    scene_cloud, voxel_size=scene_voxel
                )

            pprint(f"Scene point cloud: {len(scene_cloud.points)} points")
            pprint(f"Model point cloud: {len(model_cloud.points)} points")

            # Visualize original point clouds
            if self.visualize:
                self.visualize_point_clouds(
                    scene_cloud,
                    model_cloud,
                    os.path.join(self.output_dir, "original_pointclouds.png"),
                )

            return scene_cloud, model_cloud

        except Exception as e:
            pprint(f"Data processing error: {str(e)}")
            raise

    def load_mesh_with_fallback(self, cad_path):
        """
        Load CAD model (with fallback method for non-triangular meshes)
        """
        try:
            # First try to load with Open3D directly
            mesh = o3d.io.read_triangle_mesh(cad_path)
            if mesh.has_vertices():
                pprint(f"Successfully loaded CAD model: {cad_path}")
                return mesh

            # If Open3D loading fails, try using trimesh to load and convert
            pprint(
                f"Open3D cannot directly load {cad_path}, trying trimesh conversion..."
            )
            tmesh = trimesh.load(cad_path)
            if not isinstance(tmesh, trimesh.Trimesh):
                # If loaded mesh is not triangular, try to extract the first mesh
                if hasattr(tmesh, "geometry") and len(tmesh.geometry) > 0:
                    tmesh = next(iter(tmesh.geometry.values()))

            # Ensure it's a triangular mesh
            if not isinstance(tmesh, trimesh.Trimesh):
                raise ValueError(f"Cannot convert {cad_path} to triangular mesh")

            # Convert trimesh to Open3D mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(tmesh.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(tmesh.faces)

            # Try to load vertex colors
            if (
                hasattr(tmesh.visual, "vertex_colors")
                and tmesh.visual.vertex_colors is not None
            ):
                if len(tmesh.visual.vertex_colors) == len(tmesh.vertices):
                    colors = np.array(tmesh.visual.vertex_colors)[:, :3] / 255.0
                    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

            pprint(f"Successfully converted and loaded model: {cad_path}")
            return mesh

        except Exception as e:
            pprint(f"Failed to load CAD model: {str(e)}")
            raise ValueError(f"Cannot load CAD model: {cad_path}")

    def scale_cad_model(self, cad_mesh):
        """
        Scale CAD model to appropriate size
        """
        vertices = np.asarray(cad_mesh.vertices)
        if len(vertices) == 0:
            raise ValueError("CAD model has no vertex data")

        vertex_range = np.ptp(vertices, axis=0)
        pprint(f"CAD model vertex range: {vertex_range}")

        # Apply manual scaling factor or automatic scaling
        if self.model_scale_factor is not None:
            pprint(f"Applying manual scaling factor: {self.model_scale_factor}")
            cad_mesh.scale(self.model_scale_factor, center=tuple(np.zeros(3)))
        elif np.max(vertex_range) > 1.0:
            pprint("Automatically scaling CAD model to fit scene")
            cad_mesh.scale(self.depth_scale, center=tuple(np.zeros(3)))

        return cad_mesh

    def process_mask(self, mask):
        """
        Process mask image
        """
        if np.max(mask) <= 1:  # Binary mask (0-1)
            mask_binary = mask > 0
        else:  # Grayscale mask (0-255)
            # Use Otsu adaptive threshold
            _, mask_binary = cv2.threshold(
                mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            mask_binary = mask_binary > 0

        return mask_binary

    def visualize_input_data(self, rgb, depth, mask):
        """
        Visualize input data
        """
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        plt.title("RGB")

        plt.subplot(132)
        plt.imshow(
            depth,
            cmap="jet",
            vmin=float(np.percentile(depth[depth > 0], 5)),
            vmax=float(np.percentile(depth[depth > 0], 95)),
        )
        plt.title("Depth")

        plt.subplot(133)
        rgb_vis = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_vis)
        # Normalize mask to [0,1] for alpha
        mask_norm = mask.astype(float)
        if mask_norm.max() > 1:
            mask_norm = mask_norm / 255.0
        plt.imshow(mask_norm, cmap="jet", alpha=0.4)
        plt.title("RGB + Mask Overlay")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "input_data.png"), dpi=150)
        plt.show()
        plt.close()

    def generate_scene_point_cloud(self, rgb, depth, mask_binary):
        """
        Generate scene point cloud from RGB-D image and mask with anti-aliasing
        """
        fx, fy, cx, cy = self.camera_matrix
        height, width = depth.shape

        # Apply depth filtering to reduce noise and artifacts
        depth_filtered = self._filter_depth_artifacts(depth, mask_binary)

        # Create point cloud - only process mask region
        y_idxs, x_idxs = np.where(mask_binary)
        pprint(f"Mask region points: {len(y_idxs)}")

        # Filter invalid depth values
        valid_depth = depth_filtered[y_idxs, x_idxs] > 0
        y_idxs = y_idxs[valid_depth]
        x_idxs = x_idxs[valid_depth]

        if len(y_idxs) == 0:
            raise ValueError(
                "No valid depth points available for point cloud generation"
            )

        z_values = depth_filtered[y_idxs, x_idxs].astype(float) * self.depth_scale

        # # Apply depth discontinuity filtering to remove flying pixels and artifacts
        # z_values, y_idxs, x_idxs = self._filter_depth_discontinuities(
        #     z_values, y_idxs, x_idxs, depth_filtered
        # )

        # Convert to 3D coordinates (OpenCV coordinate system)
        x_cam = (x_idxs - cx) * z_values / fx
        y_cam = (y_idxs - cy) * z_values / fy

        points = np.column_stack((x_cam, y_cam, z_values))

        # Get colors and convert BGR to RGB
        color_values = rgb[y_idxs, x_idxs][:, ::-1] / 255.0

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(color_values)

        return pcd

    def preprocess_point_cloud(self, pcd, voxel_size=0.005):
        """
        Enhanced point cloud preprocessing: downsampling, anti-aliasing, denoising, normal estimation
        """
        # Downsampling
        pcd_down = pcd.voxel_down_sample(voxel_size)

        # Apply enhanced outlier removal specifically for streaking artifacts
        if len(pcd_down.points) > 10:
            # First pass: Statistical outlier removal for general noise
            pcd_clean, _ = pcd_down.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )

            # # Second pass: Enhanced outlier removal for streaking artifacts
            # if len(pcd_clean.points) > 10:
            #     pcd_clean = self._remove_streaking_artifacts(pcd_clean, voxel_size)
        else:
            pprint("Too few points in point cloud, skipping denoising")
            pcd_clean = pcd_down

        # Modify normal estimation parameters
        if len(pcd_clean.points) > 3:
            try:
                radius_normal = voxel_size * 4  # Increase search radius
                pcd_clean.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=radius_normal,
                        max_nn=50,  # Increase neighborhood points
                    )
                )
            except Exception:
                # Fallback to KNN search if radius estimation fails
                pcd_clean.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
                )

            # Ensure normals point toward camera (assuming camera at origin)
            pcd_clean.orient_normals_towards_camera_location(
                camera_location=np.array([0, 0, 0])
            )
        else:
            pprint("Insufficient points in point cloud, cannot estimate normals")

        return pcd_clean

    def _remove_streaking_artifacts(self, pcd, voxel_size):
        """
        Remove streaking artifacts (flying pixels and thin structures) from point cloud
        """
        points = np.asarray(pcd.points)
        if len(points) < 10:
            return pcd

        # Build KDTree for efficient neighbor search
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        # Parameters for streak detection
        min_neighbors = 8  # Minimum neighbors for a valid point
        search_radius = voxel_size * 5  # Search radius for neighbors
        max_linear_ratio = 0.3  # Maximum ratio of linear arrangement

        valid_indices = []

        for i, point in enumerate(points):
            # Find neighbors within radius
            [k, idx, _] = pcd_tree.search_radius_vector_3d(point, search_radius)

            if k < min_neighbors:
                continue  # Too few neighbors, likely an artifact

            # Get neighbor positions
            neighbor_points = points[idx[1:]]  # Exclude the point itself

            if len(neighbor_points) < 3:
                continue

            # Check if neighbors form a linear arrangement (indicating streaking)
            try:
                # Use PCA to analyze point distribution
                pca = PCA(n_components=3)
                pca.fit(neighbor_points - point)

                # Check explained variance ratios
                explained_ratios = pca.explained_variance_ratio_

                # If first component explains too much variance, points are linear
                if explained_ratios[0] > (1.0 - max_linear_ratio):
                    continue  # Skip this point as it's likely part of a streak

                # Additional check: ensure points are reasonably distributed
                distances = np.linalg.norm(neighbor_points - point, axis=1)
                if np.std(distances) < voxel_size * 0.5:
                    continue  # Points too clustered, might be noise

                valid_indices.append(i)

            except Exception:
                # If PCA fails, keep the point (conservative approach)
                valid_indices.append(i)

        # Create filtered point cloud
        if len(valid_indices) > 0:
            filtered_pcd = pcd.select_by_index(valid_indices)
            pprint(f"Streak filtering: {len(points)} -> {len(valid_indices)} points")
            return filtered_pcd
        else:
            pprint("Warning: Streak filtering removed all points, returning original")
            return pcd

    def _filter_geometric_inconsistencies(self, pcd, voxel_size, object_type):
        """
        Filter points that are geometrically inconsistent with cube/extrusion shapes
        """
        points = np.asarray(pcd.points)
        if len(points) < 20:
            return pcd

        # For cube and extrusion, we expect points to lie on planar surfaces
        # Use RANSAC to find dominant planes and filter outliers

        valid_indices = set(range(len(points)))

        # Find multiple planes (typically 3-6 for a cube/extrusion)
        max_planes = 6 if object_type in ["cube", "extrusion"] else 4
        min_points_per_plane = max(10, len(points) // 20)

        remaining_indices = list(range(len(points)))

        for plane_idx in range(max_planes):
            if len(remaining_indices) < min_points_per_plane:
                break

            # Create temporary point cloud from remaining points
            temp_pcd = pcd.select_by_index(remaining_indices)

            try:
                # Fit plane using RANSAC
                plane_model, inlier_indices = temp_pcd.segment_plane(
                    distance_threshold=voxel_size * 2, ransac_n=3, num_iterations=1000
                )

                if len(inlier_indices) < min_points_per_plane:
                    break

                # Convert local indices back to global indices
                global_inlier_indices = [remaining_indices[i] for i in inlier_indices]

                # Remove these points from remaining list
                remaining_indices = [
                    idx for idx in remaining_indices if idx not in global_inlier_indices
                ]

                pprint(f"Found plane {plane_idx + 1} with {len(inlier_indices)} points")

            except Exception:
                break

        # Remove points that don't belong to any plane (likely artifacts)
        outlier_indices = remaining_indices
        if len(outlier_indices) > 0:
            valid_indices = valid_indices - set(outlier_indices)
            pprint(f"Removed {len(outlier_indices)} geometric outliers")

        if len(valid_indices) > 0:
            filtered_pcd = pcd.select_by_index(list(valid_indices))
            return filtered_pcd
        else:
            pprint(
                "Warning: Geometric filtering removed all points, returning original"
            )
            return pcd

    def visualize_point_clouds(self, scene_cloud, model_cloud, filename):
        """
        Visualize point clouds
        """
        scene_temp = copy.deepcopy(scene_cloud)
        model_temp = copy.deepcopy(model_cloud)
        # scene_temp.paint_uniform_color([1, 0, 0])  # Red: scene point cloud
        model_temp.paint_uniform_color([0, 0, 1])  # Blue: model point cloud

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Original Point Clouds", width=1200, height=900)
        vis.add_geometry(scene_temp)
        vis.add_geometry(model_temp)

        # Set viewpoint for better visualization
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)

        vis.run()
        vis.capture_screen_image(filename)
        vis.destroy_window()

    def estimate_pose_from_point_clouds(self, scene_cloud, model_cloud):
        """
        Estimate pose from point clouds
        """
        pprint("Starting pose estimation process...")

        # Key improvement: move model point cloud to scene point cloud centroid position
        pprint("Moving model point cloud to scene point cloud centroid...")
        model_moved, T_translation = self.move_model_to_scene(model_cloud, scene_cloud)

        # Enhanced coarse registration with multiple attempts
        pprint("Performing improved coarse registration...")
        start_time = time.time()

        best_transform = None
        best_evaluation = None

        # Try multiple registration methods and keep the best result
        registration_methods = []
        reg_fitness = 0.0

        # if self.cad_name != "bin":
        if True:
            # Method 1: Teaser++ global registration
            try:
                import teaserpp_python

                def teaserpp_registration(source, target):
                    # Convert Open3D point clouds to numpy arrays
                    src_pts = np.asarray(source.points).T  # 3 x N
                    tgt_pts = np.asarray(target.points).T  # 3 x N

                    # Use FPFH features to find correspondences
                    src_fpfh = self.compute_fpfh(source, voxel_size=0.01)
                    tgt_fpfh = self.compute_fpfh(target, voxel_size=0.01)
                    src_idx, tgt_idx = self.find_correspondences(
                        src_fpfh, tgt_fpfh, source, target
                    )

                    if len(src_idx) < 4:
                        raise ValueError(
                            "Not enough correspondences for Teaser++ registration"
                        )

                    src_corr = src_pts[:, src_idx]
                    tgt_corr = tgt_pts[:, tgt_idx]

                    # Setup Teaser++ solver
                    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
                    solver_params.cbar2 = 1.0
                    solver_params.noise_bound = 0.01
                    solver_params.estimate_scaling = False
                    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
                    solver_params.rotation_gnc_factor = 1.4
                    solver_params.rotation_max_iterations = 100

                    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
                    solver.solve(src_corr, tgt_corr)
                    solution = solver.getSolution()

                    # Build transformation matrix
                    T_teaser = np.eye(4)
                    T_teaser[:3, :3] = solution.rotation
                    T_teaser[:3, 3] = solution.translation
                    return T_teaser

                transform_teaser = teaserpp_registration(model_moved, scene_cloud)
                eval_teaser = o3d.pipelines.registration.evaluate_registration(
                    model_moved, scene_cloud, 0.05, transform_teaser
                )
                registration_methods.append(("Teaser++", transform_teaser, eval_teaser))
                reg_fitness = eval_teaser.fitness
                pprint(f"Teaser++ fitness: {eval_teaser.fitness:.4f}")
            except Exception as e:
                pprint(f"Teaser++ registration failed: {str(e)}")

            # Method 2: RANSAC with smaller voxel size for finer features
            if reg_fitness < 0.5:
                try:
                    transform_ransac_fine = self.ransac_global_registration(
                        model_moved, scene_cloud, voxel_size=0.005
                    )
                    eval_ransac_fine = o3d.pipelines.registration.evaluate_registration(
                        model_moved, scene_cloud, 0.05, transform_ransac_fine
                    )
                    registration_methods.append(
                        ("RANSAC_Fine", transform_ransac_fine, eval_ransac_fine)
                    )
                    pprint(f"RANSAC Fine fitness: {eval_ransac_fine.fitness:.4f}")
                except Exception as e:
                    pprint(f"RANSAC Fine failed: {str(e)}")

        # Method 3: PCA alignment as fallback
        if not registration_methods or reg_fitness < 0.5:
            try:
                transform_pca = self.pca_alignment(model_moved, scene_cloud)
                eval_pca = o3d.pipelines.registration.evaluate_registration(
                    model_moved, scene_cloud, 0.05, transform_pca
                )

                def rotate_matrix_along_x_axis(matrix, angle):
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

                # Rotate the PCA transform to align with the scene
                z_p_camera = np.array([0, 0, 1])
                z_p_scene = transform_pca[:3, 2]
                dot = np.dot(z_p_camera, z_p_scene)
                if dot > 0:
                    transform_pca = rotate_matrix_along_x_axis(transform_pca, 180)
                registration_methods.append(("PCA", transform_pca, eval_pca))
                pprint(f"PCA fitness: {eval_pca.fitness:.4f}")
            except Exception as e:
                pprint(f"PCA failed: {str(e)}")

        # Select the best method based on fitness
        if registration_methods:
            best_method, best_transform, best_evaluation = max(
                registration_methods, key=lambda x: x[2].fitness
            )
            pprint(
                f"Selected best coarse registration method: {best_method} (fitness: {best_evaluation.fitness:.4f})"
            )
            initial_transform = best_transform
            evaluation = best_evaluation
        else:
            pprint("All coarse registration methods failed, using identity transform")
            initial_transform = np.eye(4)
            evaluation = o3d.pipelines.registration.RegistrationResult(
                transformation=initial_transform, fitness=0.0, inlier_rmse=0.0
            )

        # Combine transformations: first translation, then FGR transformation
        T_initial = initial_transform @ T_translation

        pprint(
            f"Coarse registration completed, time taken: {time.time() - start_time:.2f} seconds"
        )
        pprint(
            f"Coarse registration evaluation: fitness={evaluation.fitness:.4f}, RMSE={evaluation.inlier_rmse:.6f}"
        )
        pprint(
            f"Combined initial transformation matrix:\n{np.array_str(T_initial, precision=4, suppress_small=True)}"
        )

        # Visualize coarse registration result
        if self.visualize:
            self.visualize_registration(
                model_cloud,
                scene_cloud,
                T_initial,
                os.path.join(self.output_dir, "coarse_registration.png"),
            )

        # Fine registration (PointToPlane ICP)
        pprint("Performing ICP fine registration...")
        start_time = time.time()
        result = self.multi_stage_icp(model_cloud, scene_cloud, T_initial)
        pprint(f"ICP completed, total time: {time.time() - start_time:.2f} seconds")

        # Evaluate registration result with adaptive threshold
        scene_points = np.asarray(scene_cloud.points)
        model_points = np.asarray(model_cloud.points)
        scene_scale = np.linalg.norm(
            scene_points.max(axis=0) - scene_points.min(axis=0)
        )
        eval_threshold = max(0.02, scene_scale * 0.05)  # Adaptive evaluation threshold

        final_evaluation = o3d.pipelines.registration.evaluate_registration(
            model_cloud, scene_cloud, eval_threshold, result.transformation
        )
        pprint(
            f"Final registration evaluation (threshold={eval_threshold:.4f}): fitness={final_evaluation.fitness:.4f}, RMSE={final_evaluation.inlier_rmse:.6f}"
        )

        # Additional validation: check if the result makes geometric sense
        # Transform model and check overlap with scene
        model_transformed = copy.deepcopy(model_cloud)
        model_transformed.transform(result.transformation)

        # Check centroid distance after transformation
        scene_center = np.mean(scene_points, axis=0)
        model_center_transformed = np.mean(np.asarray(model_transformed.points), axis=0)
        centroid_distance = np.linalg.norm(scene_center - model_center_transformed)
        pprint(f"Post-registration centroid distance: {centroid_distance:.4f}")

        # If centroid distance is too large, the registration might have failed
        if centroid_distance > scene_scale * 0.3:
            pprint(
                "Warning: Large centroid distance suggests potential registration failure"
            )

        # Adjusted fitness thresholds based on point cloud characteristics
        fitness_threshold_low = max(
            0.15, 75.0 / len(scene_points)
        )  # Increased thresholds
        fitness_threshold_medium = max(0.35, 200.0 / len(scene_points))

        if final_evaluation.fitness < fitness_threshold_low:
            pprint(
                f"Low registration quality (fitness < {fitness_threshold_low:.3f}), results may be unreliable"
            )
            return None  # Registration failed, return None
        elif final_evaluation.fitness < fitness_threshold_medium:
            pprint(
                f"Medium registration quality (fitness < {fitness_threshold_medium:.3f}), please check results"
            )
            return None  # Medium quality, return None
        else:
            pprint("Good registration quality")

        return result.transformation

    def move_model_to_scene(self, model_cloud, scene_cloud):
        """
        Move model point cloud to scene point cloud centroid position
        """
        # Calculate scene point cloud centroid
        scene_points = np.asarray(scene_cloud.points)
        if len(scene_points) == 0:
            raise ValueError("Scene point cloud is empty, cannot calculate centroid")
        scene_center = np.mean(scene_points, axis=0)

        # Calculate model point cloud centroid
        model_points = np.asarray(model_cloud.points)
        if len(model_points) == 0:
            raise ValueError("Model point cloud is empty, cannot calculate centroid")
        model_center = np.mean(model_points, axis=0)

        pprint(f"Scene point cloud centroid: {scene_center}")
        pprint(f"Model point cloud centroid: {model_center}")

        # Calculate translation vector
        translation = scene_center - model_center
        pprint(f"Applying translation vector: {translation}")

        # Create translation matrix
        T = np.eye(4)
        T[:3, 3] = translation

        # Create model point cloud copy and apply translation
        model_moved = copy.deepcopy(model_cloud)
        model_moved.transform(T)

        # Visualize moved point clouds
        if self.visualize:
            self.visualize_moved_point_clouds(
                scene_cloud,
                model_moved,
                os.path.join(self.output_dir, "model_moved_to_scene.png"),
            )

        return model_moved, T

    def visualize_moved_point_clouds(self, scene_cloud, model_moved, filename):
        """
        Visualize moved point clouds
        """
        scene_temp = copy.deepcopy(scene_cloud)
        model_temp = copy.deepcopy(model_moved)
        # scene_temp.paint_uniform_color([1, 0, 0])  # Red: scene point cloud
        model_temp.paint_uniform_color([0, 0, 1])  # Blue: moved model

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Model Moved to Scene Centroid", width=1200, height=900
        )
        vis.add_geometry(scene_temp)
        vis.add_geometry(model_temp)

        # Set viewpoint for better visualization
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)

        vis.run()
        vis.capture_screen_image(filename)
        vis.destroy_window()

    def multi_stage_icp(self, source, target, init_transformation):
        """
        Multi-stage ICP registration strategy with adaptive parameters
        """
        # Calculate point cloud scale to adapt correspondence distances
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        source_scale = np.linalg.norm(
            source_points.max(axis=0) - source_points.min(axis=0)
        )
        target_scale = np.linalg.norm(
            target_points.max(axis=0) - target_points.min(axis=0)
        )
        avg_scale = (source_scale + target_scale) / 2

        pprint(
            f"Point cloud scale: source={source_scale:.4f}, target={target_scale:.4f}, avg={avg_scale:.4f}"
        )

        # More conservative correspondence distances to prevent overfitting
        loose_dist = max(0.08, avg_scale * 0.12)
        medium_dist = max(0.04, avg_scale * 0.06)
        strict_dist = max(0.015, avg_scale * 0.025)

        pprint(
            f"ICP correspondence distances: loose={loose_dist:.4f}, medium={medium_dist:.4f}, strict={strict_dist:.4f}"
        )

        # Stage 1: Loose parameters - Point to Point first for robustness
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=150
        )

        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance=loose_dist,
            init=init_transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=criteria,
        )

        evaluation1 = o3d.pipelines.registration.evaluate_registration(
            source, target, medium_dist, result.transformation
        )
        pprint(
            f"ICP Stage 1 (Point-to-Point): fitness={evaluation1.fitness:.4f}, RMSE={evaluation1.inlier_rmse:.6f}"
        )

        # Stage 2: Medium parameters - Continue with Point-to-Point for robustness
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=100
        )

        # First try continuing with Point-to-Point for better robustness
        result_p2p = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance=medium_dist,
            init=result.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=criteria,
        )

        evaluation2_p2p = o3d.pipelines.registration.evaluate_registration(
            source, target, medium_dist, result_p2p.transformation
        )
        pprint(
            f"ICP Stage 2 (Point-to-Point): fitness={evaluation2_p2p.fitness:.4f}, RMSE={evaluation2_p2p.inlier_rmse:.6f}"
        )

        # Try Point-to-Plane if we have good Point-to-Point result and both clouds have normals
        if (
            source.has_normals()
            and target.has_normals()
            and evaluation2_p2p.fitness > 0.3
        ):  # Only if Point-to-Point worked well
            pprint(
                "Attempting Point-to-Plane ICP with good Point-to-Point initialization..."
            )
            result_p2plane = o3d.pipelines.registration.registration_icp(
                source,
                target,
                max_correspondence_distance=medium_dist,
                init=result_p2p.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=criteria,
            )

            evaluation2_p2plane = o3d.pipelines.registration.evaluate_registration(
                source, target, medium_dist, result_p2plane.transformation
            )
            pprint(
                f"ICP Stage 2 (Point-to-Plane): fitness={evaluation2_p2plane.fitness:.4f}, RMSE={evaluation2_p2plane.inlier_rmse:.6f}"
            )

            # Use Point-to-Plane result if it's better, otherwise stick with Point-to-Point
            if (
                evaluation2_p2plane.fitness >= evaluation2_p2p.fitness * 0.8
            ):  # Allow small degradation for better geometry
                result = result_p2plane
                evaluation2 = evaluation2_p2plane
                pprint("Using Point-to-Plane result for better geometric accuracy")
            else:
                result = result_p2p
                evaluation2 = evaluation2_p2p
                pprint("Point-to-Plane didn't improve results, keeping Point-to-Point")
        else:
            result = result_p2p
            evaluation2 = evaluation2_p2p
            if not (source.has_normals() and target.has_normals()):
                pprint("Using Point-to-Point ICP (missing normals)")
            else:
                pprint(
                    "Using Point-to-Point ICP (Point-to-Point result not good enough for Point-to-Plane)"
                )

        # Stage 3: Fine parameters - Only if previous stage was successful
        if evaluation2.fitness > 0.2:  # Lower threshold to allow fine registration
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=80
            )

            # Use more conservative estimation method for fine registration
            final_estimation_method = (
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            if (
                source.has_normals()
                and target.has_normals()
                and evaluation2.fitness > 0.6
            ):  # Only use Point-to-Plane for very good alignments
                final_estimation_method = (
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
                pprint("Using Point-to-Plane for fine registration")
            else:
                pprint("Using Point-to-Point for fine registration")

            result_fine = o3d.pipelines.registration.registration_icp(
                source,
                target,
                max_correspondence_distance=strict_dist,
                init=result.transformation,
                estimation_method=final_estimation_method,
                criteria=criteria,
            )

            evaluation3 = o3d.pipelines.registration.evaluate_registration(
                source, target, strict_dist, result_fine.transformation
            )
            pprint(
                f"ICP Stage 3 (Fine): fitness={evaluation3.fitness:.4f}, RMSE={evaluation3.inlier_rmse:.6f}"
            )

            # Only use fine result if it doesn't significantly degrade fitness
            if (
                evaluation3.fitness >= evaluation2.fitness * 0.7
            ):  # Allow some degradation for better precision
                result = result_fine
                pprint("Using fine registration result")
            else:
                pprint("Fine registration degraded quality, keeping medium result")
        else:
            pprint("Skipping fine ICP stage due to poor intermediate results")

        return result

    def ransac_global_registration(self, source, target, voxel_size=0.01):
        """RANSAC global registration with improved parameters"""
        # Calculate FPFH features with adaptive parameters
        source_fpfh = self.compute_fpfh(source, voxel_size)
        target_fpfh = self.compute_fpfh(target, voxel_size)

        # RANSAC parameters - more conservative for better accuracy
        distance_threshold = voxel_size * 2.0  # Increased threshold for robustness

        # More iterations for better convergence
        result = (
            o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source,
                target,
                source_fpfh,
                target_fpfh,
                True,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                4,  # Increased minimum points for consensus
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.8  # Slightly more permissive edge length check
                    ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold
                    ),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.9999),
            )
        )

        return result.transformation

    def compute_fpfh(self, pcd, voxel_size):
        """
        Compute FPFH features with improved parameters
        """
        radius_normal = voxel_size * 3  # Increased for more stable normals
        if len(pcd.points) < 3:
            raise ValueError(
                "Insufficient points in point cloud, cannot calculate normals"
            )

        # Ensure point cloud has normals
        if not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50)
            )

        # Ensure normal direction consistency
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))

        # Use larger radius for more distinctive features
        radius_feature = voxel_size * 8  # Increased feature radius
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=150)
        )
        return fpfh

    def find_correspondences(self, source_fpfh, target_fpfh, source_pcd, target_pcd):
        """Find point correspondences using feature matching"""
        from scipy.spatial import cKDTree

        source_features = np.array(source_fpfh.data).T  # [N, 33]
        target_features = np.array(target_fpfh.data).T  # [M, 33]

        # Build KDTree for target features
        target_tree = cKDTree(target_features)
        s_to_t = {}
        for i in range(source_features.shape[0]):
            dist, idx = target_tree.query(source_features[i], k=1)
            s_to_t[i] = idx

        # Build KDTree for source features
        source_tree = cKDTree(source_features)
        t_to_s = {}
        for j in range(target_features.shape[0]):
            dist, idx = source_tree.query(target_features[j], k=1)
            t_to_s[j] = idx

        corr_source = []
        corr_target = []
        for i, j in s_to_t.items():
            if t_to_s.get(j, -1) == i:
                corr_source.append(i)
                corr_target.append(j)

        pprint(f"Found {len(corr_source)} mutual correspondences")
        return corr_source, corr_target

    def pca_alignment(self, source, target):
        """PCA-based initial alignment"""
        # Calculate principal directions
        source_pca = PCA(n_components=3).fit(np.asarray(source.points))
        target_pca = PCA(n_components=3).fit(np.asarray(target.points))

        # Create rotation matrix
        R = target_pca.components_.T @ source_pca.components_

        # Ensure right-handed coordinate system
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1

        # Calculate translation
        t = target_pca.mean_ - R @ source_pca.mean_

        # Build transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def visualize_registration(self, source, target, transformation, filename):
        """
        Visualize registration result and save image
        """
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.transform(transformation)

        # Set colors
        source_temp.paint_uniform_color([0, 1, 0])  # Green: source point cloud
        # target_temp.paint_uniform_color([1, 0, 0])  # Red: target point cloud

        # Create coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coord_frame.transform(transformation)

        # Visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Registration Result", width=1200, height=900)
        vis.add_geometry(source_temp)
        vis.add_geometry(target_temp)
        vis.add_geometry(coord_frame)

        # Add original coordinate frame reference
        orig_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        vis.add_geometry(orig_coord)

        # Set viewpoint for better visualization
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)

        vis.run()
        vis.capture_screen_image(filename)
        vis.destroy_window()
        pprint(f"Registration result saved to: {filename}")

    def save_pose_matrix(self, pose, filename):
        """
        Save pose matrix to file
        """
        np.savetxt(filename, pose)
        pprint(f"Pose matrix saved to: {filename}")

        # Also save in more readable format
        readable_path = filename.replace(".txt", "_readable.txt")
        with open(readable_path, "w") as f:
            f.write("Rotation matrix:\n")
            f.write(np.array2string(pose[:3, :3], precision=4, suppress_small=True))
            f.write("\n\nTranslation vector:\n")
            f.write(np.array2string(pose[:3, 3], precision=4, suppress_small=True))

            # Calculate Euler angles
            r = R.from_matrix(pose[:3, :3].copy())
            euler_angles = r.as_euler("xyz", degrees=True)
            f.write("\n\nEuler angles (degrees, xyz order):\n")
            f.write(np.array2string(euler_angles, precision=2))

            # Calculate quaternion
            quat = r.as_quat()
            f.write("\n\nQuaternion (w, x, y, z):\n")
            f.write(np.array2string(quat, precision=4))

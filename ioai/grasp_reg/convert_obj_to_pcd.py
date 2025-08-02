import os
import sys

# Try to import Open3D with fallback handling
try:
    import open3d as o3d
except ModuleNotFoundError as e:
    if "ml3d" in str(e):
        print("Warning: Open3D ML module not found. Trying to work around...")
        # Temporarily monkey-patch the problematic import
        import types
        ml_module = types.ModuleType('open3d.ml')
        sys.modules['open3d.ml'] = ml_module
        
        # Try importing again
        try:
            import open3d as o3d
        except Exception as e2:
            print(f"Failed to import Open3D even with workaround: {e2}")
            print("Please install Open3D properly or use a different version")
            sys.exit(1)
    else:
        raise e

def obj_to_pcd(obj_file: str, pcd_file: str):
    # Try reading as triangle mesh first
    mesh = o3d.io.read_triangle_mesh(obj_file)
    
    print(f"Mesh info for {obj_file}: vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")
    
    # If no triangles, try reading with a generic mesh reader or manual parsing
    if len(mesh.triangles) == 0 and len(mesh.vertices) == 0:
        print(f"Open3D couldn't read {obj_file} properly. Trying alternative approach...")
        
        # Try to manually parse the OBJ file for vertices
        vertices = []
        try:
            with open(obj_file, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            vertices.append([x, y, z])
            
            if vertices:
                print(f"Manually parsed {len(vertices)} vertices from {obj_file}")
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(vertices)
                o3d.io.write_point_cloud(pcd_file, pcd)
                print(f"Converted {obj_file} to {pcd_file} using manually parsed vertices")
                return True
            else:
                print(f"No vertices found in {obj_file}")
                return False
        except Exception as e:
            print(f"Error manually parsing {obj_file}: {e}")
            return False
    
    # Check if mesh has triangles
    if len(mesh.triangles) == 0:
        print(f"Warning: {obj_file} has no triangles.")
        
        # If no triangles but has vertices, use vertices directly as point cloud
        if len(mesh.vertices) > 0:
            print(f"Using {len(mesh.vertices)} vertices as point cloud.")
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            o3d.io.write_point_cloud(pcd_file, pcd)
            print(f"Converted {obj_file} to {pcd_file} using vertices")
            return True
        else:
            print(f"Error: {obj_file} has no vertices either. Skipping.")
            return False
    else:
        # Normal case: sample points from triangular mesh
        print(f"Sampling points from {len(mesh.triangles)} triangles.")
        pcd = mesh.sample_points_uniformly(number_of_points=100000)
        o3d.io.write_point_cloud(pcd_file, pcd)
        print(f"Converted {obj_file} to {pcd_file}")
        return True
    
if __name__ == "__main__":
    for file in os.listdir("models"):
        if file.endswith(".obj"):
            obj_file = os.path.join("models", file)
            pcd_file_name = file.replace(".obj", ".pcd")
            pcd_file = os.path.join("models", pcd_file_name)
            if os.path.exists(pcd_file):
                print(f"Skipping {obj_file}, already converted to {pcd_file}")
                continue
            try:
                success = obj_to_pcd(obj_file, pcd_file)
                if not success:
                    print(f"Failed to convert {obj_file}")
            except Exception as e:
                print(f"Error converting {obj_file}: {e}")
    print("All .obj files converted to .pcd files.")    
    print("Done.")
import open3d as o3d
import sys
import os

def visualize_obj(obj_path):
    """
    Visualize an OBJ file using Open3D
    """
    if not os.path.exists(obj_path):
        print(f"Error: File {obj_path} does not exist")
        return

    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(obj_path)
    
    # Compute normals for better visualization
    mesh.compute_vertex_normals()
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add the mesh to the visualization
    vis.add_geometry(mesh)
    
    # Set up the camera
    vis.get_render_option().mesh_show_back_face = True
    vis.get_render_option().light_on = True
    
    # Run the visualization
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_obj.py <path_to_obj_file>")
        sys.exit(1)
    
    obj_path = sys.argv[1]
    visualize_obj(obj_path)

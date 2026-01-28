"""
3D visualization utilities using Open3D.
"""

import os
import open3d as o3d


def visualize_mesh(
    mesh_path: str,
    window_name: str = "Reconstructed Model",
    width: int = 1280,
    height: int = 720,
) -> bool:
    """
    Visualize a 3D mesh file.
    
    Args:
        mesh_path: Path to the mesh file (.obj, .ply, etc.)
        window_name: Title for the visualization window
        width: Window width in pixels
        height: Window height in pixels
        
    Returns:
        True if successful, False if file not found or failed to load
    """
    if not os.path.exists(mesh_path):
        print(f"Error: File not found at {mesh_path}")
        print("Please ensure you have run the pipeline first.")
        return False

    print(f"Loading mesh from {mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(mesh_path, True)
    
    if not mesh.has_triangles():
        print("Error: The mesh seems to be empty or failed to load.")
        return False

    print("Visualizing mesh... (Close the window to exit)")
    o3d.visualization.draw_geometries(
        [mesh], 
        window_name=window_name,
        width=width,
        height=height,
        left=50,
        top=50,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=False
    )
    return True


def visualize_point_cloud(
    point_cloud: o3d.geometry.PointCloud,
    window_name: str = "Point Cloud",
    width: int = 1280,
    height: int = 720,
) -> None:
    """
    Visualize an Open3D point cloud.
    
    Args:
        point_cloud: Open3D PointCloud object
        window_name: Title for the visualization window
        width: Window width in pixels
        height: Window height in pixels
    """
    print("Visualizing point cloud... (Close the window to exit)")
    o3d.visualization.draw_geometries(
        [point_cloud], 
        window_name=window_name,
        width=width,
        height=height,
        left=50,
        top=50,
    )

"""
Mesh processing operations.
"""

import open3d as o3d


def load_rgbd_images(color_file: str, depth_file: str) -> o3d.geometry.RGBDImage:
    """
    Load and create an RGBD image from color and depth files.
    
    Args:
        color_file: Path to color image
        depth_file: Path to depth image
        
    Returns:
        Open3D RGBDImage object
    """
    color_raw = o3d.io.read_image(color_file)
    depth_raw = o3d.io.read_image(depth_file)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, 
        depth_raw, 
        depth_scale=1000.0, 
        depth_trunc=5.0, 
        convert_rgb_to_intensity=False
    )
    
    return rgbd_image


def surface_reconstruction(
    point_cloud: o3d.geometry.PointCloud
) -> o3d.geometry.TriangleMesh:
    """
    Reconstruct surface mesh from point cloud using Poisson reconstruction.
    
    Args:
        point_cloud: Input point cloud with normals
        
    Returns:
        Reconstructed triangle mesh
    """
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=25)
    )
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, 
        depth=10
    )
    return mesh

"""
Point cloud processing operations.
"""

import numpy as np
import open3d as o3d


def filter_outliers(
    point_cloud: o3d.geometry.PointCloud, 
    nb_neighbors: int = 20, 
    std_ratio: float = 2.0
) -> o3d.geometry.PointCloud:
    """
    Remove statistical outliers from a point cloud.
    
    Args:
        point_cloud: Input point cloud
        nb_neighbors: Number of neighbors to analyze for each point
        std_ratio: Standard deviation ratio threshold
        
    Returns:
        Filtered point cloud with outliers removed
    """
    cl, ind = point_cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, 
        std_ratio=std_ratio
    )
    inlier_cloud = point_cloud.select_by_index(ind)
    return inlier_cloud


def segment_point_cloud(
    point_cloud: o3d.geometry.PointCloud, 
    eps: float = 0.1, 
    min_points: int = 10
) -> list:
    """
    Segment point cloud using DBSCAN clustering.
    
    Args:
        point_cloud: Input point cloud
        eps: Maximum distance between two samples in a cluster
        min_points: Minimum number of points in a cluster
        
    Returns:
        List of segmented point cloud clusters
    """
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points))
    max_label = labels.max()
    segments = []
    for i in range(max_label + 1):
        segment = point_cloud.select_by_index(np.where(labels == i)[0])
        segments.append(segment)
    return segments

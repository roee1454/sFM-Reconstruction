
import os
import shutil
import numpy as np
import open3d as o3d
import pycolmap

from ..config import QUALITY_PROFILE, get_colmap_params
from .runner import run_command


def sparse_reconstruction(color_files: list, result_path: str):
    database_path = os.path.join(result_path, "database.db")
    image_dir = os.path.join(result_path, "images_temp")
    output_path = os.path.join(result_path, "sparse")

    os.makedirs(image_dir, exist_ok=True)

    for i, color_file in enumerate(color_files):
        image_name = f"image{i+1}.jpg"
        image_path = os.path.join(image_dir, image_name)
        shutil.copyfile(color_file, image_path)

    fe_params = get_colmap_params("feature_extraction")
    match_params = get_colmap_params("matching")
    map_params = get_colmap_params("incremental_mapping")

    print(f"\n{'='*60}")
    print(f"  COLMAP Sparse Reconstruction - Profile: {QUALITY_PROFILE}")
    print(f"{'='*60}")

    print("Extracting Features")
    if not os.path.exists(database_path):
        feature_extraction_options = pycolmap.FeatureExtractionOptions()
        feature_extraction_options.num_threads = fe_params.get("num_threads", 8)
        feature_extraction_options.max_image_size = fe_params.get("max_image_size", 2000)
        feature_extraction_options.use_gpu = False

        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.first_octave = fe_params.get("first_octave", 0)

        feature_extraction_options.sift = sift_options

        pycolmap.extract_features(
            database_path, 
            image_dir, 
            device=pycolmap.Device.cpu, 
            extraction_options=feature_extraction_options, 
            camera_model="PINHOLE"
        )
    else:
        print("Database file already exists. Skipping feature extraction.")

    print("Matching Features")
    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.num_threads = match_params.get("num_threads", 8)
    matching_options.use_gpu = False
    if match_params.get("guided_matching", False):
        matching_options.guided_matching = True
    pycolmap.match_sequential(
        database_path, 
        device=pycolmap.Device.cpu, 
        matching_options=matching_options
    )

    print("Performing Incremental Mapping")
    incremental_mapping_options = pycolmap.IncrementalPipelineOptions()
    incremental_mapping_options.num_threads = map_params.get("num_threads", 8)
    incremental_mapping_options.ba_global_frames_ratio = map_params.get("ba_global_frames_ratio", 1.2)
    incremental_mapping_options.multiple_models = map_params.get("multiple_models", False)
    
    os.makedirs(output_path, exist_ok=True)
    
    pycolmap.incremental_mapping(
        database_path, 
        image_dir, 
        output_path, 
        options=incremental_mapping_options
    )

    sparse_model_path = os.path.join(output_path, "0")
    if not os.path.exists(sparse_model_path):
        if os.path.exists(os.path.join(output_path, "cameras.bin")) or \
           os.path.exists(os.path.join(output_path, "cameras.txt")):
            sparse_model_path = output_path
        else:
            print("Error: Sparse reconstruction failed to produce a model.")
            return None

    try:
        sparse_model = pycolmap.Reconstruction(sparse_model_path)
        print(sparse_model.summary())
        return sparse_model
    except Exception as e:
        print(f"Error loading sparse model: {e}")
        return None


def convert_colmap_to_txt(sparse_model_path: str) -> bool:
    """
    Convert COLMAP binary model to TXT format.
    
    Args:
        sparse_model_path: Path to the sparse model directory
        
    Returns:
        True if successful, False otherwise
    """
    txt_output_dir = os.path.join(sparse_model_path, "sparse")
    os.makedirs(txt_output_dir, exist_ok=True)
    
    print(f"\n--- Converting COLMAP model to TXT at {txt_output_dir} ---")
    cmd = [
        "colmap", "model_converter",
        "--input_path", sparse_model_path,
        "--output_path", txt_output_dir,
        "--output_type", "TXT"
    ]
    return run_command(cmd)


def undistort_images(sparse_model_path: str, output_path: str, image_dir: str) -> bool:
    """
    Undistort images using COLMAP model.
    
    Args:
        sparse_model_path: Path to reconstruction (input)
        output_path: Path to write undistorted images (output)
        image_dir: Path to original images (input)
        
    Returns:
        True if successful, False otherwise
    """
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\n--- Undistorting Images to {output_path} ---")
    cmd = [
        "colmap", "image_undistorter",
        "--image_path", image_dir,
        "--input_path", sparse_model_path,
        "--output_path", output_path,
        "--output_type", "COLMAP",
        "--max_image_size", "2000",   # Match the extract resolution roughly
    ]
    return run_command(cmd)


def get_point_cloud_from_sparse_model(sparse_model) -> o3d.geometry.PointCloud:
    """
    Extract point cloud from COLMAP sparse reconstruction.
    
    Args:
        sparse_model: pycolmap.Reconstruction object
        
    Returns:
        Open3D PointCloud object
    """
    points = []
    colors = []
    for point in sparse_model.points3D.values():
        points.append(point.xyz)
        colors.append(point.color)
    
    points = np.array(points)
    colors = np.array(colors)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    return pcd

from ..config import settings, get_colmap_params
from ..pipeline import run_command
import open3d as o3d
import os
import pycolmap
import shutil

def sparse_reconstruction(color_files: list, result_path: str, output_callback=None):
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
    
    quality_profile = getattr(settings, "QUALITY_PROFILE", "BALANCED")
    colmap_device = getattr(settings, "COLMAP_DEVICE", pycolmap.Device.cpu)

    msg = f"\n{'='*60}\n  COLMAP Sparse Reconstruction - Profile: {quality_profile}\n{'='*60}\n"
    print(msg)
    if output_callback: output_callback(msg)

    msg = "Extracting Features\n"
    print(msg)
    if output_callback: output_callback(msg)
    
    if not os.path.exists(database_path):
        feature_extraction_options = pycolmap.FeatureExtractionOptions()
        feature_extraction_options.num_threads = fe_params.get("num_threads", 8)
        feature_extraction_options.max_image_size = fe_params.get("max_image_size", 2000)

        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.first_octave = fe_params.get("first_octave", 0)

        feature_extraction_options.sift = sift_options

        try:
            pycolmap.extract_features(
                database_path, 
                image_dir, 
                device=colmap_device, 
                extraction_options=feature_extraction_options, 
            )
        except Exception as e:
            if output_callback: output_callback(f"Error in extract_features: {e}\n")
            raise e
    else:
        msg = "Database file already exists. Skipping feature extraction.\n"
        print(msg)
        if output_callback: output_callback(msg)

    msg = "Matching Features\n"
    print(msg)
    if output_callback: output_callback(msg)
    
    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.num_threads = match_params.get("num_threads", 8)

    if match_params.get("guided_matching", False):
        matching_options.guided_matching = True
    
    pycolmap.match_sequential(
        database_path, 
        device=colmap_device, 
        matching_options=matching_options
    )

    msg = "Performing Incremental Mapping\n"
    print(msg)
    if output_callback: output_callback(msg)
    
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
            msg = "Error: Sparse reconstruction failed to produce a model.\n"
            print(msg)
            if output_callback: output_callback(msg)
            return None

    try:
        sparse_model = pycolmap.Reconstruction(sparse_model_path)
        summary = sparse_model.summary()
        print(summary)
        if output_callback: output_callback(str(summary) + "\n")
        return sparse_model
    except Exception as e:
        msg = f"Error loading sparse model: {e}\n"
        print(msg)
        if output_callback: output_callback(msg)
        return None


def convert_colmap_to_txt(sparse_model_path: str, output_callback=None) -> bool:
    txt_output_dir = os.path.join(sparse_model_path, "sparse")
    os.makedirs(txt_output_dir, exist_ok=True)
    
    msg = f"\n--- Converting COLMAP model to TXT at {txt_output_dir} ---\n"
    print(msg)
    if output_callback: output_callback(msg)
    
    cmd = [
        "colmap", "model_converter",
        "--input_path", sparse_model_path,
        "--output_path", txt_output_dir,
        "--output_type", "TXT"
    ]
    return run_command(cmd, output_callback=output_callback)


def undistort_images(sparse_model_path: str, output_path: str, image_dir: str, output_callback=None) -> bool:
    os.makedirs(output_path, exist_ok=True)
    
    msg = f"\n--- Undistorting Images to {output_path} ---\n"
    print(msg)
    if output_callback: output_callback(msg)
    
    cmd = [
        "colmap", "image_undistorter",
        "--image_path", image_dir,
        "--input_path", sparse_model_path,
        "--output_path", output_path,
        "--output_type", "COLMAP",
        "--max_image_size", "2000",
    ]
    return run_command(cmd, output_callback=output_callback)


def get_point_cloud_from_sparse_model(sparse_model) -> o3d.geometry.PointCloud:
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

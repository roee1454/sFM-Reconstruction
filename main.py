import pycolmap
import os
import numpy as np
import shutil
import open3d as o3d
import subprocess

def load_rgbd_images(color_file, depth_file):
    color_raw = o3d.io.read_image(color_file)
    depth_raw = o3d.io.read_image(depth_file)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
    
    return rgbd_image

def sparse_reconstruction(color_files, result_path):
    database_path = os.path.join(result_path, "database.db")
    image_dir = os.path.join(result_path, "images_temp")
    output_path = os.path.join(result_path, "sparse")

    os.makedirs(image_dir, exist_ok=True)

    for i, color_file in enumerate(color_files):
        image_name = f"image{i+1}.jpg"
        image_path = os.path.join(image_dir, image_name)
        shutil.copyfile(color_file, image_path)

    print("Extracting Features")
    if not os.path.exists(database_path):
        feature_extraction_options = pycolmap.FeatureExtractionOptions()
        feature_extraction_options.num_threads = 8
        feature_extraction_options.max_image_size = 2000
        feature_extraction_options.use_gpu = False

        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.first_octave = 0

        feature_extraction_options.sift = sift_options

        pycolmap.extract_features(database_path, image_dir, device=pycolmap.Device.cpu, extraction_options=feature_extraction_options, camera_model="PINHOLE")
    else:
        print("Database file already exists. Skipping feature extraction.")

    print("Matching Features")
    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.num_threads = 8
    matching_options.use_gpu = False
    pycolmap.match_sequential(database_path, device=pycolmap.Device.cpu, matching_options=matching_options)

    print("Performing Incremental Mapping")
    incremental_mapping_options = pycolmap.IncrementalPipelineOptions()
    incremental_mapping_options.num_threads = 8
    incremental_mapping_options.ba_global_frames_ratio = 1.2
    incremental_mapping_options.multiple_models = False
    
    os.makedirs(output_path, exist_ok=True)
    
    pycolmap.incremental_mapping(database_path, image_dir, output_path, options=incremental_mapping_options)

    sparse_model_path = os.path.join(output_path, "0")
    if not os.path.exists(sparse_model_path):
        if os.path.exists(os.path.join(output_path, "cameras.bin")) or os.path.exists(os.path.join(output_path, "cameras.txt")):
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

def get_point_cloud_from_sparse_model(sparse_model):
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

def filter_outliers(point_cloud, nb_neighbors=20, std_ratio=2.0):
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = point_cloud.select_by_index(ind)
    return inlier_cloud

def segment_point_cloud(point_cloud, eps=0.1, min_points=10):
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points))
    max_label = labels.max()
    segments = []
    for i in range(max_label + 1):
        segment = point_cloud.select_by_index(np.where(labels == i)[0])
        segments.append(segment)
    return segments

def surface_reconstruction(point_cloud):
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=25))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=10)
    return mesh

def run_command(cmd, cwd=None):
    """Executes a system command and prints its output."""
    print(f"Executing: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            print(f"Command failed with exit code {process.returncode}")
            return False
        return True
    except Exception as e:
        print(f"Error executing command: {e}")
        return False

def convert_colmap_to_txt(sparse_model_path):
    """Converts COLMAP binary model to TXT format expected by OpenMVS."""
    # OpenMVS InterfaceCOLMAP expects a folder containing a 'sparse' subfolder with TXT files
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

def run_openmvs_pipeline(sparse_model_path, image_dir, output_dir):
    mvs_bin = f"{os.path.dirname(os.path.abspath(__file__))}/thirdparty/OpenMVS/bin"
    
    print("\n--- Step 1: InterfaceCOLMAP ---")
    abs_sparse_model_path = os.path.abspath(sparse_model_path)
    abs_image_dir = os.path.abspath(image_dir)

    print("Absolute image path:", abs_image_dir)
    cmd = [
        os.path.join(mvs_bin, "InterfaceCOLMAP"),
        "-i", abs_sparse_model_path,
        "-o", "scene.mvs",
        "--image-folder", abs_image_dir
    ]
    if not run_command(cmd, cwd=output_dir): return False
    
    print("\n--- Step 2: DensifyPointCloud ---")
    cmd = [
        os.path.join(mvs_bin, "DensifyPointCloud"),
        "scene.mvs",
        "-o", "scene_dense.mvs",
        "--resolution-level", "2",
    ]
    if not run_command(cmd, cwd=output_dir): return False
    
    print("\n--- Step 3: ReconstructMesh ---")
    cmd = [
        os.path.join(mvs_bin, "ReconstructMesh"),
        "scene_dense.mvs",
        "-o", "scene_dense_mesh.mvs"
    ]
    if not run_command(cmd, cwd=output_dir): return False
    
    print("\n--- Step 4: RefineMesh ---")
    cmd = [
        os.path.join(mvs_bin, "RefineMesh"),
        "--resolution-level", "1",
        "scene_dense_mesh.mvs",
        "-o", "scene_dense_mesh_refine.mvs"
    ]
    if not run_command(cmd, cwd=output_dir): return False
    
    print("\n--- Step 5: TextureMesh ---")
    
    cmd = [
        os.path.join(mvs_bin, "TextureMesh"),
        "--export-type", "obj",
        "scene_dense_mesh_refine.mvs",
        "-o", "result.obj"
    ]
    if not run_command(cmd, cwd=output_dir): return False
    
    print(f"\nReconstruction complete! Result saved to: {os.path.join(output_dir, 'result.obj')}")
    return True

def main():
    print(f"Using pycolmap version: {pycolmap.__version__}")
    dataset_path = "test/images"
    result_path = "test/result"
    color_files = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    if not color_files:
        print(f"No images found in {dataset_path}")
        return
    
    sparse_dir = os.path.join(result_path, "sparse")

    if not os.path.exists(sparse_dir) or not os.listdir(sparse_dir):
        print("Estimating Camera Pose")
        sparse_model = sparse_reconstruction(color_files, result_path)
        sparse_model_path = os.path.join(sparse_dir, "0")
    else:
        print("Sparse reconstruction already exists. Loading...")
        sparse_model_path = os.path.join(sparse_dir, "0")
        if not os.path.exists(sparse_model_path):
            sparse_model_path = sparse_dir
        try:
            sparse_model = pycolmap.Reconstruction(sparse_model_path)
        except Exception as e:
            print(f"Failed to load existing reconstruction: {e}")
            print("Re-running reconstruction...")
            sparse_model = sparse_reconstruction(color_files, result_path)
            sparse_model_path = os.path.join(sparse_dir, "0")
    
    if sparse_model is None:
        print("Could not obtain a sparse reconstruction. Exiting.")
        return
    
    image_dir = os.path.join(result_path, "images_temp")
    
    if not convert_colmap_to_txt(sparse_model_path):
        print("Failed to convert COLMAP model to TXT.")
        return

    print("\n--- Starting OpenMVS Dense Reconstruction ---")
    success = run_openmvs_pipeline(sparse_model_path, image_dir, result_path)
    
    if success:
        print("Pipeline finished successfully.")
    else:
        print("Pipeline failed during OpenMVS steps.")

if __name__ == "__main__":
    main()
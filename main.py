import pycolmap
import os
import numpy as np
import shutil
import open3d as o3d

def load_rgbd_images(color_file, depth_file):
    color_raw = o3d.io.read_image(color_file)
    depth_raw = o3d.io.read_image(depth_file)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
    
    return rgbd_image

def sparse_reconstruction(color_files):
    database_path = "test/result/database.db"
    image_dir = "test/result/images_temp"
    output_path = "test/result/sparse"

    os.makedirs(image_dir, exist_ok=True)

    for i, color_file in enumerate(color_files):
        image_name = f"image{i+1}.png"
        image_path = os.path.join(image_dir, image_name)
        shutil.copyfile(color_file, image_path)

    print("Extracting Features")
    if not os.path.exists(database_path):
        feature_extraction_options = pycolmap.FeatureExtractionOptions()
        feature_extraction_options.num_threads = 8
        feature_extraction_options.max_image_size = 1600
        feature_extraction_options.use_gpu = False

        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.max_num_features = 4000
        sift_options.first_octave = 0

        pycolmap.extract_features(database_path, image_dir, device=pycolmap.Device.cpu, extraction_options=feature_extraction_options)
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

def refine_mesh(mesh):
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=1)
    mesh = mesh.filter_smooth_taubin(number_of_iterations=1)
    
    # # Remove degenerate triangles
    # mesh.remove_degenerate_triangles()
    
    # # Remove duplicated vertices
    # mesh.remove_duplicated_vertices()
    
    # # Remove non-manifold edges
    # mesh.remove_non_manifold_edges()
    return mesh

def main():
    print(f"Using pycolmap version: {pycolmap.__version__}")

    dataset_path = "test/images"
    color_files = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    if not color_files:
        print(f"No images found in {dataset_path}")
        return
    
    sparse_dir = "test/result/sparse"

    if not os.path.exists(sparse_dir) or not os.listdir(sparse_dir):
        print("Estimating Camera Pose")
        sparse_model = sparse_reconstruction(color_files)
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
            sparse_model = sparse_reconstruction(color_files)
    
    if sparse_model is None:
        print("Could not obtain a sparse reconstruction. Exiting.")
        return
    
    print("Extracting Point Cloud from Sparse Model")
    point_cloud = get_point_cloud_from_sparse_model(sparse_model)

    print("Filtering Outliers")
    filtered_pcd = filter_outliers(point_cloud)
    o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")

    print("Segmenting Point Cloud")
    segments = segment_point_cloud(filtered_pcd)

    print(f"Number of segments: {len(segments)}")

    meshes = []
    for i, segment in enumerate(segments):
        print(f"Reconstructing Segment {i+1}")
        mesh = surface_reconstruction(segment)
        filled_mesh = refine_mesh(mesh)
        meshes.append(filled_mesh)

    o3d.visualization.draw_geometries(meshes, window_name="Reconstructed Meshes")


if __name__ == "__main__":
    main()
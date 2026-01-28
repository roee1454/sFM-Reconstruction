#!/usr/bin/env python3
"""
Main entry point for the ObjectReconstruction pipeline.
"""

import os
import shutil
import pycolmap

from src.config.settings import DEFAULT_DATASET_PATH, DEFAULT_RESULT_PATH
from src.pipeline import (
    sparse_reconstruction,
    convert_colmap_to_txt,
    run_openmvs_pipeline,
    undistort_images
)

def main():
    print(f"Using pycolmap version: {pycolmap.__version__}")
    dataset_path = DEFAULT_DATASET_PATH
    result_path = DEFAULT_RESULT_PATH
    
    # Ensure dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    color_files = sorted([
        os.path.join(dataset_path, f) 
        for f in os.listdir(dataset_path) 
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    if not color_files:
        print(f"No images found in {dataset_path}")
        return
    
    sparse_dir = os.path.join(result_path, "sparse")

    # --- Sparse Reconstruction (COLMAP) ---
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
    
    # --- Model Conversion ---
    if not convert_colmap_to_txt(sparse_model_path):
        print("Failed to convert COLMAP model to TXT.")
        return

    # --- Undistort Images ---
    images_undistorted_dir = os.path.join(result_path, "images_undistorted")
    image_dir = os.path.join(result_path, "images_temp") # Original images
    
    if os.path.exists(images_undistorted_dir) and os.listdir(images_undistorted_dir):
        print("Undistorted images already exist. Skipping undistortion.")
    elif not undistort_images(sparse_model_path, images_undistorted_dir, image_dir):
        print("Failed to undistort images.")
        return

    # --- Dense Reconstruction (OpenMVS) ---
    print("\n--- Starting OpenMVS Dense Reconstruction ---")
    
    # undistort_images() creates 'sparse' (model) and 'images' (undistorted) folders
    # InterfaceCOLMAP expects the PARENT directory containing 'sparse' folder
    undistorted_model_path = images_undistorted_dir 
    undistorted_images_path = os.path.join(images_undistorted_dir, "images")
    
    success = run_openmvs_pipeline(undistorted_model_path, undistorted_images_path, result_path)
    
    if success:
        print("Pipeline finished successfully.")
        clean_redundant_files(result_path)
    else:
        print("Pipeline failed during OpenMVS steps.")


def clean_redundant_files(output_dir):
    """Clean up intermediate files to save space."""
    print(f"\n--- Cleaning up redundant files in {output_dir} ---")
    if not os.path.exists(output_dir):
        return

    keep_extensions = (".obj", ".mtl", ".jpg", ".png")
    
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)

        if filename.startswith("result") and filename.lower().endswith(keep_extensions):
            print(f"Keeping: {filename}")
            continue
            
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted file: {filename}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted directory: {filename}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == "__main__":
    main()
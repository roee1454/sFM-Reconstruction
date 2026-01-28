import os
import streamlit as st

from apps.streamlit.src.pipeline import (
    sparse_reconstruction,
    convert_colmap_to_txt,
    run_openmvs_pipeline,
    undistort_images
)

def run_reconstruction_pipeline(dataset_path, result_path, config, log_callback=None):
    color_files = sorted([
        os.path.join(dataset_path, f) 
        for f in os.listdir(dataset_path) 
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])
    
    if not color_files:
        st.error(f"No images found in {dataset_path}")
        return False

    sparse_dir = os.path.join(result_path, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    status_text = st.empty()
    progress_bar = st.progress(0)

    try:
        status_text.text("Step 1/4: Sparse Reconstruction (COLMAP)...")
        sparse_model_path = os.path.join(sparse_dir, "0")
        

        if not os.path.exists(sparse_model_path) or not os.listdir(sparse_model_path):
             sparse_model = sparse_reconstruction(color_files, result_path, output_callback=log_callback)
             if sparse_model is None:
                 st.error("Sparse reconstruction failed.")
                 return False
        else:
             status_text.text("Step 1/4: Sparse Reconstruction (Skipping, already exists)...")

        progress_bar.progress(25)
        
        status_text.text("Step 2/4: Converting Model...")
        if not convert_colmap_to_txt(sparse_model_path, output_callback=log_callback):
             st.error("Failed to convert COLMAP model to TXT.")
             return False
        progress_bar.progress(50)

        status_text.text("Step 3/4: Undistorting Images...")
        images_undistorted_dir = os.path.join(result_path, "images_undistorted")
        
        if not os.path.exists(images_undistorted_dir) or not os.listdir(images_undistorted_dir):
             if not undistort_images(sparse_model_path, images_undistorted_dir, dataset_path, output_callback=log_callback):
                 st.error("Failed to undistort images.")
                 return False
        
        progress_bar.progress(75)

        status_text.text("Step 4/4: Dense Reconstruction (OpenMVS)...")
        undistorted_model_path = images_undistorted_dir 
        undistorted_images_path = os.path.join(images_undistorted_dir, "images")
        
        success = run_openmvs_pipeline(undistorted_model_path, undistorted_images_path, result_path, output_callback=log_callback)
        
        if success:
            progress_bar.progress(100)
            status_text.text("Pipeline Finished Successfully!")
            return True
        else:
            st.error("Pipeline failed during OpenMVS steps.")
            return False

    except Exception as e:
        msg = f"An error occurred: {e}"
        st.error(msg)
        if log_callback: log_callback(msg + "\\n")
        return False

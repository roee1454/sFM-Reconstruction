import os

from ..config import (
    settings,
    OPENMVS_BIN_PATH,
    build_command_with_params,
    should_skip_refine_mesh,
)
from .runner import run_command


def run_openmvs_pipeline(sparse_model_path: str, image_dir: str, output_dir: str, output_callback=None) -> bool:
    mvs_bin = OPENMVS_BIN_PATH
    
    # Access settings dynamically
    quality_profile = getattr(settings, "QUALITY_PROFILE", "QUALITY")
    
    msg = f"\n{'='*60}\n  OpenMVS Pipeline - Profile: {quality_profile}\n{'='*60}\n"
    print(msg)
    if output_callback: output_callback(msg)
    
    scene_mvs_path = os.path.join(output_dir, "scene.mvs")
    if os.path.exists(scene_mvs_path):
        msg = "\n--- Step 1: InterfaceCOLMAP ---\nOutput file 'scene.mvs' already exists. Skipping step.\n"
        print(msg)
        if output_callback: output_callback(msg)
    else:
        msg = "\n--- Step 1: InterfaceCOLMAP ---\n"
        print(msg)
        if output_callback: output_callback(msg)
        
        abs_sparse_model_path = os.path.abspath(sparse_model_path)
        abs_image_dir = os.path.abspath(image_dir)

        print("Absolute image path:", abs_image_dir)
        cmd = [
            os.path.join(mvs_bin, "InterfaceCOLMAP"),
            "-i", abs_sparse_model_path,
            "-o", "scene.mvs",
            "--image-folder", abs_image_dir
        ]
        if not run_command(cmd, cwd=output_dir, output_callback=output_callback): 
            return False
    
    scene_dense_mvs_path = os.path.join(output_dir, "scene_dense.mvs")
    if os.path.exists(scene_dense_mvs_path):
        msg = "\n--- Step 2: DensifyPointCloud ---\nOutput file 'scene_dense.mvs' already exists. Skipping step.\n"
        print(msg)
        if output_callback: output_callback(msg)
    else:
        msg = "\n--- Step 2: DensifyPointCloud ---\n"
        print(msg)
        if output_callback: output_callback(msg)
        
        base_cmd = [
            os.path.join(mvs_bin, "DensifyPointCloud"),
            "scene.mvs",
            "-o", "scene_dense.mvs",
        ]
        cmd = build_command_with_params(base_cmd, "DensifyPointCloud")
        if not run_command(cmd, cwd=output_dir, output_callback=output_callback): 
            return False
    
    scene_dense_mesh_path = os.path.join(output_dir, "scene_dense_mesh.ply")
    if os.path.exists(scene_dense_mesh_path):
        msg = "\n--- Step 3: ReconstructMesh ---\nOutput file 'scene_dense_mesh.ply' already exists. Skipping step.\n"
        print(msg)
        if output_callback: output_callback(msg)
    else:
        msg = "\n--- Step 3: ReconstructMesh ---\n"
        print(msg)
        if output_callback: output_callback(msg)
        
        base_cmd = [
            os.path.join(mvs_bin, "ReconstructMesh"),
            "scene_dense.mvs",
            "-o", "scene_dense_mesh.ply"
        ]
        cmd = build_command_with_params(base_cmd, "ReconstructMesh")
        if not run_command(cmd, cwd=output_dir, output_callback=output_callback): 
            return False
    
    scene_dense_mesh_refine_path = os.path.join(output_dir, "scene_dense_mesh_refine.ply")
    mesh_for_texturing = "scene_dense_mesh.ply"
    
    if should_skip_refine_mesh():
        msg = "\n--- Step 4: RefineMesh ---\nSkipping RefineMesh (profile setting)\n"
        print(msg)
        if output_callback: output_callback(msg)
        mesh_for_texturing = "scene_dense_mesh.ply"
    elif os.path.exists(scene_dense_mesh_refine_path):
        msg = f"\n--- Step 4: RefineMesh ---\nOutput file '{scene_dense_mesh_refine_path}' already exists. Skipping step.\n"
        print(msg)
        if output_callback: output_callback(msg)
        mesh_for_texturing = "scene_dense_mesh_refine.ply"
    else:
        msg = "\n--- Step 4: RefineMesh ---\n"
        print(msg)
        if output_callback: output_callback(msg)
        
        base_cmd = [
            os.path.join(mvs_bin, "RefineMesh"),
            "scene_dense.mvs",
            "-m", "scene_dense_mesh.ply",
            "-o", "scene_dense_mesh_refine.ply"
        ]
        cmd = build_command_with_params(base_cmd, "RefineMesh")
        if not run_command(cmd, cwd=output_dir, output_callback=output_callback): 
            return False
        mesh_for_texturing = "scene_dense_mesh_refine.ply"
    
    result_obj_path = os.path.join(output_dir, "result.obj")
    if os.path.exists(result_obj_path):
        msg = f"\n--- Step 5: TextureMesh ---\nOutput file '{result_obj_path}' already exists. Skipping step.\n"
        print(msg)
        if output_callback: output_callback(msg)
    else:
        msg = "\n--- Step 5: TextureMesh ---\n"
        print(msg)
        if output_callback: output_callback(msg)
        
        base_cmd = [
            os.path.join(mvs_bin, "TextureMesh"),
            "--export-type", "obj",
            "scene_dense.mvs",
            "-m", mesh_for_texturing,
            "-o", "result.obj"
        ]
        cmd = build_command_with_params(base_cmd, "TextureMesh")
        if not run_command(cmd, cwd=output_dir, output_callback=output_callback): 
            return False
    
    msg = f"\nReconstruction complete! Result saved to: {os.path.join(output_dir, 'result.obj')}\n"
    print(msg)
    if output_callback: output_callback(msg)
    
    return True

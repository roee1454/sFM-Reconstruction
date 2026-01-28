import os

from ..config import (
    QUALITY_PROFILE,
    OPENMVS_BIN_PATH,
    build_command_with_params,
    should_skip_refine_mesh,
)
from .runner import run_command


def run_openmvs_pipeline(sparse_model_path: str, image_dir: str, output_dir: str) -> bool:
    mvs_bin = OPENMVS_BIN_PATH
    
    print(f"\n{'='*60}")
    print(f"  OpenMVS Pipeline - Profile: {QUALITY_PROFILE}")
    print(f"{'='*60}")
    
    scene_mvs_path = os.path.join(output_dir, "scene.mvs")
    if os.path.exists(scene_mvs_path):
        print("\n--- Step 1: InterfaceCOLMAP ---")
        print(f"Output file '{scene_mvs_path}' already exists. Skipping step.")
    else:
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
        if not run_command(cmd, cwd=output_dir): 
            return False
    
    scene_dense_mvs_path = os.path.join(output_dir, "scene_dense.mvs")
    if os.path.exists(scene_dense_mvs_path):
        print("\n--- Step 2: DensifyPointCloud ---")
        print(f"Output file '{scene_dense_mvs_path}' already exists. Skipping step.")
    else:
        print("\n--- Step 2: DensifyPointCloud ---")
        base_cmd = [
            os.path.join(mvs_bin, "DensifyPointCloud"),
            "scene.mvs",
            "-o", "scene_dense.mvs",
        ]
        cmd = build_command_with_params(base_cmd, "DensifyPointCloud")
        if not run_command(cmd, cwd=output_dir): 
            return False
    
    scene_dense_mesh_path = os.path.join(output_dir, "scene_dense_mesh.ply")
    if os.path.exists(scene_dense_mesh_path):
        print("\n--- Step 3: ReconstructMesh ---")
        print(f"Output file '{scene_dense_mesh_path}' already exists. Skipping step.")
    else:
        print("\n--- Step 3: ReconstructMesh ---")
        base_cmd = [
            os.path.join(mvs_bin, "ReconstructMesh"),
            "scene_dense.mvs",
            "-o", "scene_dense_mesh.ply"
        ]
        cmd = build_command_with_params(base_cmd, "ReconstructMesh")
        if not run_command(cmd, cwd=output_dir): 
            return False
    
    scene_dense_mesh_refine_path = os.path.join(output_dir, "scene_dense_mesh_refine.ply")
    mesh_for_texturing = "scene_dense_mesh.ply"
    
    if should_skip_refine_mesh():
        print("\n--- Step 4: RefineMesh ---")
        print("Skipping RefineMesh (profile setting)")
        mesh_for_texturing = "scene_dense_mesh.ply"
    elif os.path.exists(scene_dense_mesh_refine_path):
        print("\n--- Step 4: RefineMesh ---")
        print(f"Output file '{scene_dense_mesh_refine_path}' already exists. Skipping step.")
        mesh_for_texturing = "scene_dense_mesh_refine.ply"
    else:
        print("\n--- Step 4: RefineMesh ---")
        base_cmd = [
            os.path.join(mvs_bin, "RefineMesh"),
            "scene_dense.mvs",
            "-m", "scene_dense_mesh.ply",
            "-o", "scene_dense_mesh_refine.ply"
        ]
        cmd = build_command_with_params(base_cmd, "RefineMesh")
        if not run_command(cmd, cwd=output_dir): 
            return False
        mesh_for_texturing = "scene_dense_mesh_refine.ply"
    
    result_obj_path = os.path.join(output_dir, "result.obj")
    if os.path.exists(result_obj_path):
        print("\n--- Step 5: TextureMesh ---")
        print(f"Output file '{result_obj_path}' already exists. Skipping step.")
    else:
        print("\n--- Step 5: TextureMesh ---")
        base_cmd = [
            os.path.join(mvs_bin, "TextureMesh"),
            "--export-type", "obj",
            "scene_dense.mvs",
            "-m", mesh_for_texturing,
            "-o", "result.obj"
        ]
        cmd = build_command_with_params(base_cmd, "TextureMesh")
        if not run_command(cmd, cwd=output_dir): 
            return False
    
    print(f"\nReconstruction complete! Result saved to: {os.path.join(output_dir, 'result.obj')}")
    return True

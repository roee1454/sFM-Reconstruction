import streamlit as st
import plotly.graph_objects as go
import trimesh
import numpy as np
import os

def render_viewer(result_path):
    if not result_path:
        st.info("No results to show yet.")
        return 
    
    tab1, tab2 = st.tabs(["Point Cloud", "Mesh"])
    
    with tab1:
        _render_point_cloud(result_path)
        
    with tab2:
        _render_mesh(result_path)

def _render_point_cloud(result_path):
    ply_file = os.path.join(result_path, "scene_dense.ply")
    if not os.path.exists(ply_file):
        ply_file = os.path.join(result_path, "sparse/0/points3D.ply")
        
    if os.path.exists(ply_file):
        st.caption(f"Visualizing: {ply_file}")
        try:
            pcd = trimesh.load(ply_file)
            
            if isinstance(pcd, trimesh.points.PointCloud):
                points = pcd.vertices
                colors = pcd.colors if hasattr(pcd, 'colors') else None
                
                if len(points) > 50000:
                    indices = np.random.choice(len(points), 50000, replace=False)
                    points = points[indices]
                    if colors is not None:
                        colors = colors[indices]

                marker_dict = dict(size=2)
                if colors is not None:
                     marker_dict['color'] = colors[:, :3] 

                fig = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=points[:, 0],
                            y=points[:, 1],
                            z=points[:, 2],
                            mode='markers',
                            marker=marker_dict
                        )
                    ]
                )
                fig.update_layout(scene=dict(aspectmode='data'), height=500, margin=dict(r=0, l=0, b=0, t=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Loaded file is not a point cloud.")
        except Exception as e:
            st.error(f"Error loading point cloud: {e}")
    else:
        st.info("No point cloud found yet. Run the pipeline to generate one.")

def _render_mesh(result_path):
    obj_file = os.path.join(result_path, "scene_dense_mesh_textured.obj")
    
    if not os.path.exists(obj_file):
         obj_file = os.path.join(result_path, "scene_dense_mesh.ply")

    if os.path.exists(obj_file):
        st.caption(f"Visualizing: {obj_file}")
        try:
            mesh = trimesh.load(obj_file)
            
            if isinstance(mesh, trimesh.base.Trimesh):
                # Simplify mesh for web viewing if too heavy
                if len(mesh.faces) > 20000:
                    mesh = mesh.simplify_quadratic_decimation(20000)
                
                x, y, z = mesh.vertices.T
                i, j, k = mesh.faces.T
                
                fig = go.Figure(
                    data=[
                        go.Mesh3d(
                            x=x, y=y, z=z,
                            i=i, j=j, k=k,
                            color='lightpink',
                            opacity=0.50
                        )
                    ]
                )
                fig.update_layout(scene=dict(aspectmode='data'), height=500, margin=dict(r=0, l=0, b=0, t=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("File is not a mesh.")
        except Exception as e:
             st.error(f"Error loading mesh: {e}")
    else:
        st.info("No mesh found yet. Run the pipeline to generate one.")

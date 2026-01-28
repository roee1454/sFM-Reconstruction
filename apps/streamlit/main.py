import streamlit as st
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

streamlit_app_dir = os.path.dirname(os.path.abspath(__file__))
if streamlit_app_dir not in sys.path:
    sys.path.append(streamlit_app_dir)

from src.app.components.sidebar import render_sidebar
from src.app.components.upload import render_upload
from src.app.components.viewer import render_viewer
from src.app.components.logs import render_logs, add_log
from src.app.logic import run_reconstruction_pipeline

st.set_page_config(
    page_title="Object Reconstruction",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Object Reconstruction Pipeline")

def main():
    config = render_sidebar()
    
    result_path = os.path.join(streamlit_app_dir, "test", "result")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Dataset")
        dataset_path = render_upload()
        
        st.divider()
        
        st.header("Pipeline Control")
        run_clicked = st.button("Run Reconstruction", type="primary", use_container_width=True)
        
        st.divider()
        log_container = render_logs()

        if run_clicked:
             if dataset_path:
                 st.info(f"Starting pipeline with profile: {config['quality']} on device: {config['device']}")
                 
                 if 'logs' in st.session_state:
                     st.session_state.logs = []
                 
                 def live_log_callback(msg):
                     add_log(msg)
                     logs_all = "".join(st.session_state.logs)
                     log_container.text_area("Console Output", value=logs_all, height=300, key=f"log_view_{len(st.session_state.logs)}", disabled=True)

                 with st.spinner("Processing... This may take a while."):
                     success = run_reconstruction_pipeline(dataset_path, result_path, config, log_callback=live_log_callback)
                 
                 if success:
                     st.success("Pipeline completed!")
                     time.sleep(1)
                     st.rerun()
             else:
                 st.error("Please upload or select a dataset first.")

    with col2:
        st.header("3D Visualization")
        render_viewer(result_path)

if __name__ == "__main__":
    main()
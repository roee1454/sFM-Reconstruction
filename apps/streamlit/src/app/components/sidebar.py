import streamlit as st
from apps.streamlit.src.config.settings import QUALITY_PROFILE, COLMAP_DEVICE

def render_sidebar():
    with st.sidebar:
        st.title("Configuration")
        
        st.subheader("Reconstruction Quality")
        quality = st.select_slider(
            "Quality Profile",
            options=["SPEED", "BALANCED", "QUALITY"],
            value="BALANCED",
            help="Speed: Faster, lower detail. Quality: Slower, high detail."
        )
        
        st.subheader("Compute Device")
        device = st.radio(
            "Device",
            options=["AUTO", "CUDA", "CPU"],
            index=0,
            horizontal=True,
            help="Select the computing device for Colmap/OpenMVS."
        )
        
        return {
            "quality": quality,
            "device": device
        }

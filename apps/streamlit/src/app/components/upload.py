import streamlit as st
import os
import shutil

def render_upload():
    st.write("Upload images for reconstruction (JPG, PNG)")
    
    if "dataset_path" not in st.session_state:
        st.session_state.dataset_path = None

    uploaded_files = st.file_uploader(
        "Choose images", 
        accept_multiple_files=True,
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_files:
        upload_dir = os.path.join("test", "uploads", "current_session")
        
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
            
        progress_text = "Saving uploaded files..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, uploaded_file in enumerate(uploaded_files):
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            my_bar.progress((i + 1) / len(uploaded_files), text=progress_text)
            
        my_bar.empty()
        st.success(f"Uploaded {len(uploaded_files)} images.")
        st.session_state.dataset_path = upload_dir
        
    use_default = st.checkbox("Use default test/images folder", value=False)
    if use_default:
        default_path = os.path.join("test", "images")
        if os.path.exists(default_path):
             st.session_state.dataset_path = default_path
             st.info(f"Using default path: {default_path}")
    
    return st.session_state.dataset_path

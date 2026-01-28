import streamlit as st
import time

def render_logs()
    st.subheader("Pipeline Logs")
    
    if "logs" not in st.session_state:
        st.session_state.logs = []
        
    log_container = st.empty()    
    logs_text = "".join(st.session_state.logs)
    log_container = st.empty()
    log_container.text_area("Console Output", value=logs_text, height=300, key="log_view", disabled=True)
    
    return log_container

def add_log(message):
    if "logs" not in st.session_state:
        st.session_state.logs = []
    
    st.session_state.logs.append(message)

import streamlit as st
import os
from summarizer import summarize_pdf   # ✅ import the right function

st.set_page_config(page_title="Legal AI Summarizer", layout="wide")
st.title("📑 Legal AI Summarizer for Indian Cyber Law")

# File uploader
uploaded_file = st.file_uploader("Upload a case PDF", type=["pdf"])

if uploaded_file:
    temp_path = "temp_case.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("✅ File uploaded successfully!")
    
    if st.button("Generate FIRAC Summary"):
        with st.spinner("Processing case..."):
            try:
                summary = summarize_pdf(temp_path)   # ✅ no retriever needed
                st.markdown("### 📑 FIRAC Summary")
                st.write(summary)
            finally:
                os.remove(temp_path)
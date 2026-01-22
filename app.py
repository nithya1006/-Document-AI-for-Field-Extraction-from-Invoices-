"""
Streamlit Web App for Invoice Field Extraction

Run with: streamlit run app.py
"""

import streamlit as st
import sys
import os
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from executable import InvoiceExtractor
from utils.visualizer import Visualizer

# Page config
st.set_page_config(
    page_title="Invoice Field Extractor",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'extractor' not in st.session_state:
    with st.spinner("Initializing extraction pipeline..."):
        st.session_state.extractor = InvoiceExtractor()
        st.session_state.visualizer = Visualizer()

# Header
st.markdown('<p class="main-header">📄 Invoice Field Extraction System</p>', 
           unsafe_allow_html=True)

st.markdown("""
This system extracts key fields from invoice PDFs using:
- **OCR**: PaddleOCR for text extraction
- **NLP**: Fuzzy matching and pattern recognition
- **CV**: Signature and stamp detection
""")

# Sidebar
st.sidebar.title("⚙️ Configuration")

ocr_engine = st.sidebar.selectbox(
    "OCR Engine",
    ["paddleocr", "easyocr", "tesseract"],
    index=0
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05
)

show_visualization = st.sidebar.checkbox("Show Visualization", value=True)

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "📊 Batch Processing", "📈 Analytics"])

# Tab 1: Single Document Upload
with tab1:
    st.header("Upload Invoice PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a tractor loan quotation or invoice PDF"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📄 Document Preview")
            
            # Convert first page to image for preview
            from utils.pdf_processor import PDFProcessor
            processor = PDFProcessor()
            images = processor.pdf_to_images(tmp_path)
            
            if images:
                st.image(images[0], caption="First Page", use_column_width=True)
        
        with col2:
            st.subheader("🔍 Extraction Results")
            
            if st.button("🚀 Extract Fields", type="primary"):
                with st.spinner("Processing document..."):
                    # Extract fields
                    result = st.session_state.extractor.process_document(tmp_path)
                    
                    if 'error' in result:
                        st.error(f"❌ Extraction failed: {result['error']}")
                    else:
                        # Display results
                        st.success("✅ Extraction completed!")
                        
                        # Metrics
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        
                        with metrics_col1:
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                        
                        with metrics_col2:
                            st.metric("Processing Time", f"{result['processing_time_sec']:.2f}s")
                        
                        with metrics_col3:
                            st.metric("Cost", f"${result['cost_estimate_usd']:.4f}")
                        
                        # Extracted fields
                        st.subheader("📋 Extracted Fields")
                        
                        fields = result['fields']
                        
                        st.markdown("**Dealer Information**")
                        st.write(f"**Name:** {fields['dealer_name']}")
                        
                        st.markdown("**Asset Details**")
                        st.write(f"**Model:** {fields['model_name']}")
                        st.write(f"**Horse Power:** {fields['horse_power']} HP")
                        st.write(f"**Asset Cost:** ₹{fields['asset_cost']:,.2f}")
                        
                        st.markdown("**Document Verification**")
                        st.write(f"**Signature Present:** {'✅ Yes' if fields['signature']['present'] else '❌ No'}")
                        if fields['signature']['present']:
                            st.write(f"  Confidence: {fields['signature']['confidence']:.1%}")
                        
                        st.write(f"**Stamp Present:** {'✅ Yes' if fields['stamp']['present'] else '❌ No'}")
                        if fields['stamp']['present']:
                            st.write(f"  Confidence: {fields['stamp']['confidence']:.1%}")
                        
                        # Visualization
                        if show_visualization and images:
                            st.subheader("🎨 Visualization")
                            vis_image = st.session_state.visualizer.visualize_extraction_result(
                                images[0], result
                            )
                            st.image(vis_image, caption="Detected Fields", use_column_width=True)
                        
                        # JSON output
                        with st.expander("📝 View JSON Output"):
                            st.json(result)
                        
                        # Download button
                        json_str = json.dumps(result, indent=2)
                        st.download_button(
                            label="💾 Download JSON",
                            data=json_str,
                            file_name=f"{result['doc_id']}_result.json",
                            mime="application/json"
                        )
        
        # Clean up
        os.unlink(tmp_path)

# Tab 2: Batch Processing
with tab2:
    st.header("Batch Processing")
    st.write("Upload multiple PDF files for batch processing")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} files uploaded")
        
        if st.button("🚀 Process Batch", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # Save temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Process
                result = st.session_state.extractor.process_document(tmp_path)
                results.append(result)
                
                # Clean up
                os.unlink(tmp_path)
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ Batch processing completed!")
            
            # Display summary
            st.subheader("📊 Batch Summary")
            
            successful = sum(1 for r in results if 'error' not in r)
            failed = len(results) - successful
            avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results)
            total_time = sum(r['processing_time_sec'] for r in results)
            total_cost = sum(r.get('cost_estimate_usd', 0) for r in results)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric("Total", len(results))
            col2.metric("Successful", successful)
            col3.metric("Failed", failed)
            col4.metric("Avg Confidence", f"{avg_confidence:.1%}")
            col5.metric("Total Cost", f"${total_cost:.4f}")
            
            # Results table
            st.subheader("📋 Detailed Results")
            
            results_df = []
            for r in results:
                if 'error' not in r:
                    results_df.append({
                        'Document': r['doc_id'],
                        'Dealer': r['fields']['dealer_name'],
                        'Model': r['fields']['model_name'],
                        'HP': r['fields']['horse_power'],
                        'Cost': r['fields']['asset_cost'],
                        'Confidence': f"{r['confidence']:.1%}",
                        'Time (s)': r['processing_time_sec']
                    })
                else:
                    results_df.append({
                        'Document': r['doc_id'],
                        'Dealer': 'Error',
                        'Model': 'Error',
                        'HP': 0,
                        'Cost': 0,
                        'Confidence': '0%',
                        'Time (s)': r['processing_time_sec']
                    })
            
            import pandas as pd
            df = pd.DataFrame(results_df)
            st.dataframe(df, use_container_width=True)
            
            # Download batch results
            json_str = json.dumps(results, indent=2)
            st.download_button(
                label="💾 Download Batch Results (JSON)",
                data=json_str,
                file_name="batch_results.json",
                mime="application/json"
            )

# Tab 3: Analytics
with tab3:
    st.header("📈 Analytics & Insights")
    st.write("View system performance and statistics")
    
    # Load sample data if available
    sample_output_dir = Path("sample_output")
    
    if sample_output_dir.exists():
        json_files = list(sample_output_dir.glob("*.json"))
        
        if json_files:
            results = []
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    results.append(json.load(f))
            
            # Create visualizations
            st.session_state.visualizer.create_eda_report(results)
            
            # Display visualizations
            viz_dir = Path("visualizations")
            
            if viz_dir.exists():
                st.subheader("📊 Performance Metrics")
                
                img_path = viz_dir / "batch_metrics.png"
                if img_path.exists():
                    st.image(str(img_path), caption="Batch Metrics", use_column_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    img_path = viz_dir / "confidence_vs_time.png"
                    if img_path.exists():
                        st.image(str(img_path), caption="Confidence vs Time", 
                                use_column_width=True)
                
                with col2:
                    img_path = viz_dir / "error_analysis.png"
                    if img_path.exists():
                        st.image(str(img_path), caption="Error Analysis", 
                                use_column_width=True)
                
                # Summary stats
                stats_path = viz_dir / "summary_stats.json"
                if stats_path.exists():
                    with open(stats_path, 'r') as f:
                        stats = json.load(f)
                    
                    st.subheader("📈 Summary Statistics")
                    
                    for key, value in stats.items():
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.info("No processed documents found. Process some documents first!")
    else:
        st.info("No analytics data available yet. Process some documents to see analytics!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built for IDFC Hackathon - Convolve 4.0</p>
    <p>Powered by PaddleOCR, OpenCV, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
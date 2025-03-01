import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import os
from pdf2image import convert_from_path
from vision_model import VisionProcessor
import logging
from PyPDF2 import PdfReader
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set page config
st.set_page_config(
    page_title="Magnifier Text Extractor",
    page_icon="🔍",
    layout="wide"
)

def validate_pdf(file):
    """Validate uploaded PDF file"""
    if file is None:
        return False
    
    if not file.name.lower().endswith('.pdf'):
        st.error("Please upload a PDF file")
        return False
    
    if file.size > 50 * 1024 * 1024:  # 50MB limit
        st.error("File size too large. Please upload a file smaller than 50MB")
        return False
    
    return True

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory and return path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def detect_magnifier(page_path: str, vision_processor: VisionProcessor) -> bool:
    """Detect if a page contains a magnifier symbol"""
    try:
        detect_result = vision_processor.detect_magnifier_gemini(page_path)
        
        # Handle both boolean and dictionary return types
        if isinstance(detect_result, dict):
            return detect_result.get("success", False) and detect_result.get("found", False)
        else:
            return bool(detect_result)
    except Exception as e:
        st.error(f"Error in magnifier detection: {str(e)}")
        return False

def extract_magnifier_text(page_path: str, page_id: int, vision_processor: VisionProcessor) -> list:
    """Extract text from magnifier items on a page"""
    try:
        extracted_data = vision_processor.extract_text(page_path)
        
        if extracted_data and hasattr(extracted_data, 'magnifier_items'):
            # Create a list to store all items from this page
            items = []
            for item in extracted_data.magnifier_items:
                items.append({
                    "page_id": page_id,
                    "cycle_id": item.cycle_id,
                    "page_number": item.page_number,
                    "text": item.text_after_symbol,
                    "has_magnifier": True
                })
            return items
        else:
            # Return empty list if no items found
            return []
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return []

def process_page(page_path: str, page_id: int, vision_processor: VisionProcessor) -> list:
    """Process a single page and return results"""
    try:
        # Check for magnifier
        has_magnifier = detect_magnifier(page_path, vision_processor)
        
        if has_magnifier:
            # Extract text and metadata
            page_results = extract_magnifier_text(page_path, page_id, vision_processor)
            
            return page_results
        else:
            # Return empty list for consistency
            return []
            
    except Exception as e:
        st.error(f"Error processing page {page_id}: {str(e)}")
        # Return empty list for consistency
        return []

def main():
    st.title("Magnifier Text Extractor 🔍")
    
    # Add user guide in an expander
    with st.expander("📚 User Guide - How to Use This App", expanded=False):
        st.markdown("""
        ### Welcome to the Magnifier Text Extractor!
        
        This application helps you extract text that follows magnifier symbols (🔍) in PDF documents. Here's how to use it:
        
        #### Step 1: Upload a PDF File
        - Click the **Browse files** button in the file uploader
        - Select a PDF document from your computer
        - The app will display the total number of pages in your document
        
        #### Step 2: Preview Your PDF (Optional)
        - Expand the **PDF Preview** section to see individual pages
        - Use the page number input to navigate through the document
        - This helps you identify which pages contain magnifier symbols
        
        #### Step 3: Select Page Range
        - Choose which pages to process using the **Start Page** and **End Page** inputs
        - Processing fewer pages is faster, so select only the relevant sections if you know where the magnifiers are
        - The app will show you how many pages will be processed
        
        #### Step 4: Process the PDF
        - Click the **Process PDF** button to start extraction
        - Expand the progress section to see real-time processing
        - For each page, the app will:
          - Show the page image
          - Detect if there are any magnifier symbols
          - Extract text that follows the magnifiers
          - Display the results
        
        #### Step 5: Review and Download Results
        - After processing completes, expand the **View All Results** section
        - Review the extracted text in the table
        - Click the **Download Results as CSV** button to save the data
        
        #### Tips for Best Results:
        - Make sure your PDF has clear, visible magnifier symbols
        - For large documents, process a few pages first to test the extraction quality
        
        """)
    
    # Create two columns for upload and preview
    upload_col, preview_col = st.columns([1, 2])
    
    # File upload in the first column
    with upload_col:
        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        
        if uploaded_file and validate_pdf(uploaded_file):
            # Get filename without extension
            filename = Path(uploaded_file.name).stem
            
            # Create directory structure
            base_dir = Path("data") / filename
            image_dir = base_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Save uploaded file
                pdf_path = save_uploaded_file(uploaded_file)
                
                # Get total pages using PdfReader
                pdf_reader = PdfReader(pdf_path)
                total_pages = len(pdf_reader.pages)
                
                # Display PDF information
                st.caption("PDF Information")
                st.write(f"**Filename:** {filename}")
                st.write(f"**Total pages:** {total_pages}")
            
            except Exception as e:
                st.error(f"Error loading PDF: {str(e)}")
                return
    
    # PDF Preview in the second column
    with preview_col:
        if uploaded_file and validate_pdf(uploaded_file):
            st.subheader("PDF Preview")
            with st.expander("expand to see page"):
                page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                    
                # Display selected page
                try:
                    preview_image_path = image_dir / f"page_{page_number}.png"
                    if not preview_image_path.exists():
                        images = convert_from_path(
                            pdf_path,
                            first_page=page_number,
                            last_page=page_number,
                            dpi=200
                        )
                        if images:
                            images[0].save(str(preview_image_path))
                    
                    if preview_image_path.exists():
                        st.image(preview_image_path, width=500)
                except Exception as e:
                    st.error(f"Error displaying page {page_number}: {str(e)}")
        else:
            # If no file is uploaded, show instructions
            st.subheader("PDF Preview")
            st.info("Upload a PDF file to see preview here")
            st.write("The preview will show the selected page from your PDF document.")
    
    # Full-width processing section (outside of columns)
    if uploaded_file and validate_pdf(uploaded_file):
        st.markdown("---")  # Add a separator
        st.subheader("Processing")
        
        # Add page range selection
        st.write("### Select Page Range")
        range_col1, range_col2 = st.columns(2)
        with range_col1:
            start_page = st.number_input("Start Page", min_value=1, max_value=total_pages, value=1)
        with range_col2:
            end_page = st.number_input("End Page", min_value=start_page, max_value=total_pages, value=total_pages)

        # Show selected range
        st.write(f"Selected range: Pages {start_page} to {end_page} (Total: {end_page - start_page + 1} pages)")

        if st.button("Process PDF"):
            # Create progress indicators at the top
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create columns for image and results below the progress indicators
            img_col, result_col = st.columns([1, 1])
            
            # Create placeholder for current image in the first column
            with img_col:
                current_image_placeholder = st.empty()
            
            # Create placeholder for results in the second column
            with result_col:
                result_placeholder = st.empty()
            
            # Initialize vision processor
            vision_processor = VisionProcessor()
            
            # Placeholder for processing results
            results = []
            
            # Process only the selected page range
            total_selected_pages = end_page - start_page + 1
            
            for i in range(start_page - 1, end_page):
                # Update progress based on selected range
                progress = (i - start_page + 2) / total_selected_pages
                status_text.text(f"Processing page {i + 1}/{total_pages} ({i - start_page + 2}/{total_selected_pages})")
                progress_bar.progress(progress)
                
                # Convert page to image if needed
                image_path = image_dir / f"page_{i+1}.png"
                
                try:
                    if not image_path.exists():
                        images = convert_from_path(
                            pdf_path,
                            first_page=i+1,
                            last_page=i+1,
                            dpi=200
                        )
                        if images:
                            images[0].save(str(image_path))
                    
                    # Display current image
                    with img_col:
                        current_image_placeholder.image(
                            image_path, 
                            caption=f"Processing page {i + 1}", 
                            width=400
                        )
                    
                    # First detect if page has magnifier
                    has_magnifier = detect_magnifier(str(image_path), vision_processor)
                    
                    # Update results display with detection result
                    with result_col:
                        # Create a fresh container for the current page
                        with result_placeholder.container():
                            st.write(f"### Processing page {i + 1} of {total_pages}")
                            
                            if has_magnifier:
                                st.write(f"✅ Magnifier detected on page {i + 1}! Extracting text...")
                                
                                # Extract text if magnifier is found
                                page_results = extract_magnifier_text(str(image_path), i + 1, vision_processor)
                                
                                if page_results:
                                    st.write(f"{len(page_results)} magnifiers found")
                                    st.dataframe(page_results)
                                    results.extend(page_results)
                                else:
                                    st.write("No text could be extracted")
                            else:
                                st.write(f"❌ No magnifiers found on page {i + 1}")
                    
                    # Small delay for UI updates
                    time.sleep(1)
                    current_image_placeholder.empty()
                    result_placeholder.empty()


                except Exception as e:
                    st.error(f"Error processing page {i + 1}: {str(e)}")
                    continue
                
            # Display final results
            st.success(f"Processing complete! Found {len(results)} magnifiers.")
                
            # Display all results in a collapsible section
            st.subheader("View Results")
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Download button for CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"{filename}_magnifier_results.csv",
                    mime="text/csv",
                    help="Download the complete results as a CSV file"
                )
            
            # Clean up temporary PDF file when done
            if 'pdf_path' in locals() and os.path.exists(pdf_path):
                os.unlink(pdf_path)

if __name__ == "__main__":
    main() 
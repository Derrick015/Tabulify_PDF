import streamlit as st
import os
import tempfile
import time
import logging
import asyncio
import pymupdf
from dotenv import load_dotenv
from openai import AsyncOpenAI
import zipfile
import io
import pandas as pd
import itertools

from src.pdf_extraction import (
    get_page_pixel_data,
    get_page_text_thread,
    get_validated_table_info,
    process_tables_to_df,
    write_output_final,
    write_output_to_csv,
    remove_duplicate_dfs
)

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Configure logging
log_file = os.path.join("logs", "app.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will continue to show logs in console
    ]
)

# Log the start of the application
logging.info("Starting Tabulify PDF application")

# Page configuration
st.set_page_config(
    page_title="Tabulify PDF",
    page_icon="logo.png",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'output_final' not in st.session_state:
    st.session_state.output_final = []
if 'file_name' not in st.session_state:
    st.session_state.file_name = "output_file"
if 'custom_pages_last' not in st.session_state:
    st.session_state.custom_pages_last = ""

# Load environment variables
load_dotenv()
open_api_key = os.getenv('OPENAI_API_KEY')
if not open_api_key:
    st.error("OPENAI_API_KEY is not set. Please check your .env file.")
    st.stop()

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=open_api_key)

# App header with logo
col1, col2 = st.columns([1, 9])
with col1:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=80)
with col2:
    st.markdown("<h1 style='margin-top: 10px;'>Tabulify PDF</h1>", unsafe_allow_html=True)

# AI Model Information Section
# st.info("🤖 **This application uses OpenAI's GPT-4o Mini model to intelligently identify and extract tables from your PDFs. The AI analyzes document structure, recognizes table patterns, and converts them into structured data formats.")

st.markdown("<p style='font-size: 14px;'>Tabulify PDF uses AI to intelligently detect and extract tables from your PDFs according to your instructions. It analyses the document's structure, identifies table patterns, and converts them into structured data that can be exported in Excel or CSV format.</p>", unsafe_allow_html=True)
# Sidebar for options
with st.sidebar:
    # st.header("Settings")
    
    # Ensure unified output directory with the CLI version
    if not os.path.exists("output_files"):
        os.makedirs("output_files")
        
    # Output file name
    file_name = st.text_input("File name", value=st.session_state.file_name)
    st.session_state.file_name = file_name
    
    # Use text_area instead of text_input for more space
    user_text = st.text_area(
        "Instructions for AI", 
        value="Extract all data from the table(s)",
        height=200  # Make the box much taller
    )
    
    st.markdown("---")  # Add some space with a horizontal line
    
    # Add checkbox for table in image detection
    table_in_image = st.checkbox("Image & Inference Mode", value=True, 
                                help="Enable this mode for: (1) Extracting tables from images within PDFs, (2) Adding creative interpretations like additional columns or values based on user instructions. Note: This mode bypasses text validation for more flexible results.")
    
    # Add checkbox to include table and page information in output
    add_in_table_and_page_information = st.checkbox("Add table and page information", value=False, 
                                 help="Enable this if you want to add table name, position and page number to the table")

    model = "gpt-5-mini"
    vision_model = "gpt-5-mini"

    # st.markdown("---")  # Add some space with a horizontal line
    
    # Model selection dropdown for AI processing (hashed out - defaulted to gpt-5-mini)
    # model = st.selectbox(
    #     "Select table extraction model",
    #     options=["gpt-5", "gpt-5-mini"],
    #     index=1  # Default to gpt-5-mini as recommended option
    # )

    # vision_model = st.selectbox(
    #     "Select table identification model",
    #     options=["gpt-5", "gpt-5-mini"],
    #     index=1  # Default to gpt-5-mini as recommended option
    # )

    
    # Display information about available models to help users make appropriate selection
    # st.markdown("""
    #     <div style="font-size:0.8em; color:gray;">
    #     <strong>Model information:</strong><br>
    #     • <strong>gpt-5</strong>: Balanced performance, recommended for most tables<br>
    #     • <strong>gpt-5-mini</strong>: Faster, lower cost, but may be less accurate for complex tables<br>
    #
    #     </div>
    # """, unsafe_allow_html=True)

    # Add horizontal line for visual separation of sections
    st.markdown("---")
    
    # Output format selection section
    st.subheader("Output Format")
    file_format = st.selectbox(
        "Select file format:",
        options=["Excel (.xlsx)", "CSV (.csv)"],
        index=0  # Default to Excel format
    )
    
    # Input validation logic - ensure sensible defaults if user inputs are empty
    if not file_name.strip():
        file_name = "output_file"
        st.session_state.file_name = file_name
        st.warning("Using default filename 'output_file' as none was provided.")
    
    if not user_text.strip():
        user_text = "Extract all data from the table(s)"
        st.warning("Using default instructions 'Extract all data from the table(s)' as none were provided.")

# File upload section
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)

# Main processing logic
if uploaded_file:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name
    
    # Load PDF document and validate it can be opened
    try:
        doc = pymupdf.open(pdf_path)
        total_pages = doc.page_count
        
        st.success(f"Successfully loaded PDF with {total_pages} pages.")
        
        # Page range selection section - allows users to choose which pages to process
        st.subheader("Page Range Selection")
        range_option = st.radio("Select pages to process:", 
                               ["All pages", "Specific range", "Custom pages"])
        
        if range_option == "All pages":
            # Process the entire document
            page_indices = list(range(total_pages))
            st.info(f"Processing all {total_pages} pages")
            
        elif range_option == "Specific range":
            # Allow selection of a continuous range of pages
            col1, col2 = st.columns(2)
            
            # Initialize end_page in session state if it doesn't exist
            # This preserves the value between reruns of the Streamlit app
            if 'end_page' not in st.session_state:
                st.session_state.end_page = min(5, total_pages)  # Default to page 5 or max
                
            with col1:
                # Start page selection with input validation
                start_page = st.number_input("Start page", min_value=1, max_value=total_pages, value=1, key="start_page")
            
            with col2:
                # End page selection with dynamic minimum value based on start page
                # Ensures end page is always >= start page
                if 'end_page' not in st.session_state:
                    st.session_state.end_page = min(5, total_pages)  # Default to page 5 or max
                elif st.session_state.end_page < start_page:
                    st.session_state.end_page = start_page
                
                end_page = st.number_input(
                    "End page", 
                    min_value=start_page, 
                    max_value=total_pages, 
                    key="end_page"
                )
            
            # Convert 1-indexed user input to 0-indexed page indices for processing
            page_indices = list(range(start_page - 1, end_page))
            st.info(f"Processing pages {start_page} to {end_page} (total: {len(page_indices)} pages)")
            
            # Display preview of start and end pages to help users verify selection
            st.subheader("Range Preview")
            preview_col1, preview_col2 = st.columns(2)
            
            with preview_col1:
                st.markdown(f"**Start Page ({start_page})**")
                start_page_index = start_page - 1
                # Render the start page preview image
                page = doc.load_page(start_page_index)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # Scale up for better visibility
                img_bytes = pix.tobytes("png")
                st.image(img_bytes, caption=f"Page {start_page}", width='stretch')
            
            with preview_col2:
                st.markdown(f"**End Page ({end_page})**")
                end_page_index = end_page - 1
                # Render the end page preview image
                page = doc.load_page(end_page_index)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # Scale up for better visibility
                img_bytes = pix.tobytes("png")
                st.image(img_bytes, caption=f"Page {end_page}", width='stretch')
            
        else:  # Custom pages option
            # Allow selection of non-consecutive pages using comma-separated list
            custom_pages = st.text_input("Enter page numbers separated by commas (e.g., 1,3,5,8)")
            preview_button = st.button("Preview Pages")
            
            # Process and preview custom pages when requested or when using previously entered values
            if custom_pages and (preview_button or 'custom_pages_last' in st.session_state and st.session_state.custom_pages_last == custom_pages):
                try:
                    # Store current custom pages value in session state to maintain preview between interactions
                    st.session_state.custom_pages_last = custom_pages
                    
                    # Parse and validate the page numbers entered by the user
                    page_nums = [int(p.strip()) for p in custom_pages.split(",")]
                    # Filter out invalid page numbers
                    valid_pages = [p for p in page_nums if 1 <= p <= total_pages]
                    page_indices = [p - 1 for p in valid_pages]  # Convert to 0-based indices for internal use
                    
                    # Warn if some entered page numbers were invalid
                    if len(valid_pages) != len(page_nums):
                        st.warning(f"Some page numbers were out of range and will be ignored. Valid range: 1-{total_pages}")
                    
                    st.info(f"Processing {len(page_indices)} pages: {', '.join(map(str, valid_pages))}")
                    
                    # Display previews of custom pages (up to 4 to avoid overcrowding the UI)
                    if valid_pages:
                        st.subheader("Page Previews")
                        preview_pages = valid_pages[:4]  # Show max 4 previews
                        
                        # Create a dynamic number of columns based on the preview pages
                        columns = st.columns(min(len(preview_pages), 4))
                        for i, page_num in enumerate(preview_pages):
                            with columns[i]:
                                st.markdown(f"**Page {page_num}**")
                                # Render the preview for each selected page
                                page = doc.load_page(page_num - 1)
                                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
                                img_bytes = pix.tobytes("png")
                                st.image(img_bytes, caption=f"Page {page_num}", width='stretch')
                        
                        # Indicate if not all selected pages are shown in the preview
                        if len(valid_pages) > 4:
                            st.info(f"Showing first 4 of {len(valid_pages)} selected pages")
                except ValueError:
                    # Handle invalid input (non-numeric values)
                    st.error("Please enter valid page numbers separated by commas")
                    page_indices = []
            else:
                # Initialize session state if first time
                if 'custom_pages_last' not in st.session_state:
                    st.session_state.custom_pages_last = ""
                    
                page_indices = []
                if not custom_pages:
                    st.warning("Please specify at least one page number")
                elif not preview_button:
                    st.info("Click 'Preview Pages' to see the selected pages")
        
        # Show the process button only if page_indices is not empty
        if page_indices:
            # Info message about skipping pages without tables
            st.info("ℹ️ Pages with no tables will be skipped")
            
            # Only show the process button if we haven't completed processing or if we're reprocessing
            process_button = st.button("Process Selected Pages")
            
            if process_button:
                # Reset the processing state
                st.session_state.processing_complete = False
                st.session_state.output_final = []
                
                async def process_pages():
                    """
                    Asynchronously processes PDF pages to extract tables.
                    
                    This function:
                    1. Creates tasks for each selected page
                    2. Extracts both text and image data from each page
                    3. Validates and identifies tables using AI
                    4. Processes identified tables into DataFrame objects
                    5. Updates progress indicators throughout the process
                    6. Handles errors and retries when necessary
                    
                    Returns a list of processed table data
                    """
                    tasks = []
                    results_output = []
                    
                    # Shared progress tracking (thread-safe for async tasks)
                    progress_data = {
                        'completed': 0,
                        'total': len(page_indices),
                        'current_page': None
                    }

                    try:
                        # Bound overall page concurrency
                        # Priority: explicit env var -> CPU-based heuristic
                        env_val = os.getenv("PAGE_MAX_CONCURRENCY")
                        if env_val is not None:
                            try:
                                page_max = max(1, int(env_val))
                            except Exception:
                                page_max = 0
                        else:
                            page_max = 0
                        if page_max <= 0:
                            cpu_count = os.cpu_count() or 4
                            # Heuristic: leave one core free; hard-cap at 4 to control memory from 500 DPI renders
                            page_max = max(1, min(4, cpu_count - 1))
                        logging.info(f"Using PAGE_MAX_CONCURRENCY={page_max}")
                        page_semaphore = asyncio.Semaphore(page_max)

                        async def process_one_page(page_no: int):
                            async with page_semaphore:
                                # Update progress data (safe - no Streamlit calls here)
                                progress_data['current_page'] = page_no + 1

                                # Optional fast-skip using PyMuPDF table detection when not using image mode
                                if not table_in_image:
                                    try:
                                        local_doc = pymupdf.open(pdf_path)
                                        try:
                                            local_page = local_doc.load_page(page_no)
                                            tabs = local_page.find_tables()
                                            if len(tabs.tables) == 0:
                                                return None
                                        finally:
                                            local_doc.close()
                                    except Exception as e:
                                        logging.warning(f"Fast table detection failed on page {page_no + 1}: {e}")

                                # Concurrent text/image extraction in threads
                                extracted_text, base64_image = await asyncio.gather(
                                    asyncio.to_thread(get_page_text_thread, pdf_path, page_no),
                                    asyncio.to_thread(get_page_pixel_data, pdf_path, page_no, 500, 'png'),
                                )

                                # LLM header/column identification
                                num_tables, table_headers, columns_per_table, table_info_confidence = await get_validated_table_info(
                                    user_text=user_text,
                                    openai_client=openai_client,
                                    base64_image=base64_image,
                                    model=vision_model
                                )

                                logging.info(f"num_tables: {num_tables}")
                                logging.info(f"table_headers: {table_headers}")

                                if num_tables == 0:
                                    return None

                                # Extract tables for this page
                                page_tables = await process_tables_to_df(
                                    table_headers=table_headers,
                                    expected_columns_per_table=columns_per_table,
                                    user_text=user_text,
                                    extracted_text=extracted_text,
                                    base64_image=base64_image,
                                    openai_client=openai_client,
                                    page_number=page_no,
                                    table_in_image=table_in_image,
                                    add_in_table_and_page_information=add_in_table_and_page_information,
                                    append_raw_extracted_text=False,
                                    model=model
                                )
                                return page_no, page_tables

                        # Schedule all page tasks
                        coros = [process_one_page(pn) for pn in page_indices]

                        results_by_page = {}
                        for fut in asyncio.as_completed(coros):
                            try:
                                result = await fut
                            except Exception as e:
                                logging.error(f"Page task failed: {e}")
                                result = None

                            # Expect a tuple: (page_no, tables)
                            if result is not None:
                                try:
                                    page_no_result, tables = result
                                except Exception:
                                    page_no_result, tables = None, None
                                if page_no_result is not None and tables and len(tables) > 0:
                                    results_by_page[page_no_result] = tables

                            # Update progress after each completion
                            progress_data['completed'] += 1

                        # Reconstruct results in the exact order of selected page indices
                        ordered_results = [results_by_page[pn] for pn in page_indices if pn in results_by_page]
                        # Save ordered page numbers (0-indexed) for accurate labeling in previews
                        st.session_state.ordered_page_numbers = [pn for pn in page_indices if pn in results_by_page]
                        
                        return ordered_results, progress_data

                    except Exception as e:
                        logging.error(f"Processing error details: {str(e)}")
                        return [], progress_data
                
                # Start the asynchronous processing workflow
                start_time = time.time()
                
                with st.spinner(f"Processing {len(page_indices)} page(s)..."):
                    try:
                        # Run the async function in the main thread
                        result = asyncio.run(process_pages())
                        output_final, progress_data = result
                        
                        # Store the output in session state for persistence between Streamlit reruns
                        st.session_state.output_final = output_final
                        st.session_state.processing_complete = True
                        
                        # Calculate and display processing time for performance feedback
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        st.success(f"Processing completed in {elapsed_time:.2f} seconds")
                        
                    except Exception as e:
                        logging.error(f"Async processing error: {str(e)}")
                        st.error("An issue occurred during processing. Please try again. If the issue persists, try with a different page range or check your PDF file.")
                        st.session_state.output_final = []
                        st.session_state.processing_complete = False
            
            # Results display section - shows after processing is complete
            if st.session_state.processing_complete:
                output_final = st.session_state.output_final
                
                # Only show preview and download options if we have results
                if output_final and len(output_final) > 0:
                    # Add a toggle to control preview visibility
                    # This helps manage UI complexity for large results
                    show_preview = st.checkbox("Show data preview", value=True)
                    
                    if show_preview:
                        # Let user select how to format the preview
                        # Different formats are useful for different use cases
                        preview_format = st.selectbox(
                            "Select preview format:",
                            options=["Format 1: All tables concatenated", 
                                    "Format 2: Tables by page", 
                                    "Format 3: All tables on one sheet"],
                            index=2  # Default to Format 3
                        )
                        
                        # Add download button for the currently selected format
                        # Ensure the output directory exists
                        if not os.path.exists("output_files"):
                            os.makedirs("output_files")
                            
                        # Create the download button based on file format
                        if file_format == "Excel (.xlsx)":
                            # Get the format option from the radio button selection
                            format_option = int(preview_format.split(":")[0].split(" ")[1])
                            excel_file = f'output_files/{file_name}_format_{format_option}.xlsx'
                            write_output_final(output_final, excel_path=excel_file, option=format_option)
                            
                            with open(excel_file, "rb") as file:
                                st.download_button(
                                    label="Download",
                                    data=file,
                                    file_name=f"{file_name}_{preview_format.split(':')[0].strip().replace(' ', '_').lower()}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        else:  # CSV format
                            csv_base_path = f'output_files/{file_name}'
                            format_option = int(preview_format.split(":")[0].split(" ")[1])
                            
                            if format_option == 1:  # Format 1: All tables concatenated
                                csv_file = f'{csv_base_path}_concatenated.csv'
                                write_output_to_csv(output_final, csv_base_path=csv_base_path, option=1)
                                
                                with open(csv_file, "rb") as file:
                                    st.download_button(
                                        label="Download",
                                        data=file,
                                        file_name=f"{file_name}_concatenated.csv",
                                        mime="text/csv"
                                    )
                            elif format_option == 2:  # Format 2: Tables by page
                                # For CSV format 2, we create a zip with multiple files
                                csv_files = write_output_to_csv(output_final, csv_base_path=csv_base_path, option=2)
                                
                                if csv_files:
                                    # Create a zip file for multiple CSV files
                                    zip_buffer = io.BytesIO()
                                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                        for csv_path in csv_files:
                                            filename = os.path.basename(csv_path)
                                            with open(csv_path, "rb") as f:
                                                zip_file.writestr(filename, f.read())
                                    
                                    # Set buffer position to start
                                    zip_buffer.seek(0)
                                    
                                    st.download_button(
                                        label="Download",
                                        data=zip_buffer,
                                        file_name=f"{file_name}_pages.zip",
                                        mime="application/zip"
                                    )
                            else:  # Format 3: All tables on one sheet
                                csv_file = f'{csv_base_path}_all_tables_with_gaps.csv'
                                write_output_to_csv(output_final, csv_base_path=csv_base_path, option=3)
                                
                                with open(csv_file, "rb") as file:
                                    st.download_button(
                                        label="Download",
                                        data=file,
                                        file_name=f"{file_name}_all_tables_with_gaps.csv",
                                        mime="text/csv"
                                    )
                        
                        # Ensure all dfs are unique
                        all_dfs = list(itertools.chain.from_iterable(output_final))
                        all_dfs = remove_duplicate_dfs(all_dfs)
                        
                        if preview_format == "Format 1: All tables concatenated":
                            # Format 1: All tables concatenated vertically
                            st.markdown("**Preview of 'All tables concatenated' format:**")
                            
                            if all_dfs:
                                # Limit to first 100 rows for preview
                                merged_df = pd.concat(all_dfs, axis=0)
                                merged_df = merged_df.reset_index(drop=True)
                                preview_rows = min(100, len(merged_df))
                                st.dataframe(merged_df.head(preview_rows), width='stretch')
                                if len(merged_df) > preview_rows:
                                    st.info(f"Showing first {preview_rows} rows out of {len(merged_df)} total rows. Download the file to see all data.")
                            else:
                                st.info("No tables found to preview.")
                                
                        elif preview_format == "Format 2: Tables by page":
                            # Format 2: Tables by page
                            st.markdown("**Preview of 'Tables by page' format:**")
                            
                            # Create tabs for each page
                            if output_final:
                                # Limit to first 5 pages for preview
                                preview_pages = min(5, len(output_final))
                                page_labels = []
                                if 'ordered_page_numbers' in st.session_state and st.session_state.ordered_page_numbers:
                                    page_labels = [f"Page {pn + 1}" for pn in st.session_state.ordered_page_numbers[:preview_pages]]
                                else:
                                    page_labels = [f"Page {i+1}" for i in range(preview_pages)]
                                tabs = st.tabs(page_labels)
                                
                                for i in range(preview_pages):
                                    with tabs[i]:
                                        if output_final[i]:  # If page has tables
                                            for j, df in enumerate(output_final[i]):
                                                st.markdown(f"**Table {j+1}**")
                                                st.dataframe(df, width='stretch')
                                                if j < len(output_final[i]) - 1:
                                                    st.markdown("---")
                                        else:
                                            st.info("No tables found on this page.")
                                
                                if len(output_final) > 5:
                                    st.info(f"Showing first 5 pages out of {len(output_final)} total pages. Download the file to see all pages.")
                            else:
                                st.info("No pages with tables found to preview.")
                                
                        else:  # Format 3: All tables on one sheet
                            # Format 3: All tables on one sheet with gaps
                            st.markdown("**Preview of 'All tables on one sheet' format:**")
                            
                            # Limit to first 5 tables for preview
                            preview_dfs = all_dfs[:min(5, len(all_dfs))]
                            
                            # Display each table with a separator
                            for i, df in enumerate(preview_dfs):
                                st.markdown(f"**Table {i+1}**")
                                st.dataframe(df, width='stretch')
                                
                                # Add a separator between tables (except after the last one)
                                if i < len(preview_dfs) - 1:
                                    st.markdown("---")
                            
                            # Show a message if there are more tables
                            if len(all_dfs) > 5:
                                st.info("Download the file to see all tables.")
                    
                    # Add a horizontal line between preview and download sections
                    st.markdown("---")
                    

                else:
                    st.warning("No tables were found in the selected pages.")
    
    except Exception as e:
        st.error("An issue occurred while processing the PDF. Please try again or try with a different PDF file.")
        logging.error(f"PDF processing error details: {str(e)}")
    
    finally:
        # Close the document before attempting to delete the file
        if 'doc' in locals() and doc:
            doc.close()
            
        # Clean up the temporary file with better error handling
        if os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
            except PermissionError:
                logging.warning(f"Could not delete temporary file {pdf_path} - it may still be in use")
            except Exception as e:
                logging.warning(f"Error deleting temporary file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<p style='font-size: 12px;'>AIs can make mistakes, please review the output before using it for any purpose.</p>", unsafe_allow_html=True)

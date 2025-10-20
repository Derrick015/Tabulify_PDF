import logging
import pymupdf
import base64
import pandas as pd
import itertools
import asyncio
import os
import re, unicodedata
from openai import AsyncOpenAI
from pydantic import  create_model
from src.llm import table_identification_llm,  vision_llm_parser

# Concurrency controls for OpenAI calls
# Increased default from 8 to 16 for better parallelism with OpenAI API
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "16"))
SEMAPHORE_LIMIT = asyncio.Semaphore(MAX_CONCURRENCY)

async def with_openai_semaphore(coro_func, *args, **kwargs):
    """Run an async OpenAI call under a bounded semaphore."""
    async with SEMAPHORE_LIMIT:
        return await coro_func(*args, **kwargs)


def remove_duplicate_dfs(df_list):
    """
    Remove duplicate DataFrames from a list by hashing a safe, hashable representation
    of each DataFrame.

    The previous implementation failed when a DataFrame contained unhashable
    objects (e.g. list, dict, set) because ``hash_pandas_object`` requires the
    underlying NumPy array to contain hashable types. We now coerce any
    unhashable cell value to a string before hashing so that the function works
    for all DataFrames produced by the extraction pipeline.

    Parameters
    ----------
    df_list : list[pd.DataFrame]
        List of DataFrames collected during extraction.

    Returns
    -------
    list[pd.DataFrame]
        The input list with duplicate DataFrames removed (order preserved).
    """
    seen = set()
    unique_dfs = []
    for df in df_list:
        # Coerce unhashable elements (lists, dicts, sets, etc.) to strings so that
        # pd.util.hash_pandas_object will not raise TypeError.
        df_hashable = df.map(
            lambda x: str(x) if isinstance(x, (list, dict, set)) else x
        )
        # Sort columns to ensure consistent hashing irrespective of column order.
        df_hash = pd.util.hash_pandas_object(
            df_hashable.sort_index(axis=1), index=False
        ).sum()
        if df_hash not in seen:
            unique_dfs.append(df)
            seen.add(df_hash)
    return unique_dfs


def get_page_pixel_data(pdf_path, page_no, dpi=300, image_type='png'):
    """
    Converts a specified PDF page to a base64-encoded image.

    Parameters:
        pdf_path (str): The path to the PDF file
        page_no (int): Page number (0-indexed)
        dpi (int): Resolution in dots per inch
        image_type (str): Image file format ('png', 'jpeg', etc.)

    Returns:
        str: Base64-encoded image representation
    """
    logging.info(f"Converting PDF page {page_no + 1} to base64 image. DPI={dpi}, Format={image_type}")
    doc = pymupdf.open(pdf_path)
    page_count = doc.page_count
    if page_no >= page_count or page_no < 0:
        logging.error(f"Page number {page_no} out of range. Total pages: {page_count}")
        raise ValueError(f"Page number {page_no} out of range. Total pages: {page_count}")
    page = doc[page_no]

    zoom_factor = dpi / 72
    matrix = pymupdf.Matrix(zoom_factor, zoom_factor)

    pix = page.get_pixmap(matrix=matrix)
    png_data = pix.tobytes(image_type)
    base64_image = base64.b64encode(png_data).decode('utf-8')

    doc.close()
    logging.info("Finished converting page to base64.")
    return base64_image

def get_page_text_thread(pdf_path: str, page_no: int) -> str:
    """
    Thread-friendly helper to extract text for a single page by opening
    the document in this thread. Safe to be called via asyncio.to_thread.
    """
    doc = pymupdf.open(pdf_path)
    try:
        return doc.load_page(page_no).get_text()
    finally:
        doc.close()



def _describe_exception(err: Exception) -> str:
    """
    Build a detailed string for exceptions, including HTTP body if present.
    Useful for surfacing OpenAI 4xx/5xx details hidden by TaskGroup.
    """
    try:
        parts = [f"{type(err).__name__}: {str(err)}"]
        # Common OpenAI SDK attrs
        for attr in ["status_code", "status", "code", "type", "message", "param"]:
            val = getattr(err, attr, None)
            if val:
                parts.append(f"{attr}={val}")

        resp = getattr(err, "response", None)
        if resp is not None:
            try:
                body = getattr(resp, "body", None)
                if body is not None:
                    if isinstance(body, (bytes, bytearray)):
                        body_str = body.decode("utf-8", errors="ignore")
                    else:
                        body_str = str(body)
                    parts.append(f"body={body_str[:2000]}")
                elif hasattr(resp, "json"):
                    try:
                        j = resp.json()
                        parts.append(f"json={str(j)[:2000]}")
                    except Exception:
                        pass
                elif hasattr(resp, "text"):
                    try:
                        parts.append(f"text={str(resp.text)[:2000]}")
                    except Exception:
                        pass
            except Exception:
                pass

        # ExceptionGroup support (Python 3.11)
        if hasattr(err, "exceptions") and isinstance(getattr(err, "exceptions"), list):
            try:
                sub_msgs = []
                for sub in err.exceptions:  # type: ignore[attr-defined]
                    if isinstance(sub, Exception):
                        sub_msgs.append(_describe_exception(sub))
                    else:
                        sub_msgs.append(str(sub))
                if sub_msgs:
                    parts.append("sub_exceptions=[" + "; ".join(m[:400] for m in sub_msgs) + "]")
            except Exception:
                pass

        return " | ".join(parts)
    except Exception:
        return f"{type(err).__name__}: {str(err)}"

async def get_validated_table_info(user_text, openai_client, base64_image, model):
    """
    Attempt to retrieve consistent table information by making multiple calls
    to the table identification LLM. If there's a majority match or exact match
    between attempts, return that; otherwise return the third attempt's output.
    
    Parameters:
        text_input (str): Extracted text from the PDF page
        user_text (str): User's request or instructions
        open_api_key (str): OpenAI API key
        base64_image (str): Base64-encoded image of the PDF page
        model (str): OpenAI model to use
        
    Returns:
        tuple: (num_tables, table_headers, columns_per_table, confidence_level)
            - num_tables (int): Number of tables detected
            - table_headers (list[str]): Identified table headers with positions
            - columns_per_table (list[list[str]] | None): For each table, a list of expected column names
            - confidence_level (int): Confidence level (0=highest, higher numbers=lower confidence)
    """

    async def async_pattern_desc():
        """
        Inner async function to call the table identification LLM.
        
        Returns:
            str: The response text from the LLM containing table identification information
        """
        return await with_openai_semaphore(
            table_identification_llm,
            user_text=user_text,
            base64_image=base64_image,
            openai_client=openai_client,
            model=model
        )

    tasks = []
    # Create first two tasks
    async with asyncio.TaskGroup() as tg:
        tasks.append(tg.create_task(async_pattern_desc()))
        # tasks.append(tg.create_task(async_pattern_desc()))
    
    # Wait for first two tasks
    output1 = await tasks[0]
    # output2 = await tasks[1]
    logging.debug(f"LLM attempt 1 output:\n{output1}")
    # logging.debug(f"LLM attempt 2 output:\n{output2}")

    num_tables1 = output1.num_tables
    headers1 = output1.table_headers_and_positions
    columns1 = getattr(output1, "columns_per_table", None)

    return num_tables1, headers1, columns1, 0



def extract_columns(response_text, tables_to_target):
    """
    Extract column info from the LLM response text using regex.
    
    Parameters:
        response_text (str): The text response from the LLM containing column information
        tables_to_target (list): List of table headers to target for extraction
        
    Returns:
        list: List of dictionaries containing extracted column information with keys:
            - index: Table index
            - table_header: Header of the table
            - column_names: List of column names
            - example_values_per_column: Example values for each column
            - table_location: Location information for the table
    """
    logging.debug("Extracting columns from LLM response.")
    pattern = r'index:\s*\[(\d+)\].*?column_names:\s*\[(.*?)\].*?example_value_per_column:\s*\[(.*?)\].*?table_location:\s*\[(.*?)\]'
    matches = re.findall(pattern, response_text)

    results = []
    for index_str, columns_str, example_values_str, location_str in matches:
        index_value = int(index_str)
        columns_list = [col.strip().strip('"\'') for col in columns_str.split(',')]

        # Parse example values into a dictionary
        example_values = {}
        for pair in example_values_str.split(','):
            if ':' in pair:
                key, value = pair.split(':')
                key = key.strip().strip('"\'')
                value = value.strip().strip('()').strip('"\'')
                example_values[key] = value

        header = tables_to_target[index_value]
        results.append({
            "index": index_value,
            "table_header": header,
            "column_names": columns_list,
            "example_values_per_column": example_values,
            "table_location": location_str.strip()
        })
    logging.debug(f"Extracted columns result: {results}")
    return results



def rows_to_df(rows):
    
    records = []
    for r in rows:
        if hasattr(r, "model_dump"):          # Pydantic v2
            records.append(r.model_dump(by_alias=True))
        elif hasattr(r, "dict"):              # Pydantic v1
            records.append(r.dict(by_alias=True))
        elif isinstance(r, dict):             # already dicts
            records.append(r)
        else:
            raise TypeError(f"Unsupported row type: {type(r)}")
    return pd.DataFrame(records)




def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text))  # expands “ﬁ” → “fi”, etc. It’s false because the extracted text contains Unicode ligatures (e.g., “ﬁ” is a single character U+FB01), while your sample string has plain “fi”.
    text = re.sub(r"[\W_]+", " ", text).lower().strip()
    text = re.sub(r"\s+", "", text).strip()
    return text

def sanitize_field_name(original_name: str, fallback_prefix: str = "field") -> str:
    """
    Convert arbitrary column/header text into a safe JSONSchema/Pydantic field name.
    - Keep alphanumerics and underscore
    - Replace other characters with underscore
    - Ensure it starts with a letter; otherwise prefix
    - Collapse repeated underscores and trim
    - Lowercase for stability
    """
    try:
        if not isinstance(original_name, str):
            original_name = str(original_name)
        # Replace invalid chars
        safe = re.sub(r"[^A-Za-z0-9_]", "_", original_name)
        # Collapse multiple underscores
        safe = re.sub(r"_+", "_", safe)
        safe = safe.strip("_")
        if not safe:
            safe = f"{fallback_prefix}_1"
        # Must start with a letter
        if not re.match(r"^[A-Za-z]", safe):
            safe = f"{fallback_prefix}_{safe}"
        return safe.lower()
    except Exception:
        return f"{fallback_prefix}_1"

async def process_tables_to_df(
    table_headers,
    expected_columns_per_table,
    user_text, 
    extracted_text, 
    base64_image, 
    openai_client, 
    page_number,
    table_in_image,
    add_in_table_and_page_information,  
    append_raw_extracted_text,
    model,
    raise_openai_errors: bool = False,
    max_retries=2,
    delay=2,
    max_extract_retries=2
):
    """
    Process tables by calling an LLM parser with exponential backoff.
    
    Parameters:
        table_headers (list): List of table headers to process
        user_text (str): User's text input
        extracted_text (str): Text extracted from the PDF
        base64_image (str): Base64-encoded image of the PDF page
        open_api_key (str): OpenAI API key
        page_number (int): Page number being processed (0-indexed)
        table_in_image (bool): Whether the table is in the image
        add_in_table_and_page_information (bool): Whether to add table and page information
        model (str): LLM model to use
        max_retries (int): Maximum number of retries for API calls
        delay (int): Initial delay in seconds before retrying
        max_extract_retries_for_extraction_failures (int): Maximum retries for extraction failures
        
    Returns:
        list: List of pandas DataFrames containing the extracted table data
    """
    try:
        logging.info(f"Processing tables to DataFrame for page {page_number + 1}")

        # Backwards-compatible: expected_columns_per_table is optionally provided via closure (set later)
        # expected_columns_per_table = locals().get("expected_columns_per_table", None) # This line is removed
        results_output = []  # ensure defined even if attempts fail
        confidence_level = 0
        first_error: Exception | None = None
        # Build per-table column name sanitization maps so schema is valid for strict structured outputs
        per_table_safe_column_maps: dict[int, dict[str, str]] = {}
        for attempt in range(max_retries):
            try:
                logging.info(f"[Starting: Model {model}] Attempt {attempt+1} of {max_retries}. Delay={delay}")
                tasks = []
                async with asyncio.TaskGroup() as tg:
                    for idx, table in enumerate(table_headers):
                        exp_cols = expected_columns_per_table[idx]
                        if exp_cols:
                            # Sanitize to valid JSON schema property names
                            safe_to_original: dict[str, str] = {}
                            safe_cols: list[str] = []
                            for pos, col_name in enumerate(exp_cols):
                                safe_name = sanitize_field_name(col_name, fallback_prefix=f"col{pos}")
                                # Handle collisions by appending an index
                                collision_index = 2
                                base_name = safe_name
                                while safe_name in safe_to_original and safe_to_original[safe_name] != col_name:
                                    safe_name = f"{base_name}_{collision_index}"
                                    collision_index += 1
                                safe_to_original[safe_name] = col_name
                                safe_cols.append(safe_name)
                            per_table_safe_column_maps[idx] = safe_to_original
                            # One row type with the exact expected columns as required string fields
                            RowModel = create_model(
                                f"Row_{idx}",
                                **{col: (str, ...) for col in safe_cols}
                            )
                            # The output is a list of such rows under output_dictionary
                            OutputDictionary = create_model(
                                f"OutputDictionary_{idx}",
                                output_dictionary=(list[RowModel], ...))
                        else:
                            raise ValueError(f"Expected columns per table is not a list or index {idx} is out of range")

                        async def one_call(i: int, table_name: str, schema_model):
                            try:
                                return await with_openai_semaphore(
                                    vision_llm_parser,
                                    user_text=user_text,
                                    text_input=extracted_text,
                                    table_to_target=table_name,
                                    base64_image=base64_image,
                                    openai_client=openai_client,
                                    model=model,
                                    structure_output=schema_model
                                )
                            except Exception as call_err:
                                nonlocal first_error
                                if first_error is None:
                                    first_error = call_err
                                logging.error(
                                    f"OpenAI table extraction failed [page {page_number + 1}, table {i} '{table_name}']: {_describe_exception(call_err)}"
                                )
                                # Prevent TaskGroup from cancelling others by swallowing here
                                return call_err

                        tasks.append(
                            tg.create_task(one_call(idx, table, OutputDictionary))
                        )

                # Collect results, preserving errors as values
                gathered = [t.result() for t in tasks]
                results_output = []
                errors_seen = []
                for r in gathered:
                    if isinstance(r, Exception):
                        errors_seen.append(r)
                        results_output.append(None)
                    else:
                        results_output.append(r)

                if errors_seen and raise_openai_errors and first_error is not None:
                    # Re-raise to surface exact OpenAI/API error
                    raise first_error
                logging.info(f"Length of results output: {len(results_output)}")
                logging.info(f"Successfully retrieved data using model '{model}'.")
                confidence_level = 0
                break
            except Exception as e:
                logging.warning(
                    f"[Failed: Model {model}] Attempt {attempt+1} of {max_retries} failed: {_describe_exception(e)}. "
                    f"Retrying in {delay} second(s)..."
                )
                if attempt == max_retries - 1:
                    confidence_level = 1
                    logging.warning(f"Max retries with '{model}' exhausted.")
                    if raise_openai_errors and first_error is not None:
                        raise first_error
                else:
                    await asyncio.sleep(delay)

        # 2) Process the results into DataFrames
        # logging.info(f"Comparing results ouput {len(results_output)} with the table headers {len(table_headers) } for page {page_number + 1}")
        



        # 3) Process the DataFrames
        df_list = []

        for i, out in enumerate(results_output or []):
            extract_retry_count = 0

            # max_extract_retries = max_extract_retries_for_extraction_failures  # Maximum number of retries for extraction failures
            
            while extract_retry_count < max_extract_retries:
                try:
                    logging.info(f"Extracting DataFrame for table index {i}")

                    # If results_output is a list per table (e.g., [[Row_0...], [Row_1...], ...]):
                    if isinstance(out, list):
                        df = rows_to_df(out)
                        # Rename sanitized columns back to original for this table index if mapping exists
                        safe_to_original = per_table_safe_column_maps.get(i, {})
                        if safe_to_original:
                            df = df.rename(columns={safe: orig for safe, orig in safe_to_original.items()})
                    else:
                        raise ValueError(f"Results output is not a list for table index {i}")

                    logging.info(f"Parsed DataFrame for table index {i} with shape {df.shape}")

                    if not table_in_image:
                        # Replace any values that are not in the extracted text with "N/A"  
                        extracted_text_normalized = normalize_text(extracted_text)

                        if append_raw_extracted_text:
                            df_raw_extracted_text = df.copy()
                            df_raw_extracted_text.columns = [f"{col}_llm_extracted_raw" for col in df_raw_extracted_text.columns]

                        df[df.columns] = df[df.columns].map(
                            lambda val: val if normalize_text(val) in extracted_text_normalized else "N/A"
                        )

                        if append_raw_extracted_text:
                            df = pd.concat([df, df_raw_extracted_text], axis=1)
                        
                    if add_in_table_and_page_information:
                        # Split the table header and position information
                        # New format: "Screw Machine -- Located at the bottom left of the page"
                        header_str = str(table_headers[i])
                        table_header = ""
                        table_position = ""
                        if "--" in header_str:
                            parts = header_str.split("--", 1)
                            table_header = parts[0].strip(' "\'')
                            table_position = parts[1].strip(' "\'')
                        else:
                            table_header = header_str.strip(' "\'')
                            table_position = ""
                        # Add as separate columns
                        df['TableHeader'] = table_header
                        df['TablePosition'] = table_position
                        df['PageNumber'] = page_number + 1
                        df['ConfidenceForExtraction'] = confidence_level

                        logging.info(f"extraction_confidence: {confidence_level}")

                    df_list.append(df)
                    break  # Successfully extracted, exit the retry loop
                
                except Exception as e:
                    extract_retry_count += 1
                    confidence_level = confidence_level + 1
                    if extract_retry_count <= max_extract_retries:
                        logging.warning(f"Could not extract table with index {i} on page {page_number + 1}. Retry attempt {extract_retry_count}...")

                        try:
                            logging.info(f"Regenerating table data for index {i}, table '{table_headers[i]}'")
                            # Rebuild OutputDictionary for this index
                            exp_cols_retry = expected_columns_per_table[i]
                            # Rebuild and store sanitized mapping for retry
                            safe_to_original_retry: dict[str, str] = {}
                            safe_cols_retry: list[str] = []
                            for pos, col_name in enumerate(exp_cols_retry):
                                safe_name = sanitize_field_name(col_name, fallback_prefix=f"col{pos}")
                                collision_index = 2
                                base_name = safe_name
                                while safe_name in safe_to_original_retry and safe_to_original_retry[safe_name] != col_name:
                                    safe_name = f"{base_name}_{collision_index}"
                                    collision_index += 1
                                safe_to_original_retry[safe_name] = col_name
                                safe_cols_retry.append(safe_name)
                            per_table_safe_column_maps[i] = safe_to_original_retry
                            RowModelRetry = create_model(
                                f"Row_{i}_retry",
                                **{col: (str, ...) for col in safe_cols_retry}
                            )
                            OutputDictionaryRetry = create_model(
                                f"OutputDictionary_{i}_retry",
                                output_dictionary=(list[RowModelRetry], ...)
                            )
                            try:
                                out = await with_openai_semaphore(
                                    vision_llm_parser,
                                    user_text=user_text,
                                    text_input=extracted_text,
                                    table_to_target=table_headers[i],
                                    base64_image=base64_image,
                                    openai_client=openai_client,
                                    model=model,
                                    structure_output=OutputDictionaryRetry
                                )
                            except Exception as regen_err:
                                logging.error(
                                    f"Regeneration OpenAI error [page {page_number + 1}, table {i} '{table_headers[i]}']: {_describe_exception(regen_err)}"
                                )
                                if raise_openai_errors:
                                    raise
                                out = None
                            results_output[i] = out  # Update the results_output with the new result
                            logging.info(f"Regenerated table data for index {i}, with model, table '{table_headers[i]}, output was {out}")
                        except Exception as regen_error:
                            logging.error(f"Failed to regenerate table data: {str(regen_error)}")
                            # Continue to next retry or exit loop if max retries reached
                    else:
                        logging.warning(f"Could not extract table with index {i} on page {page_number + 1} after {max_extract_retries} retries, skipping.")
                        break  # Exit the retry loop after max retries

        logging.info(f"Completed processing tables to DataFrame for page {page_number + 1}. Total tables extracted: {len(df_list)}")
        
        # Handle case where all tables failed extraction
        if not df_list:
            logging.error(f"No tables could be extracted from the results. - Page {page_number + 1}")
            # df_list.append(pd.DataFrame()) # apend empty df
            df_list = []
        return df_list
    
    except Exception as e:
        logging.error(f"Error processing tables to DataFrame: {str(e)}")
        df_list = []
        return df_list


async def extract_tables_from_pdf(
    pdf_path: str,
    pages=None,
    page_range=None,
    instructions="Extract all data from the table(s)",
    table_in_image=False,
    add_table_info=False,
    append_raw_extracted_text=False,
    table_identification_model="gpt-5-mini",
    table_extraction_model="gpt-5-mini",
    output_name="output_file",
    output_format="xlsx",
    style="sheet",
    save_output=True,
    return_dataframes=True,
    open_api_key=None,
    raise_openai_errors=False
):
    """
    Extract tables from PDF - Notebook friendly version
    
    Parameters:
    -----------
    pdf_path : str
        Path to the PDF file
    pages : str or list, optional
        Comma-separated string or list of page numbers (1-indexed)
    page_range : tuple, optional
        (start, end) tuple for page range (1-indexed, inclusive)
    instructions : str
        Instructions for the LLM
    table_in_image : bool
        Enable Image & Inference Mode
    add_table_info : bool
        Include table and page metadata
    table_identification_model : str
        LLM model to use for table identification
    table_extraction_model : str
        LLM model to use for table extraction
    output_name : str
        Base name for output files
    output_format : str
        "xlsx" or "csv"
    style : str
        "concatenated", "by_page", or "sheet"
    save_output : bool
        Whether to save output files
    return_dataframes : bool
        Whether to return DataFrames for notebook display
    
    Returns:
    --------
    list or None
        List of DataFrames if return_dataframes=True, else None
    """
    
    openai_client = AsyncOpenAI(api_key=open_api_key)

    logging.info("Opening PDF: %s", pdf_path)
    doc = pymupdf.open(pdf_path)
    # local parser to avoid importing the duplicate helper
    def _local_parse_page_selection(total_pages: int, pages_val=None, page_range_val=None):
        if pages_val:
            if isinstance(pages_val, str):
                page_nums = [int(p.strip()) for p in pages_val.split(",") if p.strip()]
            else:
                page_nums = pages_val if isinstance(pages_val, list) else [pages_val]
            validated = [p for p in page_nums if 1 <= p <= total_pages]
            if not validated:
                raise ValueError("No valid page numbers supplied.")
            return [p - 1 for p in validated]
        if page_range_val:
            start, end = page_range_val
            if not (1 <= start <= end <= total_pages):
                raise ValueError("Invalid range values. Ensure 1 <= start <= end <= total_pages")
            return list(range(start - 1, end))
        return list(range(total_pages))

    page_indices = _local_parse_page_selection(doc.page_count, pages, page_range)
    logging.info("Processing %d page(s): %s", len(page_indices), [i + 1 for i in page_indices])

    results_output = []

    # Bound overall page concurrency (local semaphore)
    cpu_count = os.cpu_count() or 4
    # Increased from 4 to 6 since we reduced DPI from 500 to 200 (less memory per page)
    page_max = max(1, min(6, cpu_count - 1))
    logging.info(f"Using PAGE_MAX_CONCURRENCY={page_max}")

    page_semaphore = asyncio.Semaphore(page_max)

    async def process_one_page(page_no: int):
        async with page_semaphore:
            page = doc.load_page(page_no)
            if not table_in_image:
                if len(page.find_tables().tables) == 0:
                    logging.info("Page %d: no tables detected via PyMuPDF, skipping.", page_no + 1)
                    return None

            # Run text and image extraction concurrently
            # Using 200 DPI for faster processing while maintaining quality
            extracted_text, base64_image = await asyncio.gather(
                asyncio.to_thread(get_page_text_thread, pdf_path, page_no),
                asyncio.to_thread(get_page_pixel_data, pdf_path, page_no, 200, "png"),
            )

            # Validate via LLM (bounded)
            num_tables, table_headers, columns_per_table, table_info_confidence = await get_validated_table_info(
                user_text=instructions,
                openai_client=openai_client,
                base64_image=base64_image,
                model=table_identification_model,
            )

            if num_tables == 0:
                logging.info("Page %d: LLM reported no tables, skipping.", page_no + 1)
                return None

            logging.info("Page %d: %d table(s) detected.", page_no + 1, num_tables)

            try:
                return await process_tables_to_df(
                    table_headers=table_headers,
                    expected_columns_per_table=columns_per_table,
                    user_text=instructions,
                    extracted_text=extracted_text,
                    base64_image=base64_image,
                    openai_client=openai_client,
                    page_number=page_no,
                    raise_openai_errors=raise_openai_errors,
                    table_in_image=table_in_image,
                    add_in_table_and_page_information=add_table_info,
                    append_raw_extracted_text=append_raw_extracted_text,
                    model=table_extraction_model,
                )
            except Exception as e:
                logging.error(f"Page {page_no+1} processing failed: {e}")
                return []

    # Schedule all selected pages concurrently with a cap
    page_tasks = [asyncio.create_task(process_one_page(pn)) for pn in page_indices]
    page_results = await asyncio.gather(*page_tasks, return_exceptions=True)
    for res in page_results:
        if isinstance(res, Exception) or res is None:
            continue
        results_output.append(res)

    doc.close()
    
    if not results_output:
        logging.warning("No tables were extracted.")
        return pd.DataFrame()
    
    # Save output if requested
    if save_output:
        os.makedirs("output_files", exist_ok=True)
        base_path = os.path.join("output_files", output_name)
        
        style_map = {
            "concatenated": 1,
            "by_page": 2,
            "sheet": 3,
        }
        option = style_map[style]
        
        if output_format == "xlsx":
            excel_path = f"{base_path}.xlsx" if option == 1 else f"{base_path}_format_{option}.xlsx"
            write_output_final(results_output, excel_path=excel_path, option=option)
            logging.info("Excel written to %s", excel_path)
        else:
            csv_files = write_output_to_csv(results_output, csv_base_path=base_path, option=option)
            if isinstance(csv_files, list):
                logging.info("CSV files written: %s", ", ".join(csv_files))
            else:
                logging.info("CSV file written to %s", csv_files)
    
    # Return DataFrames for notebook display
    if return_dataframes:
        try:
            all_dfs = remove_duplicate_dfs(list(itertools.chain.from_iterable(results_output)))
            logging.info("Extraction complete. %d unique table(s) found.", len(all_dfs))
            return all_dfs
        except:
            logging.error("Error removing duplicate DataFrames. Returning all DataFrames.")
            return results_output
    
    return None



def sanitize_worksheet_name(name):
    """
    Sanitie Excel worksheet names by removing or replacing characters that are not allowed.
    
    Excel worksheet naming rules:
    - Can't exceed 31 characters
    - Can't contain: [ ] : * ? / \
    - Can't be 'History' as it's a reserved name
    
    Args:
        name (str): The original worksheet name
        
    Returns:
        str: Sanitized worksheet name safe for Excel
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[\[\]:*?/\\]', '_', str(name))
    
    # Truncate to 31 characters (Excel limit)
    if len(sanitized) > 31:
        sanitized = sanitized[:31]
        
    # Make sure it's not empty or 'History' (reserved name)
    if not sanitized or sanitized.lower() == 'history':
        sanitized = 'Sheet1'
        
    return sanitized

def write_output_final(output_final, excel_path, option=1, gap_rows=2):
    """
    Writes nested lists of DataFrames (`output_final`) to Excel in 3 different ways.

    Parameters:
        output_final (list): A list of lists of DataFrames
        excel_path (str): Output Excel filename/path
        option (int): Choose 1 of 3 write modes:
                   1 = Horizontally merge (side-by-side) all DataFrames into one wide table (one sheet)
                   2 = Each top-level group on its own sheet, with `gap_rows` blank rows between sub-DataFrames
                   3 = Flatten all DataFrames onto one sheet vertically, with `gap_rows` blank rows between them
        gap_rows (int): How many blank rows to insert between tables (used in options 2 and 3)
        
    Returns:
        None
    """
    logging.info(f"Writing output to Excel at '{excel_path}' with option={option}.")
    
    def sanitize_dataframe(df):
        """
        Create a clean copy of a DataFrame with problematic characters replaced in both
        column names and string data to ensure compatibility with Excel limitations.
        
        Parameters:
            df (pandas.DataFrame): The DataFrame to sanitize
            
        Returns:
            pandas.DataFrame: A sanitized copy of the input DataFrame with problematic characters
                             replaced and formatted to avoid Excel compatibility issues
        """
        # Create a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Replace problematic characters in column names
        df_clean.columns = [re.sub(r'[\[\]:*?/\\]', '_', str(col)) for col in df_clean.columns]
        
        # Replace problematic characters in string data
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':  # Only process string columns
                # Replace problematic characters with underscores
                df_clean[col] = df_clean[col].astype(str).apply(
                    lambda x: re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', x) if pd.notna(x) else x
                )
        
        return df_clean
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            
            if option == 1:
                logging.debug("Option 1: Merging all DataFrames horizontally on one sheet.")
                all_dfs = list(itertools.chain.from_iterable(output_final))
                # Sanitize each DataFrame before concatenation
                all_dfs_clean = [sanitize_dataframe(df) for df in all_dfs]
                merged_df = pd.concat(all_dfs_clean, axis=0)
                merged_df.to_excel(writer, sheet_name=sanitize_worksheet_name("AllTablesMerged"), index=False)
                
            elif option == 2:
                logging.debug("Option 2: Each group on a different sheet, gap_rows between each.")
                for page_idx, df_group in enumerate(output_final):
                    sheet_name = sanitize_worksheet_name(f"Page_{page_idx+1}")
                    start_row = 0
                    for df in df_group:
                        # Sanitize the DataFrame
                        df_clean = sanitize_dataframe(df)
                        df_clean.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                        start_row += len(df_clean) + 1 + gap_rows
                        
            elif option == 3:
                logging.debug("Option 3: Flatten all DataFrames on one sheet vertically with gap_rows.")
                all_dfs = list(itertools.chain.from_iterable(output_final))
                sheet_name = sanitize_worksheet_name("AllTablesWithGaps")
                start_row = 0
                for df in all_dfs:
                    # Sanitize the DataFrame
                    df_clean = sanitize_dataframe(df)
                    df_clean.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                    start_row += len(df_clean) + 1 + gap_rows
                
            else:
                logging.error("Invalid `option` provided to write_output_final.")
                raise ValueError("Invalid `option` - must be 1, 2, or 3.")
    
    except Exception as e:
        logging.error(f"Error writing to Excel: {str(e)}")
        raise  # Re-raise the exception after logging

    logging.info("Excel file writing complete.")

def write_output_to_csv(output_final, csv_base_path, option=1, gap_rows=2):
    """
    Writes nested lists of DataFrames (`output_final`) to CSV files in 3 different ways.

    Parameters:
        output_final (list): A list of lists of DataFrames
        csv_base_path (str): Base path/filename for CSV output (without extension)
        option (int): Choose 1 of 3 write modes:
                   1 = Horizontally merge all DataFrames into one CSV file
                   2 = Each top-level group in its own CSV file, with gap rows between tables
                   3 = Flatten all DataFrames into one CSV file with gap rows between them
        gap_rows (int): How many blank rows to insert between tables (for options 2 and 3)
        
    Returns:
        list: List of paths to generated CSV files
    """
    logging.info(f"Writing output to CSV at '{csv_base_path}' with option={option}.")
    generated_files = []
    
    def sanitize_dataframe(df):
        """
        Create a clean copy of a DataFrame with problematic characters replaced in both
        column names and string data to ensure compatibility with CSV format limitations.
        
        Parameters:
            df (pandas.DataFrame): The DataFrame to sanitize
            
        Returns:
            pandas.DataFrame: A sanitized copy of the input DataFrame with problematic characters
                             replaced and formatted to avoid CSV compatibility issues
        """
        # Create a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Replace problematic characters in column names
        df_clean.columns = [re.sub(r'[\[\]:*?/\\]', '_', str(col)) for col in df_clean.columns]
        
        # Replace problematic characters in string data
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':  # Only process string columns
                df_clean[col] = df_clean[col].astype(str).apply(
                    lambda x: re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', x) if pd.notna(x) else x
                )
        
        return df_clean
    
    try:
        if option == 1:
            logging.debug("Option 1: Merging all DataFrames into one CSV file.")
            all_dfs = list(itertools.chain.from_iterable(output_final))
            # Sanitize each DataFrame before concatenation
            all_dfs_clean = [sanitize_dataframe(df) for df in all_dfs]
            merged_df = pd.concat(all_dfs_clean, axis=0)
            
            csv_path = f"{csv_base_path}_concatenated.csv"
            merged_df.to_csv(csv_path, index=False)
            generated_files.append(csv_path)
            
        elif option == 2:
            logging.debug("Option 2: Each group in a separate CSV file.")
            for page_idx, df_group in enumerate(output_final):
                if not df_group:  # Skip empty groups
                    continue
                    
                # Create a new DataFrame for each page with appropriate gaps
                result_df = pd.DataFrame()
                current_row = 0
                
                for df in df_group:
                    df_clean = sanitize_dataframe(df)
                    
                    # Add blank rows if not at the start
                    if current_row > 0:
                        for _ in range(gap_rows):
                            result_df = pd.concat([result_df, pd.DataFrame([[''] * len(df_clean.columns)], columns=df_clean.columns)])
                            current_row += 1
                    
                    # Add the actual data
                    result_df = pd.concat([result_df, df_clean])
                    current_row += len(df_clean)
                
                csv_path = f"{csv_base_path}_page_{page_idx+1}.csv"
                result_df.to_csv(csv_path, index=False)
                generated_files.append(csv_path)
                
        elif option == 3:
            logging.debug("Option 3: Flatten all DataFrames into one CSV with gaps.")
            all_dfs = list(itertools.chain.from_iterable(output_final))
            
            # First determine the maximum column count across all tables
            max_cols = max([len(df.columns) for df in all_dfs]) if all_dfs else 0
            
            # Create a new large DataFrame with appropriate gaps
            result_df = pd.DataFrame()
            
            for i, df in enumerate(all_dfs):
                df_clean = sanitize_dataframe(df)
                
                # Add blank rows if not at the start
                if i > 0:
                    blank_df = pd.DataFrame([[''] * max_cols])
                    for _ in range(gap_rows):
                        result_df = pd.concat([result_df, blank_df])
                
                # Add the current DataFrame
                result_df = pd.concat([result_df, df_clean])
            
            csv_path = f"{csv_base_path}_all_tables_with_gaps.csv"
            result_df.to_csv(csv_path, index=False)
            generated_files.append(csv_path)
            
        else:
            logging.error("Invalid `option` provided to write_output_to_csv.")
            raise ValueError("Invalid `option` - must be 1, 2, or 3.")
    
    except Exception as e:
        logging.error(f"Error writing to CSV: {str(e)}")
        raise  # Re-raise the exception after logging

    logging.info(f"CSV file writing complete. Generated files: {generated_files}")
    return generated_files


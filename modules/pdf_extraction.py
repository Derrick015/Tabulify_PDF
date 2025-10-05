import tkinter as tk
from tkinter import filedialog
import logging
import pymupdf
import base64
import pandas as pd
import ast
import itertools
import asyncio
import os
import pymupdf
import itertools
import pandas as pd
import re, unicodedata
from openai import AsyncOpenAI
import warnings
from pydantic import  create_model
from typing import Iterable, List, Dict, Tuple, Set, Callable, Optional
from modules.llm import table_identification_llm,  vision_llm_parser
# Concurrency controls
OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "8"))
PAGE_MAX_CONCURRENCY = int(os.getenv("PAGE_MAX_CONCURRENCY", "8"))
OPENAI_SEMAPHORE = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)
PAGE_SEMAPHORE = asyncio.Semaphore(PAGE_MAX_CONCURRENCY)

async def with_openai_semaphore(coro_func, *args, **kwargs):
    """Run an async OpenAI call under a bounded semaphore."""
    async with OPENAI_SEMAPHORE:
        return await coro_func(*args, **kwargs)
try:
    import ahocorasick  # pyahocorasick
except Exception as e:  # pragma: no cover
    raise ImportError(
        "pyahocorasick is required for matching. Install it with 'pip install pyahocorasick'"
    ) from e


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
        df_hashable = df.applymap(
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


def extract_text_from_pages(pdf_input, pages=None):
    """
    Extracts text from specified pages in a PDF file using PyMuPDF.

    Parameters:
        pdf_input (str or file-like object): The path to the PDF file or a file-like object.
        pages (int, list, tuple, or None): 
            - If an integer, extracts text from that specific page (0-indexed).
            - If a list of integers, extracts text from the specified pages.
            - If a tuple of two integers, treats it as a range (start, end) and extracts from start (inclusive)
              to end (exclusive).
            - If None, extracts text from all pages.

    Returns:
        str: The concatenated text extracted from the specified pages.
    """
    logging.info("Starting text extraction from PDF.")
    logging.debug(f"Received pdf_input={pdf_input}, pages={pages}")

    text = ""

    # Open the PDF file using PyMuPDF.  
    if isinstance(pdf_input, str):
        logging.debug(f"Opening PDF file from path: {pdf_input}")
        doc = pymupdf.open(pdf_input)
    else:
        logging.debug("Opening PDF file from file-like object.")
        pdf_input.seek(0)
        pdf_bytes = pdf_input.read()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    total_pages = doc.page_count
    logging.debug(f"PDF has {total_pages} pages.")

    # Determine which pages to extract.
    if pages is None:
        page_indices = range(total_pages)
    elif isinstance(pages, int):
        if pages < 0 or pages >= total_pages:
            logging.error(f"Page index {pages} is out of range. Total pages: {total_pages}")
            raise ValueError(f"Page index {pages} is out of range. Total pages: {total_pages}")
        page_indices = [pages]
    elif isinstance(pages, (list, tuple)):
        if isinstance(pages, tuple) and len(pages) == 2:
            start, end = pages
            if not (isinstance(start, int) and isinstance(end, int)):
                logging.error("Start and end values must be integers.")
                raise ValueError("Start and end values must be integers.")
            if start < 0 or end > total_pages or start >= end:
                logging.error("Invalid page range specified.")
                raise ValueError("Invalid page range specified.")
            page_indices = range(start, end)
        else:
            page_indices = []
            for p in pages:
                if not isinstance(p, int):
                    logging.error("Page indices must be integers.")
                    raise ValueError("Page indices must be integers.")
                if p < 0 or p >= total_pages:
                    logging.error(f"Page index {p} is out of range. Total pages: {total_pages}")
                    raise ValueError(f"Page index {p} is out of range. Total pages: {total_pages}")
                page_indices.append(p)
    else:
        logging.error("Parameter 'pages' must be an int, list, tuple, or None.")
        raise ValueError("Parameter 'pages' must be an int, list, tuple, or None.")

    # Extract text from the specified pages.
    for i in page_indices:
        logging.debug(f"Extracting text from page {i + 1}")
        page = doc.load_page(i)
        page_text = page.get_text()
        text += f"\n\n--- Page {i + 1} ---\n\n" + page_text + "\n|-|+++|-|\n"
    
    doc.close()
    logging.info("Completed text extraction.")
    return text


def _normalize_for_match(text: str) -> str:
    """
    Normalize text for matching: lowercase and replace non-alphanumerics with single spaces,
    preserving token boundaries. Multiple non-alphanumerics collapse to one space.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text)


def _compact_with_index_map(norm_text: str) -> tuple[str, list[int]]:
    """
    Build a compacted version of norm_text (remove spaces) and an index map from
    each compact character index to its corresponding index in norm_text.
    """
    compact_chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(norm_text):
        if ch != " ":
            compact_chars.append(ch)
            index_map.append(i)
    return "".join(compact_chars), index_map


def _canonical_alnum_lower(text: str) -> str:
    """Lowercase and remove all non-alphanumeric characters (including spaces/punctuations)."""
    s = str(text).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def _is_alphanumeric_combo(s: str) -> bool:
    """Return True if the term contains at least one letter and at least one digit."""
    has_alpha = any(c.isalpha() for c in s)
    has_digit = any(c.isdigit() for c in s)
    return has_alpha and has_digit


def _passes_alnum_boundary_rule(term_compact: str, norm_text: str, compact_index_map: list[int], start_c: int, end_c: int) -> bool:
    """
    For alphanumeric terms (mix of letters and digits), enforce boundary rule when
    matching in compact text: reject if the character immediately before or after
    the match (in the spaced, normalized text) is alphanumeric.

    This prevents partial-embedded matches such as 'rm920' within 'rm 92080'.
    """
    # Only enforce for alphanumeric combo terms
    if not _is_alphanumeric_combo(term_compact):
        return True

    if not compact_index_map:
        return True

    start_norm = compact_index_map[start_c]
    end_norm = compact_index_map[end_c]

    before_char = norm_text[start_norm - 1] if start_norm - 1 >= 0 else " "
    after_char = norm_text[end_norm + 1] if end_norm + 1 < len(norm_text) else " "

    if before_char.isalnum():
        return False
    if after_char.isalnum():
        return False
    return True


def _prepare_model_terms(df_models: pd.DataFrame, column: str = "modelNumber") -> List[str]:
    """
    Prepare a deduplicated, length-sorted list of normalized model terms from a DataFrame.

    - Drops null/empty values
    - Strips brand suffixes like "/SMC" if present
    - De-duplicates and sorts by length desc to prefer longer matches first
    """
    if column not in df_models.columns:
        raise ValueError(f"Expected column '{column}' in df_models")

    raw_values = (
        df_models[column]
        .astype(str)
        .map(lambda x: x.strip())
        .replace({"": pd.NA, "nan": pd.NA})
        .dropna()
    )

    cleaned: Set[str] = set()
    for value in raw_values:
        # Remove common brand suffix separators like '/SMC', '-SMC', ' SMC'
        value_wo_brand = re.sub(r"[\s\-/]*smc\s*$", "", value, flags=re.IGNORECASE)
        value_norm_spaced = _normalize_for_match(value_wo_brand).strip()
        if value_norm_spaced:
            cleaned.add(value_norm_spaced)
            # Add compact variant to allow delimiterless matches like "alpha200"
            compact = value_norm_spaced.replace(" ", "")
            if compact:
                cleaned.add(compact)

    # Return list sorted by token length descending to reduce substring ambiguities
    return sorted(cleaned, key=lambda s: (-len(s), s))


def _build_aho_automaton(terms: List[str]):
    """
    Build and return an Aho–Corasick automaton for the provided terms.
    Terms should already be normalized (spaced or compact variants included).
    """
    if ahocorasick is None:
        raise ImportError("pyahocorasick is required. Please install it: pip install pyahocorasick")
    automaton = ahocorasick.Automaton()
    for term in terms:
        if term:
            automaton.add_word(term, term)
    automaton.make_automaton()
    return automaton


def find_model_pages_in_pdf(
    pdf_path: str,
    df_models: pd.DataFrame,
    model_column: str = "modelNumber",
    max_pages: int | None = None,
    use_aho: bool = True,
) -> Dict[int, List[str]]:
    """
    Scan a PDF and return a mapping of page_index -> matched model terms.

    - Uses PyMuPDF text extraction per page
    - Normalizes both page text and model terms for robust matching
    - Returns 0-indexed page indices

    Parameters:
        pdf_path: Absolute path to the PDF file
        df_models: DataFrame with a column of model numbers
        model_column: Column name containing model numbers
        max_pages: Optional cap for pages to scan (useful for large PDFs)
    """
    logging.debug(f"Scanning PDF for model occurrences: {pdf_path}")
    doc = pymupdf.open(pdf_path)
    try:
        total_pages = doc.page_count
        if max_pages is not None:
            total_pages = min(total_pages, max_pages)

        model_terms = _prepare_model_terms(df_models, column=model_column)
        if not model_terms:
            logging.warning("No model terms provided after cleaning; returning empty result.")
            return {}

        matches: Dict[int, List[str]] = {}
        automaton = None
        if use_aho and ahocorasick is not None:
            try:
                automaton = _build_aho_automaton(model_terms)
            except Exception as e:
                logging.warning(f"Falling back to regex matcher due to Aho–Corasick init error: {e}")
                automaton = None

        for page_index in range(total_pages):
            page_text = doc.load_page(page_index).get_text() or ""
            page_norm = _normalize_for_match(page_text)
            page_compact, compact_index_map = _compact_with_index_map(page_norm)

            page_hits: List[str] = []
            if automaton is not None:
                # Run over spaced text
                for _, found in automaton.iter(page_norm):
                    page_hits.append(found)
                # Also run over compact text to catch delimiterless hits
                for end_idx, found in automaton.iter(page_compact):
                    start_idx = end_idx - len(found) + 1
                    if _passes_alnum_boundary_rule(found, page_norm, compact_index_map, start_idx, end_idx):
                        page_hits.append(found)
                # Deduplicate while preserving order
                seen_local = set()
                page_hits = [t for t in page_hits if not (t in seen_local or seen_local.add(t))]
                # Numeric-safe filter: digit-only terms must be contained in a single numeric token
                if page_hits:
                    filtered: List[str] = []
                    tokens = page_norm.split()
                    for t in page_hits:
                        if t.isdigit():
                            if any(tok.isdigit() and re.search(rf"\b{re.escape(t)}\b", tok) for tok in tokens):
                                filtered.append(t)
                        else:
                            filtered.append(t)
                    page_hits = filtered
            else:
                for term in model_terms:
                    token_pattern = rf"\b{re.escape(term)}\b"
                    if re.search(token_pattern, page_norm) is not None:
                        page_hits.append(term)
                    else:
                        # Allow compact matching for delimiterless text
                        term_compact = term.replace(" ", "")
                        for m in re.finditer(re.escape(term_compact), page_compact):
                            start_idx = m.start()
                            end_idx = m.end() - 1
                            # Numeric-safe: if term is digits only, ensure within one numeric token in spaced text
                            if term.isdigit():
                                if any(tok.isdigit() and re.search(rf"\b{re.escape(term)}\b", tok) for tok in page_norm.split()):
                                    page_hits.append(term)
                                    break
                            else:
                                if _passes_alnum_boundary_rule(term_compact, page_norm, compact_index_map, start_idx, end_idx):
                                    page_hits.append(term)
                                    break

            if page_hits:
                matches[page_index] = page_hits

        logging.debug(
            f"Completed scanning {pdf_path}. Found matches on {len(matches)} page(s)."
        )
        return matches
    finally:
        doc.close()


def find_model_pages_in_directory(
    directory_path: str,
    df_models: pd.DataFrame,
    model_column: str = "modelNumber",
    recursive: bool = True,
    max_pages_per_pdf: int | None = None,
    show_progress: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Dict[int, List[str]]]:
    """
    Scan all PDFs in a directory for model number occurrences.

    Returns mapping: pdf_path -> { page_index -> [matched_terms...] }

    Parameters:
        directory_path: Folder to scan for PDFs
        df_models: DataFrame containing model numbers
        model_column: Column in df_models for model numbers
        recursive: Recurse into subdirectories
        max_pages_per_pdf: Optional limit of pages per PDF to scan
        show_progress: Display a progress bar if available
        progress_callback: Optional callable receiving (current_index, total_count, pdf_path)
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")

    # 1) Collect all PDF paths first so we know total for progress reporting
    pdf_paths: List[str] = []
    if recursive:
        for root, _, files in os.walk(directory_path):
            for name in files:
                if name.lower().endswith(".pdf"):
                    pdf_paths.append(os.path.join(root, name))
    else:
        try:
            for name in os.listdir(directory_path):
                full_path = os.path.join(directory_path, name)
                if os.path.isfile(full_path) and name.lower().endswith(".pdf"):
                    pdf_paths.append(full_path)
        except FileNotFoundError:
            raise ValueError(f"Directory does not exist: {directory_path}")

    total_pdfs = len(pdf_paths)
    if total_pdfs == 0:
        logging.info("No PDF files found in directory: %s", directory_path)
        return {}

    # 2) Set up iterator with optional progress bar
    use_tqdm = False
    iterator = pdf_paths
    if show_progress:
        try:
            from tqdm.auto import tqdm  # type: ignore
            iterator = tqdm(pdf_paths, desc="Scanning PDFs", unit="file")
            use_tqdm = True
        except Exception:
            # Fallback: no tqdm available; proceed without a progress bar
            use_tqdm = False

    # 3) Scan each PDF and optionally report progress
    result: Dict[str, Dict[int, List[str]]] = {}
    for idx, pdf_path in enumerate(iterator, start=1):
        if progress_callback is not None:
            try:
                progress_callback(idx, total_pdfs, pdf_path)
            except Exception as cb_err:
                logging.warning("Progress callback failed: %s", cb_err)

        try:
            page_map = find_model_pages_in_pdf(
                pdf_path=pdf_path,
                df_models=df_models,
                model_column=model_column,
                max_pages=max_pages_per_pdf,
            )
            if page_map:
                result[pdf_path] = page_map
                if use_tqdm:
                    try:
                        # Show how many pages had matches for this PDF
                        matched_pages = len(page_map)
                        iterator.set_postfix({"matched_pages": matched_pages})  # type: ignore[attr-defined]
                    except Exception:
                        pass
        except Exception as e:
            logging.error("Failed scanning '%s': %s", pdf_path, e)

    return result


def save_model_page_matches_to_csv(
    matches: Dict[str, Dict[int, List[str]]],
    output_csv_path: str,
) -> None:
    """
    Save directory scan matches to a tidy CSV with columns:
    pdf_path, page_number_1_indexed, matched_term
    """
    rows = []
    for pdf_path, page_map in matches.items():
        for page_index, terms in page_map.items():
            for term in terms:
                rows.append(
                    {
                        "pdf_path": pdf_path,
                        "page_number": page_index + 1,
                        "matched_term": term,
                    }
                )
    df = pd.DataFrame(rows, columns=["pdf_path", "page_number", "matched_term"])
    df.to_csv(output_csv_path, index=False)


def find_model_matches_in_texts(
    df_texts: pd.DataFrame,
    text_column: str,
    df_models: pd.DataFrame,
    model_column: str = "modelNumber",
    use_aho: bool = True,
    longest_only: bool = False,
) -> pd.DataFrame:
    """
    Match model terms against free-text fields in a DataFrame using the same
    normalization and multi-strategy matching as the PDF scanner.

    Parameters:
        df_texts: DataFrame containing a text column to scan (e.g., product_info)
        text_column: Name of the text column in df_texts
        df_models: DataFrame containing model values (e.g., dedup_model_value)
        model_column: Column in df_models with the model values (default 'modelNumber')
        use_aho: If True and pyahocorasick is available, use Aho–Corasick for speed

    Returns:
        A copy of df_texts with two extra columns:
            - matched_terms: list[str] of matched model terms (canonicalized by removing spaces)
            - num_matches: int number of matches
    """
    if text_column not in df_texts.columns:
        raise ValueError(f"Expected column '{text_column}' in df_texts")
    if model_column not in df_models.columns:
        raise ValueError(f"Expected column '{model_column}' in df_models")

    model_terms = _prepare_model_terms(df_models, column=model_column)
    if not model_terms:
        logging.warning("No model terms provided after cleaning; returning input with empty matches.")
        result = df_texts.copy()
        result["matched_terms"] = [[] for _ in range(len(result))]
        result["num_matches"] = 0
        return result

    automaton = None
    if use_aho and ahocorasick is not None:
        try:
            automaton = _build_aho_automaton(model_terms)
        except Exception as e:
            logging.warning(f"Falling back to regex matcher due to Aho–Corasick init error: {e}")
            automaton = None

    # Build canonical term set for exact token matching (space boundaries only)
    # We disregard punctuation by canonicalizing to alphanumeric lowercase.
    canonical_term_to_originals: Dict[str, List[str]] = {}
    max_canonical_len = 0
    for term in model_terms:
        canon = _canonical_alnum_lower(term)
        if not canon:
            continue
        canonical_term_to_originals.setdefault(canon, []).append(term)
        if len(canon) > max_canonical_len:
            max_canonical_len = len(canon)

    matched_lists: List[List[str]] = []
    for _, row in df_texts.iterrows():
        text_value = row.get(text_column, "")
        # Tokenize strictly by whitespace, but compare canonicalized (alnum-only, lowercase)
        # This enforces space boundaries and ignores punctuation for comparisons.
        # Example: "REM RE4431 433" → tokens: ["REM","RE4431","433"], and
        # canonical("RE4431-AF") => "re4431af" which won't match token "re4431".
        tokens_raw = str(text_value).split()
        tokens_canonical: List[str] = [_canonical_alnum_lower(tok) for tok in tokens_raw]

        hits: List[str] = []
        if tokens_canonical:
            token_set = set(tok for tok in tokens_canonical if tok)
            # 1) Exact token matches for any term
            for tok in token_set:
                originals = canonical_term_to_originals.get(tok)
                if originals:
                    hits.extend(originals)

            # 2) Across-space/punctuation matches only for alphanumeric-combo terms
            if max_canonical_len > 0 and len(tokens_canonical) > 1:
                num_tokens = len(tokens_canonical)
                for i in range(num_tokens):
                    if not tokens_canonical[i]:
                        continue
                    concatenated = tokens_canonical[i]
                    current_len = len(concatenated)
                    # Extend to subsequent tokens up to max_canonical_len
                    for j in range(i + 1, num_tokens):
                        part = tokens_canonical[j]
                        if not part:
                            continue
                        new_len = current_len + len(part)
                        if new_len > max_canonical_len:
                            break
                        concatenated = concatenated + part
                        current_len = new_len
                        # Only allow across-token matches for alphanumeric-combo patterns
                        if concatenated in canonical_term_to_originals and _is_alphanumeric_combo(concatenated):
                            hits.extend(canonical_term_to_originals[concatenated])

            # 3) Compact partial match within a single token for alphanumeric terms
            # Allow a match if the token starts with the model term (ignoring punctuation inside the term),
            # regardless of whether the next character is alphanumeric or not (i.e., allow embedded prefix).
            for raw_tok in tokens_raw:
                if not raw_tok:
                    continue
                raw_lower = str(raw_tok).lower()
                canonical_acc = ""
                last_idx = -1
                for idx, ch in enumerate(raw_lower):
                    if ch.isalnum():
                        canonical_acc += ch
                        last_idx = idx
                        if len(canonical_acc) > max_canonical_len:
                            break
                        if canonical_acc in canonical_term_to_originals and _is_alphanumeric_combo(canonical_acc):
                            hits.extend(canonical_term_to_originals[canonical_acc])
                    else:
                        # skip punctuation inside the token
                        continue

        # Canonicalize by returning compact representation (remove spaces)
        seen_compact: Set[str] = set()
        canonical_hits: List[str] = []
        for t in hits:
            key = t.replace(" ", "")
            if key and key not in seen_compact:
                seen_compact.add(key)
                canonical_hits.append(key)

        if longest_only and canonical_hits:
            # Keep only the single longest canonical match
            longest = max(canonical_hits, key=len)
            matched_lists.append([longest])
        else:
            matched_lists.append(canonical_hits)

    result = df_texts.copy()
    result["matched_terms"] = matched_lists
    result["num_matches"] = result["matched_terms"].map(lambda x: len(x) if isinstance(x, list) else 0)
    return result

def select_pdf_file():
    """
    Opens a file dialog for the user to select a PDF file.

    Returns:
        str: The path to the selected PDF file, or an empty string if no file was selected.
    """
    logging.info("Opening file selection dialog.")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    pdf_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF Files", "*.pdf")]
    )
    
    root.destroy()
    
    if pdf_path:
        logging.info(f"Selected PDF file: {pdf_path}")
    else:
        logging.info("No PDF file was selected.")
    return pdf_path

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



def compare_table_headers(headers1, headers2):
    """
    Compare two lists of table headers and return True if they are the same.
    
    Parameters:
        headers1 (list): First list of table headers to compare
        headers2 (list): Second list of table headers to compare
        
    Returns:
        bool: True if the headers match, False otherwise
    """
    logging.debug(f"Comparing table headers:\n{headers1}\n{headers2}")
    if len(headers1) != len(headers2):
        logging.debug("Header length mismatch.")
        return False
    
    same = all(h1.strip() == h2.strip() for h1, h2 in zip(headers1, headers2))
    logging.debug(f"Headers are the same: {same}")
    return same

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

    # num_tables2 = output2.num_tables
    # headers2 = output2.table_headers_and_positions
    # columns2 = getattr(output2, "columns_per_table", None)

    # if compare_table_headers(headers1, headers2) or (num_tables1 == num_tables2 and num_tables1 is not None):
    #     logging.info("Initial table info match or same table count. Returning first attempt's result.")
    #     return num_tables1, headers1, columns1, 0 # 0 indicates the highest confidence. The higher the number, the lower the confidence. 
    return num_tables1, headers1, columns1, 0

    # Create third task if needed
    # async with asyncio.TaskGroup() as tg:
    #     task3 = tg.create_task(async_pattern_desc())
    # output3 = await task3
    # logging.debug(f"LLM attempt 3 output:\n{output3}")

    # num_tables3 = output3.num_tables
    # headers3 = output3.table_headers_and_positions
    # columns3 = getattr(output3, "columns_per_table", None)

    # logging.debug(f"headers3: {headers3}")
    # logging.debug(f"num_tables3: {num_tables3}")

    # if compare_table_headers(headers3, headers1) or (num_tables3 == num_tables1 and num_tables3 is not None):
    #     logging.info("Majority match found with first and third results.")
    #     return num_tables1, headers1, columns1, 1
    
    # if compare_table_headers(headers3, headers2) or (num_tables3 == num_tables2 and num_tables3 is not None):
    #     logging.info("Majority match found with second and third results.")
    #     return num_tables2, headers2, columns2, 1

    # logging.warning("No matches found. Returning third run results for table_headers.")
    # return num_tables3, headers3, columns3, 2

def compare_column_data(data1, data2):
    """
    Compare two sets of column data results and return (bool, issue_table_headers).
    If mismatch occurs, return which table headers encountered an issue.
    
    Parameters:
        data1 (list): First set of column data to compare
        data2 (list): Second set of column data to compare
        
    Returns:
        tuple: (match_found, issue_table_headers)
            - match_found (bool): True if the column data matches, False otherwise
            - issue_table_headers (list): List of table headers that had issues
    """
    logging.debug("Comparing column data for consistency.")
    issue_table_headers = []

    if len(data1) != len(data2):
        logging.warning("Column data length mismatch")
        return False, issue_table_headers
    
    data1_sorted = sorted(data1, key=lambda x: x["index"])
    data2_sorted = sorted(data2, key=lambda x: x["index"])
    
    for item1, item2 in zip(data1_sorted, data2_sorted):
        # Compare column names for the same index
        if set(item1["column_names"]) != set(item2["column_names"]):
            logging.warning(f"Column names mismatch for index {item1['index']}")
            logging.warning(f"Set 1: {item1['column_names']}")
            logging.warning(f"Set 2: {item2['column_names']}")
            issue_table_headers.append(item1["table_header"])
            return False, issue_table_headers
            
    logging.debug("Column data match found.")
    return True, issue_table_headers

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



def parse_variable_data_to_df(text):
    """
    Parse variable data text into a pandas DataFrame.
    
    Parameters:
        text (str): Text containing variable data in the format [key:value]
        
    Returns:
        pandas.DataFrame: DataFrame containing the parsed variable data
    """
    logging.info("Parsing variable data into DataFrame.")
    pattern = r"\[([^\]:]+):([^]]+)\]"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    data = {}
    max_len = 0

    for key, val in matches:
        items = [item.replace("***", "").strip() for item in val.split("|-|")]
        data[key.strip()] = items
        max_len = max(max_len, len(items))

    df = pd.DataFrame({
        col: values + [None]*(max_len - len(values))
        for col, values in data.items()
    })

    logging.debug(f"Variable data DataFrame shape: {df.shape}")
    return df

def extract_df_from_string(text):
    """
    Extracts a DataFrame from a string that contains a Python list/dict-like structure.
    
    Parameters:
        text (str): String containing a Python list/dict-like structure
        
    Returns:
        pandas.DataFrame: DataFrame created from the extracted data structure
        
    Raises:
        ValueError: If no tables can be extracted from the string
    """
    logging.debug("Extracting DataFrame from string representation.")
    match = re.search(r'(\[.*\])', text, re.DOTALL)
    if match:
        data = ast.literal_eval(match.group(1))
        df = pd.DataFrame(data)
        logging.debug(f"Extracted DataFrame shape: {df.shape}")
        return df
    logging.error("No tables extracted from the string.")
    raise ValueError("No tables extracted from the page")


def rows_to_df(rows):
    # rows is like [Row_0(...), Row_0(...), ...]
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



# Helper functions for notebook usage

def parse_page_selection(total_pages: int, pages=None, page_range=None):
    """Translate page selection into a list of zero-indexed page numbers."""
    
    if pages:
        if isinstance(pages, str):
            page_nums = [int(p.strip()) for p in pages.split(",") if p.strip()]
        else:
            page_nums = pages if isinstance(pages, list) else [pages]
        validated = [p for p in page_nums if 1 <= p <= total_pages]
        if not validated:
            raise ValueError("No valid page numbers supplied.")
        return [p - 1 for p in validated]

    if page_range:
        start, end = page_range
        if not (1 <= start <= end <= total_pages):
            raise ValueError("Invalid range values. Ensure 1 <= start <= end <= total_pages")
        return list(range(start - 1, end))

    # default: all pages
    return list(range(total_pages))


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
    page_indices = parse_page_selection(doc.page_count, pages, page_range)
    logging.info("Processing %d page(s): %s", len(page_indices), [i + 1 for i in page_indices])

    results_output = []

    async def process_one_page(page_no: int):
        async with PAGE_SEMAPHORE:
            page = doc.load_page(page_no)
            if not table_in_image:
                if len(page.find_tables().tables) == 0:
                    logging.info("Page %d: no tables detected via PyMuPDF, skipping.", page_no + 1)
                    return None

            # Run text and image extraction concurrently
            extracted_text, base64_image = await asyncio.gather(
                asyncio.to_thread(get_page_text_thread, pdf_path, page_no),
                asyncio.to_thread(get_page_pixel_data, pdf_path, page_no, 300, "png"),
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


# Convenience wrapper for notebook usage
async def extract_pdf_tables(pdf_path, **kwargs):
    """Async wrapper for the main extraction function - use with await in notebooks."""
    return await extract_tables_from_pdf(pdf_path, **kwargs)




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






# ---------------------------
# PDF page export utilities
# ---------------------------

def _coalesce_page_indices_to_ranges(page_indices: Iterable[int]) -> List[Tuple[int, int]]:
    """
    Convert a collection of 0-indexed page indices into inclusive ranges
    suitable for PyMuPDF's insert_pdf (from_page/to_page inclusive).

    Example: [0,1,2, 4,5, 9] -> [(0,2), (4,5), (9,9)]
    """
    unique_sorted = sorted(set(int(i) for i in page_indices))
    if not unique_sorted:
        return []

    ranges: List[Tuple[int, int]] = []
    start = unique_sorted[0]
    prev = start

    for idx in unique_sorted[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = idx
        prev = idx
    ranges.append((start, prev))
    return ranges


def extract_pages_to_pdf(pdf_path: str, page_indices: Iterable[int], output_pdf_path: str) -> str:
    """
    Extract specific 0-indexed pages from a single PDF and save them into a new PDF.

    Args:
        pdf_path: Absolute or relative path to the source PDF.
        page_indices: Iterable of 0-indexed page numbers to extract.
        output_pdf_path: Where to write the new PDF.

    Returns:
        The output PDF path.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    ranges = _coalesce_page_indices_to_ranges(page_indices)
    if not ranges:
        raise ValueError("No valid page indices provided.")

    out_dir = os.path.dirname(output_pdf_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    src = pymupdf.open(pdf_path)
    try:
        total_pages = src.page_count
        for start, end in ranges:
            if start < 0 or end >= total_pages:
                raise ValueError(
                    f"Page range ({start}, {end}) out of bounds for document with {total_pages} pages"
                )
        dest = pymupdf.open()
        try:
            for start, end in ranges:
                dest.insert_pdf(src, from_page=start, to_page=end)
            dest.save(output_pdf_path)
        finally:
            dest.close()
    finally:
        src.close()

    logging.info("Saved extracted pages to %s", output_pdf_path)
    return output_pdf_path


def export_directory_matches_to_pdf(
    matches: Dict[str, Dict[int, List[str]]],
    output_pdf_path: str,
) -> str:
    """
    Combine matched pages across many PDFs (as returned by find_model_pages_in_directory)
    into a single PDF in ascending file and page order.

    Args:
        matches: Mapping of pdf_path -> { page_index (0-based) -> [matched_terms...] }
        output_pdf_path: Where to save the combined PDF

    Returns:
        The output PDF path.
    """
    if not matches:
        raise ValueError("'matches' is empty; nothing to export.")

    out_dir = os.path.dirname(output_pdf_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    combined = pymupdf.open()
    try:
        for pdf_path in sorted(matches.keys()):
            page_map = matches[pdf_path]
            if not page_map:
                continue
            # Coalesce indices for fewer insert operations
            page_indices = sorted(set(int(i) for i in page_map.keys()))
            ranges = _coalesce_page_indices_to_ranges(page_indices)

            if not os.path.exists(pdf_path):
                logging.warning("Skipping missing PDF: %s", pdf_path)
                continue

            src = pymupdf.open(pdf_path)
            try:
                total = src.page_count
                for start, end in ranges:
                    if start < 0 or end >= total:
                        logging.warning(
                            "Skipping out-of-bounds range (%d,%d) for '%s' with %d pages",
                            start,
                            end,
                            pdf_path,
                            total,
                        )
                        continue
                    combined.insert_pdf(src, from_page=start, to_page=end)
            finally:
                src.close()

        if combined.page_count == 0:
            raise ValueError("No pages were added to the output PDF.")

        combined.save(output_pdf_path)
    finally:
        combined.close()

    logging.info("Saved combined matched pages to %s", output_pdf_path)
    return output_pdf_path


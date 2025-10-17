"""Pytest configuration and shared fixtures."""
import pytest
import asyncio
import pandas as pd
from unittest.mock import MagicMock, AsyncMock
from openai import AsyncOpenAI
import base64
import os


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a temporary PDF file path for testing."""
    pdf_file = tmp_path / "test.pdf"
    # Create a simple mock PDF file
    pdf_file.write_bytes(b"%PDF-1.4\n%Test PDF\n")
    return str(pdf_file)


@pytest.fixture
def sample_base64_image():
    """Return a sample base64-encoded image string."""
    # A 1x1 pixel transparent PNG
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    return base64.b64encode(png_bytes).decode('utf-8')


@pytest.fixture
def sample_dataframe():
    """Return a sample DataFrame for testing."""
    return pd.DataFrame({
        'Column1': ['A', 'B', 'C'],
        'Column2': [1, 2, 3],
        'Column3': ['X', 'Y', 'Z']
    })


@pytest.fixture
def sample_dataframes_list(sample_dataframe):
    """Return a list of sample DataFrames."""
    df2 = pd.DataFrame({
        'Col1': ['D', 'E', 'F'],
        'Col2': [4, 5, 6]
    })
    return [sample_dataframe, df2]


@pytest.fixture
def mock_openai_client():
    """Create a mock AsyncOpenAI client."""
    client = MagicMock(spec=AsyncOpenAI)
    client.beta = MagicMock()
    client.beta.chat = MagicMock()
    client.beta.chat.completions = MagicMock()
    return client


@pytest.fixture
def mock_table_info_response():
    """Create a mock TableInfo response."""
    mock_response = MagicMock()
    mock_response.num_tables = 2
    mock_response.table_headers_and_positions = [
        "Table 1 -- Located at top of page",
        "Table 2 -- Located at bottom of page"
    ]
    mock_response.columns_per_table = [
        ["Column A", "Column B", "Column C"],
        ["Column X", "Column Y"]
    ]
    return mock_response


@pytest.fixture
def mock_parsed_response():
    """Create a mock parsed response from OpenAI."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message = MagicMock()
    mock.choices[0].message.parsed = MagicMock()
    return mock


@pytest.fixture
def sample_extracted_text():
    """Return sample extracted text from a PDF."""
    return """
    Table: Product Information
    Product Name    Price    Quantity
    Widget A        $10.99   100
    Widget B        $15.99   50
    Widget C        $20.99   75
    """


@pytest.fixture
def sample_output_final(sample_dataframes_list):
    """Return a sample output_final structure (list of list of DataFrames)."""
    return [sample_dataframes_list]


@pytest.fixture
def mock_pymupdf_document():
    """Create a mock PyMuPDF document."""
    mock_doc = MagicMock()
    mock_doc.page_count = 5
    
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Sample text"
    
    # Mock find_tables
    mock_tables = MagicMock()
    mock_tables.tables = [MagicMock()]
    mock_page.find_tables.return_value = mock_tables
    
    # Mock pixmap
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"fake image data"
    mock_page.get_pixmap.return_value = mock_pix
    
    mock_doc.load_page.return_value = mock_page
    mock_doc.__getitem__ = lambda self, idx: mock_page
    
    return mock_doc


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output_files"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, tmp_path):
    """Setup test environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-123")
    monkeypatch.setenv("MAX_CONCURRENCY", "4")


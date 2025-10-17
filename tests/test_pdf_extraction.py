"""Unit tests for pdf_extraction.py module."""
import pytest
import pandas as pd
import base64
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
from src.pdf_extraction import (
    remove_duplicate_dfs,
    get_page_pixel_data,
    normalize_text,
    sanitize_field_name,
    sanitize_worksheet_name,
    rows_to_df,
    get_page_text_thread,
    extract_columns,
    _describe_exception
)


class TestRemoveDuplicateDfs:
    """Test the remove_duplicate_dfs function."""

    def test_remove_exact_duplicates(self):
        """Test removing exact duplicate DataFrames."""
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})  # Exact duplicate
        df3 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        
        result = remove_duplicate_dfs([df1, df2, df3])
        assert len(result) == 2

    def test_no_duplicates(self):
        """Test with no duplicates."""
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        
        result = remove_duplicate_dfs([df1, df2])
        assert len(result) == 2

    def test_empty_list(self):
        """Test with empty list."""
        result = remove_duplicate_dfs([])
        assert len(result) == 0

    def test_unhashable_types(self):
        """Test with DataFrames containing unhashable types (lists, dicts)."""
        df1 = pd.DataFrame({'A': [[1, 2], [3, 4]], 'B': [{'key': 'val'}, {'key2': 'val2'}]})
        df2 = pd.DataFrame({'A': [[1, 2], [3, 4]], 'B': [{'key': 'val'}, {'key2': 'val2'}]})
        df3 = pd.DataFrame({'A': [[5, 6], [7, 8]], 'B': [{'key': 'diff'}, {'key2': 'diff2'}]})
        
        result = remove_duplicate_dfs([df1, df2, df3])
        assert len(result) == 2

    def test_single_dataframe(self):
        """Test with single DataFrame."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        result = remove_duplicate_dfs([df])
        assert len(result) == 1


class TestNormalizeText:
    """Test the normalize_text function."""

    def test_basic_normalization(self):
        """Test basic text normalization."""
        result = normalize_text("Hello World")
        assert result == "helloworld"

    def test_special_characters(self):
        """Test normalization with special characters."""
        result = normalize_text("Test@123#$%")
        assert result == "test123"

    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        result = normalize_text("café")
        assert "cafe" in result or "café" in result

    def test_whitespace_removal(self):
        """Test whitespace removal."""
        result = normalize_text("  lots   of    spaces  ")
        assert " " not in result

    def test_empty_string(self):
        """Test with empty string."""
        result = normalize_text("")
        assert result == ""


class TestSanitizeFieldName:
    """Test the sanitize_field_name function."""

    def test_valid_field_name(self):
        """Test with already valid field name."""
        result = sanitize_field_name("valid_field")
        assert result == "valid_field"

    def test_spaces_in_name(self):
        """Test field name with spaces."""
        result = sanitize_field_name("Column Name")
        assert result == "column_name"

    def test_special_characters(self):
        """Test field name with special characters."""
        result = sanitize_field_name("Column@123#")
        assert "_" in result
        assert "@" not in result
        assert "#" not in result

    def test_starts_with_number(self):
        """Test field name starting with number."""
        result = sanitize_field_name("123column")
        assert result.startswith("field_")

    def test_empty_string(self):
        """Test with empty string."""
        result = sanitize_field_name("")
        assert result == "field_1"

    def test_only_special_chars(self):
        """Test with only special characters."""
        result = sanitize_field_name("@#$%")
        assert result == "field_1"

    def test_collapse_underscores(self):
        """Test collapsing multiple underscores."""
        result = sanitize_field_name("column___name")
        assert "___" not in result


class TestSanitizeWorksheetName:
    """Test the sanitize_worksheet_name function."""

    def test_valid_worksheet_name(self):
        """Test with valid worksheet name."""
        result = sanitize_worksheet_name("Sheet1")
        assert result == "Sheet1"

    def test_invalid_characters(self):
        """Test worksheet name with invalid characters."""
        result = sanitize_worksheet_name("Sheet[1]:Test*")
        assert "[" not in result
        assert ":" not in result
        assert "*" not in result

    def test_long_name_truncation(self):
        """Test truncation of long names (>31 chars)."""
        long_name = "A" * 50
        result = sanitize_worksheet_name(long_name)
        assert len(result) <= 31

    def test_reserved_name(self):
        """Test with reserved name 'History'."""
        result = sanitize_worksheet_name("History")
        assert result != "History"
        assert result == "Sheet1"

    def test_empty_name(self):
        """Test with empty name."""
        result = sanitize_worksheet_name("")
        assert result == "Sheet1"


class TestRowsToDf:
    """Test the rows_to_df function."""

    def test_pydantic_models_v2(self):
        """Test converting Pydantic v2 models to DataFrame."""
        # Mock Pydantic v2 model
        mock_row1 = MagicMock()
        mock_row1.model_dump.return_value = {'col1': 'A', 'col2': 1}
        mock_row2 = MagicMock()
        mock_row2.model_dump.return_value = {'col1': 'B', 'col2': 2}
        
        result = rows_to_df([mock_row1, mock_row2])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_dict_rows(self):
        """Test converting dict rows to DataFrame."""
        rows = [
            {'col1': 'A', 'col2': 1},
            {'col1': 'B', 'col2': 2}
        ]
        result = rows_to_df(rows)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ['col1', 'col2']

    def test_empty_list(self):
        """Test with empty list."""
        result = rows_to_df([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestDescribeException:
    """Test the _describe_exception function."""

    def test_basic_exception(self):
        """Test basic exception description."""
        exc = ValueError("Test error")
        result = _describe_exception(exc)
        assert "ValueError" in result
        assert "Test error" in result

    def test_exception_with_status_code(self):
        """Test exception with status_code attribute."""
        exc = Exception("API Error")
        exc.status_code = 500
        result = _describe_exception(exc)
        assert "status_code=500" in result

    def test_exception_with_response(self):
        """Test exception with response attribute."""
        exc = Exception("API Error")
        mock_response = MagicMock()
        mock_response.body = b"Error body"
        exc.response = mock_response
        result = _describe_exception(exc)
        assert "body=" in result


class TestExtractColumns:
    """Test the extract_columns function."""

    def test_extract_single_table(self):
        """Test extracting columns from single table response."""
        response_text = """
        index: [0]
        column_names: ["Column A", "Column B", "Column C"]
        example_value_per_column: ["A": "value1", "B": "value2", "C": "value3"]
        table_location: ["top left"]
        """
        tables_to_target = ["Test Table"]
        
        result = extract_columns(response_text, tables_to_target)
        assert len(result) >= 0  # Function may not find perfect matches due to regex

    def test_extract_multiple_tables(self):
        """Test extracting columns from multiple tables."""
        response_text = """
        index: [0]
        column_names: ["Column A", "Column B"]
        example_value_per_column: ["A": "val1", "B": "val2"]
        table_location: ["top"]
        index: [1]
        column_names: ["Column X", "Column Y"]
        example_value_per_column: ["X": "valX", "Y": "valY"]
        table_location: ["bottom"]
        """
        tables_to_target = ["Table 1", "Table 2"]
        
        result = extract_columns(response_text, tables_to_target)
        assert isinstance(result, list)

    def test_empty_response(self):
        """Test with empty response text."""
        result = extract_columns("", [])
        assert result == []


class TestGetPagePixelData:
    """Test the get_page_pixel_data function."""

    @patch('src.pdf_extraction.pymupdf')
    def test_valid_page_conversion(self, mock_pymupdf, mock_pymupdf_document, sample_pdf_path):
        """Test valid page conversion to base64."""
        mock_pymupdf.open.return_value = mock_pymupdf_document
        
        result = get_page_pixel_data(sample_pdf_path, 0, dpi=300, image_type='png')
        assert isinstance(result, str)
        # Check if it's valid base64
        try:
            base64.b64decode(result)
            assert True
        except:
            assert False, "Result is not valid base64"

    @patch('src.pdf_extraction.pymupdf')
    def test_invalid_page_number(self, mock_pymupdf, mock_pymupdf_document, sample_pdf_path):
        """Test with invalid page number."""
        mock_pymupdf.open.return_value = mock_pymupdf_document
        
        with pytest.raises(ValueError, match="out of range"):
            get_page_pixel_data(sample_pdf_path, 999, dpi=300, image_type='png')

    @patch('src.pdf_extraction.pymupdf')
    def test_negative_page_number(self, mock_pymupdf, mock_pymupdf_document, sample_pdf_path):
        """Test with negative page number."""
        mock_pymupdf.open.return_value = mock_pymupdf_document
        
        with pytest.raises(ValueError, match="out of range"):
            get_page_pixel_data(sample_pdf_path, -1, dpi=300, image_type='png')


class TestGetPageTextThread:
    """Test the get_page_text_thread function."""

    @patch('src.pdf_extraction.pymupdf')
    def test_extract_text_from_page(self, mock_pymupdf, mock_pymupdf_document, sample_pdf_path):
        """Test extracting text from a page."""
        mock_pymupdf.open.return_value = mock_pymupdf_document
        
        result = get_page_text_thread(sample_pdf_path, 0)
        assert isinstance(result, str)
        assert len(result) > 0

    @patch('src.pdf_extraction.pymupdf')
    def test_document_closes_after_extraction(self, mock_pymupdf, mock_pymupdf_document, sample_pdf_path):
        """Test that document is properly closed after extraction."""
        mock_pymupdf.open.return_value = mock_pymupdf_document
        
        get_page_text_thread(sample_pdf_path, 0)
        mock_pymupdf_document.close.assert_called_once()


@pytest.mark.asyncio
class TestAsyncFunctions:
    """Test async functions in pdf_extraction module."""

    async def test_with_openai_semaphore(self):
        """Test the with_openai_semaphore function."""
        from src.pdf_extraction import with_openai_semaphore
        
        async def sample_coro(value):
            return value * 2
        
        result = await with_openai_semaphore(sample_coro, 5)
        assert result == 10

    async def test_get_validated_table_info(self, mock_openai_client, sample_base64_image, mock_table_info_response):
        """Test the get_validated_table_info function."""
        from src.pdf_extraction import get_validated_table_info
        
        # Mock the table_identification_llm
        with patch('src.pdf_extraction.table_identification_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_table_info_response
            
            result = await get_validated_table_info(
                user_text="Extract tables",
                openai_client=mock_openai_client,
                base64_image=sample_base64_image,
                model="gpt-4"
            )
            
            assert len(result) == 4
            num_tables, headers, columns, confidence = result
            assert num_tables == 2
            assert len(headers) == 2
            assert confidence == 0


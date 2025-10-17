"""Unit tests for main.py module."""
import pytest
import argparse
from unittest.mock import MagicMock, AsyncMock, patch
from src.main import (
    parse_page_selection,
    build_arg_parser,
)


class TestParsePageSelection:
    """Test the parse_page_selection function."""

    def test_all_pages_default(self):
        """Test default behavior - all pages."""
        args = argparse.Namespace(pages=None, range=None)
        result = parse_page_selection(10, args)
        assert result == list(range(10))

    def test_specific_pages(self):
        """Test selecting specific pages."""
        args = argparse.Namespace(pages="1,3,5", range=None)
        result = parse_page_selection(10, args)
        assert result == [0, 2, 4]  # 0-indexed

    def test_page_range(self):
        """Test selecting a page range."""
        args = argparse.Namespace(pages=None, range=[2, 5])
        result = parse_page_selection(10, args)
        assert result == [1, 2, 3, 4]  # 0-indexed, inclusive range

    def test_invalid_pages_filtered(self):
        """Test that invalid page numbers are filtered out."""
        args = argparse.Namespace(pages="1,15,20", range=None)
        result = parse_page_selection(10, args)
        assert result == [0]  # Only page 1 is valid

    def test_no_valid_pages_error(self):
        """Test error when no valid pages supplied."""
        args = argparse.Namespace(pages="100,200", range=None)
        with pytest.raises(ValueError, match="No valid page numbers"):
            parse_page_selection(10, args)

    def test_invalid_range_error(self):
        """Test error with invalid range."""
        args = argparse.Namespace(pages=None, range=[10, 5])  # End before start
        with pytest.raises(ValueError, match="Invalid --range values"):
            parse_page_selection(10, args)

    def test_range_exceeds_total_pages(self):
        """Test error when range exceeds total pages."""
        args = argparse.Namespace(pages=None, range=[1, 20])
        with pytest.raises(ValueError, match="Invalid --range values"):
            parse_page_selection(10, args)

    def test_pages_with_whitespace(self):
        """Test parsing pages with extra whitespace."""
        args = argparse.Namespace(pages=" 1 , 2 , 3 ", range=None)
        result = parse_page_selection(10, args)
        assert result == [0, 1, 2]

    def test_single_page(self):
        """Test selecting a single page."""
        args = argparse.Namespace(pages="5", range=None)
        result = parse_page_selection(10, args)
        assert result == [4]


class TestBuildArgParser:
    """Test the build_arg_parser function."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = build_arg_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_required_pdf_path(self):
        """Test that pdf_path is required."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])  # No arguments

    def test_default_values(self):
        """Test default argument values."""
        parser = build_arg_parser()
        args = parser.parse_args(["test.pdf"])
        
        assert args.pdf_path == "test.pdf"
        assert args.output_name == "output_file"
        assert args.format == "xlsx"
        assert args.style == "sheet"
        assert args.instructions == "Extract all data from the table(s)"
        assert args.table_in_image is False
        assert args.add_table_info is False
        assert args.model == "gpt-4.1"
        assert args.vision_model == "gpt-4.1"

    def test_output_format_choices(self):
        """Test output format choices."""
        parser = build_arg_parser()
        
        # Valid choice
        args = parser.parse_args(["test.pdf", "--format", "csv"])
        assert args.format == "csv"
        
        # Invalid choice should raise error
        with pytest.raises(SystemExit):
            parser.parse_args(["test.pdf", "--format", "json"])

    def test_style_choices(self):
        """Test style choices."""
        parser = build_arg_parser()
        
        for style in ["concatenated", "by_page", "sheet"]:
            args = parser.parse_args(["test.pdf", "--style", style])
            assert args.style == style

    def test_page_range_parsing(self):
        """Test page range parsing."""
        parser = build_arg_parser()
        args = parser.parse_args(["test.pdf", "--range", "1", "10"])
        assert args.range == [1, 10]

    def test_pages_parsing(self):
        """Test individual pages parsing."""
        parser = build_arg_parser()
        args = parser.parse_args(["test.pdf", "--pages", "1,2,3,5,8"])
        assert args.pages == "1,2,3,5,8"

    def test_mutually_exclusive_range_and_pages(self):
        """Test that range and pages are mutually exclusive."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["test.pdf", "--range", "1", "5", "--pages", "1,2,3"])

    def test_custom_instructions(self):
        """Test custom instructions."""
        parser = build_arg_parser()
        custom_text = "Extract only the price column"
        args = parser.parse_args(["test.pdf", "--instructions", custom_text])
        assert args.instructions == custom_text

    def test_table_in_image_flag(self):
        """Test table-in-image flag."""
        parser = build_arg_parser()
        args = parser.parse_args(["test.pdf", "--table-in-image"])
        assert args.table_in_image is True

    def test_add_table_info_flag(self):
        """Test add-table-info flag."""
        parser = build_arg_parser()
        args = parser.parse_args(["test.pdf", "--add-table-info"])
        assert args.add_table_info is True

    def test_custom_model(self):
        """Test custom model selection."""
        parser = build_arg_parser()
        args = parser.parse_args(["test.pdf", "--model", "gpt-4-turbo"])
        assert args.model == "gpt-4-turbo"

    def test_custom_vision_model(self):
        """Test custom vision model selection."""
        parser = build_arg_parser()
        args = parser.parse_args(["test.pdf", "--vision-model", "gpt-4-vision"])
        assert args.vision_model == "gpt-4-vision"

    def test_output_name_custom(self):
        """Test custom output name."""
        parser = build_arg_parser()
        args = parser.parse_args(["test.pdf", "-o", "my_output"])
        assert args.output_name == "my_output"


@pytest.mark.asyncio
class TestProcessPdf:
    """Test the process_pdf async function."""

    @patch('src.main.pymupdf')
    @patch('src.main.os.getenv')
    async def test_process_pdf_no_api_key(self, mock_getenv, mock_pymupdf):
        """Test process_pdf raises error when API key is missing."""
        from src.main import process_pdf
        
        mock_getenv.return_value = None
        args = argparse.Namespace(
            pdf_path="test.pdf",
            pages=None,
            range=None,
            table_in_image=False,
            instructions="Extract tables",
            model="gpt-4",
            vision_model="gpt-4",
            add_table_info=False
        )
        
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            await process_pdf(args)

    @patch('src.main.pymupdf')
    @patch('src.main.AsyncOpenAI')
    @patch('src.main.os.getenv')
    async def test_process_pdf_basic_flow(self, mock_getenv, mock_openai, mock_pymupdf, mock_pymupdf_document):
        """Test basic process_pdf flow."""
        from src.main import process_pdf
        
        mock_getenv.return_value = "test-api-key"
        mock_pymupdf.open.return_value = mock_pymupdf_document
        
        # Mock the async functions
        with patch('src.main.get_page_pixel_data', return_value="base64image"), \
             patch('src.main.get_validated_table_info', new_callable=AsyncMock) as mock_validate, \
             patch('src.main.process_tables_to_df', new_callable=AsyncMock) as mock_process:
            
            mock_validate.return_value = (0, [], None, 0)  # No tables found
            
            args = argparse.Namespace(
                pdf_path="test.pdf",
                pages=None,
                range=None,
                table_in_image=False,
                instructions="Extract tables",
                model="gpt-4",
                vision_model="gpt-4",
                add_table_info=False
            )
            
            result = await process_pdf(args)
            assert isinstance(result, list)


class TestMainFunction:
    """Test the main function."""

    @patch('src.main.asyncio.run')
    @patch('src.main.write_output_final')
    @patch('src.main.os.makedirs')
    def test_main_with_xlsx_output(self, mock_makedirs, mock_write, mock_asyncio_run, sample_dataframes_list):
        """Test main function with Excel output."""
        from src.main import main
        
        mock_asyncio_run.return_value = [sample_dataframes_list]
        
        with patch('sys.argv', ['main.py', 'test.pdf', '--format', 'xlsx']):
            main()
            mock_write.assert_called_once()

    @patch('src.main.asyncio.run')
    @patch('src.main.write_output_to_csv')
    @patch('src.main.os.makedirs')
    def test_main_with_csv_output(self, mock_makedirs, mock_write, mock_asyncio_run, sample_dataframes_list):
        """Test main function with CSV output."""
        from src.main import main
        
        mock_asyncio_run.return_value = [sample_dataframes_list]
        mock_write.return_value = ["output.csv"]
        
        with patch('sys.argv', ['main.py', 'test.pdf', '--format', 'csv']):
            main()
            mock_write.assert_called_once()

    @patch('src.main.asyncio.run')
    def test_main_no_tables_extracted(self, mock_asyncio_run):
        """Test main function when no tables are extracted."""
        from src.main import main
        
        mock_asyncio_run.return_value = []
        
        with patch('sys.argv', ['main.py', 'test.pdf']):
            main()  # Should complete without error

    @patch('src.main.asyncio.run')
    def test_main_processing_error(self, mock_asyncio_run):
        """Test main function when processing fails."""
        from src.main import main
        
        mock_asyncio_run.side_effect = Exception("Processing error")
        
        with patch('sys.argv', ['main.py', 'test.pdf']):
            main()  # Should log error and return gracefully


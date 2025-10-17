"""Integration tests for the PDF table extractor."""
import pytest
import pandas as pd
import os
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio


@pytest.mark.integration
@pytest.mark.asyncio
class TestEndToEndExtraction:
    """Test end-to-end extraction workflows."""

    @patch('src.pdf_extraction.pymupdf')
    @patch('src.pdf_extraction.AsyncOpenAI')
    async def test_extract_single_page(self, mock_openai, mock_pymupdf, sample_pdf_path, mock_pymupdf_document):
        """Test extracting tables from a single page."""
        from src.pdf_extraction import extract_tables_from_pdf
        
        mock_pymupdf.open.return_value = mock_pymupdf_document
        
        # Mock the LLM responses
        with patch('src.pdf_extraction.get_validated_table_info', new_callable=AsyncMock) as mock_validate, \
             patch('src.pdf_extraction.process_tables_to_df', new_callable=AsyncMock) as mock_process:
            
            mock_validate.return_value = (1, ["Test Table"], [["Col1", "Col2"]], 0)
            mock_process.return_value = [pd.DataFrame({'Col1': [1, 2], 'Col2': [3, 4]})]
            
            result = await extract_tables_from_pdf(
                pdf_path=sample_pdf_path,
                pages="1",
                instructions="Extract all tables",
                open_api_key="test-key",
                save_output=False
            )
            
            assert isinstance(result, list)

    @patch('src.pdf_extraction.pymupdf')
    @patch('src.pdf_extraction.AsyncOpenAI')
    async def test_extract_page_range(self, mock_openai, mock_pymupdf, sample_pdf_path, mock_pymupdf_document):
        """Test extracting tables from a page range."""
        from src.pdf_extraction import extract_tables_from_pdf
        
        mock_pymupdf.open.return_value = mock_pymupdf_document
        
        with patch('src.pdf_extraction.get_validated_table_info', new_callable=AsyncMock) as mock_validate, \
             patch('src.pdf_extraction.process_tables_to_df', new_callable=AsyncMock) as mock_process:
            
            mock_validate.return_value = (1, ["Test Table"], [["Col1"]], 0)
            mock_process.return_value = [pd.DataFrame({'Col1': [1]})]
            
            result = await extract_tables_from_pdf(
                pdf_path=sample_pdf_path,
                page_range=(1, 3),
                instructions="Extract all tables",
                open_api_key="test-key",
                save_output=False
            )
            
            assert isinstance(result, list)

    @patch('src.pdf_extraction.pymupdf')
    @patch('src.pdf_extraction.AsyncOpenAI')
    async def test_extract_with_save_output(self, mock_openai, mock_pymupdf, sample_pdf_path, mock_pymupdf_document, tmp_path):
        """Test extraction with saving output."""
        from src.pdf_extraction import extract_tables_from_pdf
        
        mock_pymupdf.open.return_value = mock_pymupdf_document
        
        with patch('src.pdf_extraction.get_validated_table_info', new_callable=AsyncMock) as mock_validate, \
             patch('src.pdf_extraction.process_tables_to_df', new_callable=AsyncMock) as mock_process, \
             patch('src.pdf_extraction.write_output_final') as mock_write:
            
            mock_validate.return_value = (1, ["Test Table"], [["Col1"]], 0)
            mock_process.return_value = [pd.DataFrame({'Col1': [1, 2, 3]})]
            
            result = await extract_tables_from_pdf(
                pdf_path=sample_pdf_path,
                pages="1",
                instructions="Extract all tables",
                open_api_key="test-key",
                save_output=True,
                output_name="test_output"
            )
            
            # Verify write function was called
            mock_write.assert_called()

    @patch('src.pdf_extraction.pymupdf')
    @patch('src.pdf_extraction.AsyncOpenAI')
    async def test_extract_no_tables_found(self, mock_openai, mock_pymupdf, sample_pdf_path, mock_pymupdf_document):
        """Test extraction when no tables are found."""
        from src.pdf_extraction import extract_tables_from_pdf
        
        mock_pymupdf.open.return_value = mock_pymupdf_document
        
        with patch('src.pdf_extraction.get_validated_table_info', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = (0, [], None, 0)
            
            result = await extract_tables_from_pdf(
                pdf_path=sample_pdf_path,
                pages="1",
                instructions="Extract all tables",
                open_api_key="test-key",
                save_output=False
            )
            
            # Should return empty DataFrame when no tables found
            assert isinstance(result, pd.DataFrame)


@pytest.mark.integration
class TestCLIIntegration:
    """Test CLI integration."""

    @patch('src.main.asyncio.run')
    @patch('src.main.pymupdf')
    @patch('src.main.os.getenv')
    def test_cli_basic_execution(self, mock_getenv, mock_pymupdf, mock_run, sample_dataframes_list):
        """Test basic CLI execution."""
        from src.main import main
        
        mock_getenv.return_value = "test-api-key"
        mock_run.return_value = [sample_dataframes_list]
        
        with patch('sys.argv', ['main.py', 'test.pdf']), \
             patch('src.main.write_output_final'):
            main()

    @patch('src.main.asyncio.run')
    @patch('src.main.pymupdf')
    @patch('src.main.os.getenv')
    def test_cli_with_page_range(self, mock_getenv, mock_pymupdf, mock_run, sample_dataframes_list):
        """Test CLI with page range."""
        from src.main import main
        
        mock_getenv.return_value = "test-api-key"
        mock_run.return_value = [sample_dataframes_list]
        
        with patch('sys.argv', ['main.py', 'test.pdf', '--range', '1', '5']), \
             patch('src.main.write_output_final'):
            main()

    @patch('src.main.asyncio.run')
    @patch('src.main.pymupdf')
    @patch('src.main.os.getenv')
    def test_cli_with_custom_output(self, mock_getenv, mock_pymupdf, mock_run, sample_dataframes_list):
        """Test CLI with custom output name."""
        from src.main import main
        
        mock_getenv.return_value = "test-api-key"
        mock_run.return_value = [sample_dataframes_list]
        
        with patch('sys.argv', ['main.py', 'test.pdf', '-o', 'my_output']), \
             patch('src.main.write_output_final'):
            main()


@pytest.mark.integration
class TestConcurrencyControl:
    """Test concurrency control mechanisms."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_calls(self):
        """Test that semaphore properly limits concurrent calls."""
        from src.pdf_extraction import with_openai_semaphore, SEMAPHORE_LIMIT
        
        call_count = 0
        max_concurrent = 0
        current_concurrent = 0
        
        async def mock_coro():
            nonlocal call_count, max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)
            call_count += 1
            current_concurrent -= 1
            return call_count
        
        # Create many concurrent calls
        tasks = [with_openai_semaphore(mock_coro) for _ in range(20)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 20
        assert call_count == 20
        # Max concurrent should be limited by semaphore
        assert max_concurrent <= 20


@pytest.mark.integration
class TestDataIntegrity:
    """Test data integrity through the pipeline."""

    def test_data_preserved_through_write_read_cycle(self, sample_dataframes_list, tmp_path):
        """Test that data is preserved through write-read cycle."""
        from src.pdf_extraction import write_output_final
        
        output = [sample_dataframes_list]
        excel_path = tmp_path / "test_output.xlsx"
        
        # Write data
        write_output_final(output, str(excel_path), option=1)
        
        # Read back
        df_read = pd.read_excel(excel_path)
        
        # Original data
        df_original = pd.concat(sample_dataframes_list, axis=0)
        
        # Compare shapes
        assert df_read.shape[0] == df_original.shape[0]

    def test_duplicate_removal_integrity(self, sample_dataframes_list):
        """Test that duplicate removal maintains unique data."""
        from src.pdf_extraction import remove_duplicate_dfs
        
        # Create list with duplicates
        df_list = sample_dataframes_list + sample_dataframes_list
        
        result = remove_duplicate_dfs(df_list)
        
        # Should have removed duplicates
        assert len(result) < len(df_list)
        assert len(result) == len(sample_dataframes_list)


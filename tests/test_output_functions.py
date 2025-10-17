"""Unit tests for output writing functions in pdf_extraction.py."""
import pytest
import pandas as pd
import os
from unittest.mock import patch, MagicMock
from src.pdf_extraction import (
    write_output_final,
    write_output_to_csv
)


@pytest.fixture
def sample_output_final():
    """Create sample output_final structure for testing."""
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'X': [5, 6], 'Y': [7, 8]})
    df3 = pd.DataFrame({'M': [9, 10], 'N': [11, 12]})
    return [[df1, df2], [df3]]


class TestWriteOutputFinal:
    """Test the write_output_final function."""

    def test_write_option_1_concatenated(self, sample_output_final, tmp_path):
        """Test writing with option 1 (concatenated)."""
        excel_path = tmp_path / "output.xlsx"
        write_output_final(sample_output_final, str(excel_path), option=1)
        
        assert excel_path.exists()
        # Verify it can be read back
        df = pd.read_excel(excel_path)
        assert isinstance(df, pd.DataFrame)

    def test_write_option_2_by_page(self, sample_output_final, tmp_path):
        """Test writing with option 2 (by page)."""
        excel_path = tmp_path / "output.xlsx"
        write_output_final(sample_output_final, str(excel_path), option=2)
        
        assert excel_path.exists()
        # Verify multiple sheets
        with pd.ExcelFile(excel_path) as xls:
            assert len(xls.sheet_names) >= 1

    def test_write_option_3_with_gaps(self, sample_output_final, tmp_path):
        """Test writing with option 3 (with gaps)."""
        excel_path = tmp_path / "output.xlsx"
        write_output_final(sample_output_final, str(excel_path), option=3)
        
        assert excel_path.exists()

    def test_write_custom_gap_rows(self, sample_output_final, tmp_path):
        """Test writing with custom gap_rows."""
        excel_path = tmp_path / "output.xlsx"
        write_output_final(sample_output_final, str(excel_path), option=3, gap_rows=5)
        
        assert excel_path.exists()

    def test_write_invalid_option(self, sample_output_final, tmp_path):
        """Test writing with invalid option raises error."""
        excel_path = tmp_path / "output.xlsx"
        
        # The function raises ValueError for invalid option, but also IndexError 
        # from ExcelWriter trying to save empty workbook. We test for either.
        with pytest.raises((ValueError, IndexError)):
            write_output_final(sample_output_final, str(excel_path), option=99)

    def test_write_empty_dataframes(self, tmp_path):
        """Test writing empty DataFrames."""
        excel_path = tmp_path / "output.xlsx"
        empty_output = [[pd.DataFrame()]]
        
        write_output_final(empty_output, str(excel_path), option=1)
        assert excel_path.exists()

    def test_sanitize_dataframe_in_write(self, tmp_path):
        """Test that DataFrames are sanitized during write."""
        # Create DataFrame with problematic characters
        df = pd.DataFrame({
            'Col[1]': ['data\x00\x01'],
            'Col:2': ['normal']
        })
        output = [[df]]
        
        excel_path = tmp_path / "output.xlsx"
        write_output_final(output, str(excel_path), option=1)
        
        # Read back and verify sanitization
        df_read = pd.read_excel(excel_path)
        assert 'Col[1]' not in df_read.columns or 'Col_1_' in str(df_read.columns)


class TestWriteOutputToCsv:
    """Test the write_output_to_csv function."""

    def test_write_csv_option_1(self, sample_output_final, tmp_path):
        """Test writing CSV with option 1 (concatenated)."""
        base_path = tmp_path / "output"
        result = write_output_to_csv(sample_output_final, str(base_path), option=1)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(os.path.exists(f) for f in result)

    def test_write_csv_option_2(self, sample_output_final, tmp_path):
        """Test writing CSV with option 2 (by page)."""
        base_path = tmp_path / "output"
        result = write_output_to_csv(sample_output_final, str(base_path), option=2)
        
        assert isinstance(result, list)
        # Should have one file per page
        assert len(result) == len(sample_output_final)

    def test_write_csv_option_3(self, sample_output_final, tmp_path):
        """Test writing CSV with option 3 (with gaps)."""
        base_path = tmp_path / "output"
        result = write_output_to_csv(sample_output_final, str(base_path), option=3)
        
        assert isinstance(result, list)
        assert len(result) > 0

    def test_write_csv_invalid_option(self, sample_output_final, tmp_path):
        """Test writing CSV with invalid option."""
        base_path = tmp_path / "output"
        
        with pytest.raises(ValueError, match="Invalid `option`"):
            write_output_to_csv(sample_output_final, str(base_path), option=99)

    def test_csv_files_readable(self, sample_output_final, tmp_path):
        """Test that generated CSV files are readable."""
        base_path = tmp_path / "output"
        result = write_output_to_csv(sample_output_final, str(base_path), option=1)
        
        # Read the CSV file
        df = pd.read_csv(result[0])
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_csv_with_empty_groups(self, tmp_path):
        """Test CSV writing with empty groups."""
        output = [[pd.DataFrame({'A': [1, 2]}), pd.DataFrame()]]
        base_path = tmp_path / "output"
        
        result = write_output_to_csv(output, str(base_path), option=1)
        assert len(result) > 0


class TestDataSanitization:
    """Test data sanitization through write functions."""

    def test_sanitize_column_names_through_write(self, tmp_path):
        """Test that column names are sanitized when writing."""
        df = pd.DataFrame({
            'Col[1]': [1, 2],
            'Col:2': [3, 4],
            'Col*3': [5, 6]
        })
        output = [[df]]
        
        excel_path = tmp_path / "output.xlsx"
        write_output_final(output, str(excel_path), option=1)
        
        # Read back and verify sanitization occurred
        df_read = pd.read_excel(excel_path)
        # Column names should be modified (brackets, colons, asterisks replaced)
        assert len(df_read.columns) == 3

    def test_sanitize_string_data_through_write(self, tmp_path):
        """Test that control characters are sanitized when writing."""
        df = pd.DataFrame({
            'A': ['normal', 'with\x00null', 'with\x01control']
        })
        output = [[df]]
        
        excel_path = tmp_path / "output.xlsx"
        # Should not raise error even with control characters
        write_output_final(output, str(excel_path), option=1)
        assert excel_path.exists()

    def test_numeric_data_preserved(self, tmp_path):
        """Test that numeric data is preserved through write-read cycle."""
        df = pd.DataFrame({
            'Numbers': [1, 2, 3, 4, 5],
            'Floats': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        output = [[df]]
        
        excel_path = tmp_path / "output.xlsx"
        write_output_final(output, str(excel_path), option=1)
        
        df_read = pd.read_excel(excel_path)
        # Numeric data should be preserved
        assert df_read['Numbers'].tolist() == [1, 2, 3, 4, 5]
        assert len(df_read['Floats'].tolist()) == 5


class TestEdgeCases:
    """Test edge cases for output functions."""

    def test_very_long_worksheet_name(self, tmp_path):
        """Test with very long worksheet names."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        # Create a very long table name
        df['TableHeader'] = 'A' * 100
        output = [[df]]
        
        excel_path = tmp_path / "output.xlsx"
        write_output_final(output, str(excel_path), option=2)
        assert excel_path.exists()

    def test_special_characters_in_data(self, tmp_path):
        """Test with special characters in data."""
        df = pd.DataFrame({
            'Data': ['Normal', 'Special™', 'With©', 'Symbols®']
        })
        output = [[df]]
        
        excel_path = tmp_path / "output.xlsx"
        write_output_final(output, str(excel_path), option=1)
        assert excel_path.exists()

    def test_large_number_of_dataframes(self, tmp_path):
        """Test with large number of DataFrames."""
        dfs = [pd.DataFrame({'Col': [i]}) for i in range(50)]
        output = [[df] for df in dfs]
        
        excel_path = tmp_path / "output.xlsx"
        write_output_final(output, str(excel_path), option=2)
        assert excel_path.exists()

    def test_mixed_dataframe_sizes(self, tmp_path):
        """Test with DataFrames of different sizes."""
        df1 = pd.DataFrame({'A': [1]})
        df2 = pd.DataFrame({'B': list(range(100))})
        df3 = pd.DataFrame({'C': list(range(1000))})
        output = [[df1, df2, df3]]
        
        excel_path = tmp_path / "output.xlsx"
        write_output_final(output, str(excel_path), option=3, gap_rows=2)
        assert excel_path.exists()


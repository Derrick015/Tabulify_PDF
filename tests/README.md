# Unit Tests for AI-Powered PDF Table Extractor

This directory contains comprehensive unit tests for the PDF Table Extractor project.

## Test Structure

- **`test_pdf_extraction.py`** - Tests for PDF extraction functions including:
  - DataFrame deduplication
  - Text normalization
  - Field name sanitization
  - Page data extraction
  - Async functions

- **`test_llm.py`** - Tests for LLM integration including:
  - Table identification
  - Vision parsing
  - API response handling

- **`test_main.py`** - Tests for CLI functionality including:
  - Argument parsing
  - Page selection logic
  - Main workflow

- **`test_output_functions.py`** - Tests for output writing including:
  - Excel file generation
  - CSV file generation
  - Data sanitization
  - Multiple output formats

- **`test_integration.py`** - Integration tests including:
  - End-to-end extraction workflows
  - CLI integration
  - Concurrency control
  - Data integrity

- **`test_app.py`** - Tests for Streamlit app including:
  - Page range selection
  - File format handling
  - Preview generation
  - Input validation

- **`conftest.py`** - Shared pytest fixtures and configuration

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_pdf_extraction.py
```

### Run specific test class
```bash
pytest tests/test_pdf_extraction.py::TestRemoveDuplicateDfs
```

### Run specific test function
```bash
pytest tests/test_pdf_extraction.py::TestRemoveDuplicateDfs::test_remove_exact_duplicates
```

### Run tests with specific marker
```bash
pytest -m unit          # Run only unit tests
pytest -m integration   # Run only integration tests
pytest -m asyncio       # Run only async tests
```

### Run tests in parallel (requires pytest-xdist)
```bash
pytest -n auto
```

### Run tests with verbose output
```bash
pytest -v
```

### Run tests and stop on first failure
```bash
pytest -x
```

## Test Coverage

To generate a coverage report:

```bash
pytest --cov=src --cov-report=term-missing
```

To generate an HTML coverage report:

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## Writing New Tests

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Use descriptive test names that explain what is being tested
3. Use fixtures from `conftest.py` where appropriate
4. Mock external dependencies (API calls, file I/O) appropriately
5. Add appropriate markers for test categorization
6. Ensure tests are isolated and don't depend on each other

Example test structure:
```python
class TestMyFunction:
    """Test the my_function function."""

    def test_basic_case(self):
        """Test basic functionality."""
        result = my_function(input_data)
        assert result == expected_output

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

## Fixtures

Common fixtures available in `conftest.py`:

- `sample_pdf_path` - Temporary PDF file path
- `sample_base64_image` - Sample base64-encoded image
- `sample_dataframe` - Sample pandas DataFrame
- `mock_openai_client` - Mock OpenAI client
- `mock_table_info_response` - Mock table identification response
- `sample_extracted_text` - Sample PDF text
- `mock_pymupdf_document` - Mock PyMuPDF document

## Continuous Integration

These tests are designed to run in CI/CD pipelines. The test suite should:
- Complete in a reasonable time
- Not require external API calls (use mocks)
- Not require actual PDF files (use fixtures)
- Be deterministic and reproducible

## Troubleshooting

If tests fail:

1. Check that all dependencies are installed: `pip install -r requirements.txt`
2. Ensure you're running from the project root directory
3. Check that environment variables are set correctly (tests use mock values)
4. Review the test output for specific error messages
5. Run with `-v` flag for more detailed output

## Notes

- Tests use mocking extensively to avoid external dependencies
- Async tests are marked with `@pytest.mark.asyncio`
- Integration tests are marked with `@pytest.mark.integration`
- Some Streamlit tests are limited due to the interactive nature of the framework


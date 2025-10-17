"""Unit tests for llm.py module."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.llm import TableInfo, table_identification_llm, vision_llm_parser


class TestTableInfo:
    """Test the TableInfo Pydantic model."""

    def test_valid_table_info(self):
        """Test creating valid TableInfo instance."""
        info = TableInfo(
            num_tables=2,
            table_headers_and_positions=["Table 1 -- top", "Table 2 -- bottom"],
            columns_per_table=[["Col1", "Col2"], ["ColA", "ColB"]]
        )
        assert info.num_tables == 2
        assert len(info.table_headers_and_positions) == 2
        assert len(info.columns_per_table) == 2

    def test_table_info_without_columns(self):
        """Test TableInfo without columns_per_table."""
        info = TableInfo(
            num_tables=1,
            table_headers_and_positions=["Table 1 -- center"]
        )
        assert info.num_tables == 1
        assert info.columns_per_table is None

    def test_zero_tables(self):
        """Test TableInfo with zero tables."""
        info = TableInfo(
            num_tables=0,
            table_headers_and_positions=["no tables"]
        )
        assert info.num_tables == 0


@pytest.mark.asyncio
class TestTableIdentificationLlm:
    """Test the table_identification_llm function."""

    async def test_successful_table_identification(self, mock_openai_client, sample_base64_image):
        """Test successful table identification."""
        # Setup mock response
        mock_parsed = MagicMock()
        mock_parsed.num_tables = 2
        mock_parsed.table_headers_and_positions = ["Table A -- top", "Table B -- bottom"]
        mock_parsed.columns_per_table = [["Col1", "Col2"], ["ColX", "ColY"]]
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.parsed = mock_parsed
        
        mock_openai_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        result = await table_identification_llm(
            user_text="Extract all tables",
            base64_image=sample_base64_image,
            openai_client=mock_openai_client,
            model="gpt-4"
        )
        
        assert result.num_tables == 2
        assert len(result.table_headers_and_positions) == 2
        assert len(result.columns_per_table) == 2

    async def test_no_tables_found(self, mock_openai_client, sample_base64_image):
        """Test when no tables are found."""
        mock_parsed = MagicMock()
        mock_parsed.num_tables = 0
        mock_parsed.table_headers_and_positions = ["no tables"]
        mock_parsed.columns_per_table = None
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.parsed = mock_parsed
        
        mock_openai_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        result = await table_identification_llm(
            user_text="Extract all tables",
            base64_image=sample_base64_image,
            openai_client=mock_openai_client,
            model="gpt-4"
        )
        
        assert result.num_tables == 0

    async def test_custom_structure_output(self, mock_openai_client, sample_base64_image):
        """Test with custom structure output."""
        custom_structure = MagicMock()
        
        mock_parsed = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.parsed = mock_parsed
        
        mock_openai_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        result = await table_identification_llm(
            user_text="Extract tables",
            base64_image=sample_base64_image,
            openai_client=mock_openai_client,
            model="gpt-4",
            structure_output=custom_structure
        )
        
        # Verify the custom structure was passed
        call_args = mock_openai_client.beta.chat.completions.parse.call_args
        assert call_args[1]['response_format'] == custom_structure

    async def test_with_specific_user_instructions(self, mock_openai_client, sample_base64_image):
        """Test with specific user instructions."""
        mock_parsed = MagicMock()
        mock_parsed.num_tables = 1
        mock_parsed.table_headers_and_positions = ["Specific Table -- middle"]
        mock_parsed.columns_per_table = [["A", "B", "C"]]
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.parsed = mock_parsed
        
        mock_openai_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        result = await table_identification_llm(
            user_text="Extract only the price table",
            base64_image=sample_base64_image,
            openai_client=mock_openai_client,
            model="gpt-4"
        )
        
        assert result.num_tables == 1


@pytest.mark.asyncio
class TestVisionLlmParser:
    """Test the vision_llm_parser function."""

    async def test_successful_parsing(self, mock_openai_client, sample_base64_image, sample_extracted_text):
        """Test successful vision LLM parsing."""
        # Mock output dictionary
        mock_output_dict = [
            {'column_a': 'value1', 'column_b': 'value2'},
            {'column_a': 'value3', 'column_b': 'value4'}
        ]
        
        mock_parsed = MagicMock()
        mock_parsed.output_dictionary = mock_output_dict
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.parsed = mock_parsed
        
        mock_openai_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        # Mock structure output
        structure_output = MagicMock()
        
        result = await vision_llm_parser(
            user_text="Extract data",
            text_input=sample_extracted_text,
            table_to_target="Test Table -- top",
            base64_image=sample_base64_image,
            openai_client=mock_openai_client,
            model="gpt-4",
            structure_output=structure_output
        )
        
        assert result == mock_output_dict
        assert len(result) == 2

    async def test_parsing_with_table_position(self, mock_openai_client, sample_base64_image, sample_extracted_text):
        """Test parsing with table position information."""
        mock_output_dict = [{'col1': 'data1'}]
        
        mock_parsed = MagicMock()
        mock_parsed.output_dictionary = mock_output_dict
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.parsed = mock_parsed
        
        mock_openai_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        structure_output = MagicMock()
        
        result = await vision_llm_parser(
            user_text="Extract data",
            text_input=sample_extracted_text,
            table_to_target="Product Table -- Located at bottom right",
            base64_image=sample_base64_image,
            openai_client=mock_openai_client,
            model="gpt-4",
            structure_output=structure_output
        )
        
        assert isinstance(result, list)
        # Verify table_to_target was included in the message
        call_args = mock_openai_client.beta.chat.completions.parse.call_args
        message_content = call_args[1]['messages'][0]['content'][0]['text']
        assert "Product Table" in message_content

    async def test_parsing_empty_table(self, mock_openai_client, sample_base64_image, sample_extracted_text):
        """Test parsing when table is empty."""
        mock_output_dict = []
        
        mock_parsed = MagicMock()
        mock_parsed.output_dictionary = mock_output_dict
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.parsed = mock_parsed
        
        mock_openai_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        structure_output = MagicMock()
        
        result = await vision_llm_parser(
            user_text="Extract data",
            text_input=sample_extracted_text,
            table_to_target="Empty Table -- top",
            base64_image=sample_base64_image,
            openai_client=mock_openai_client,
            model="gpt-4",
            structure_output=structure_output
        )
        
        assert result == []

    async def test_parsing_with_different_models(self, mock_openai_client, sample_base64_image, sample_extracted_text):
        """Test parsing with different model names."""
        mock_output_dict = [{'data': 'test'}]
        
        mock_parsed = MagicMock()
        mock_parsed.output_dictionary = mock_output_dict
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.parsed = mock_parsed
        
        mock_openai_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        structure_output = MagicMock()
        
        models_to_test = ["gpt-4", "gpt-4-turbo", "gpt-5"]
        
        for model in models_to_test:
            result = await vision_llm_parser(
                user_text="Extract data",
                text_input=sample_extracted_text,
                table_to_target="Table",
                base64_image=sample_base64_image,
                openai_client=mock_openai_client,
                model=model,
                structure_output=structure_output
            )
            
            assert result == mock_output_dict
            # Verify correct model was used
            call_args = mock_openai_client.beta.chat.completions.parse.call_args
            assert call_args[1]['model'] == model


# AI Powered PDF Table Extractor

A powerful tool that uses AI models to extract tabular data from PDF documents.

## Video Demo
https://github.com/user-attachments/assets/6644dce8-45af-4366-835f-e47913fc627e

## Overview

The AI Powered PDF Table Extractor is designed to solve the challenging problem of extracting structured table data from PDF files. It leverages advanced multimodal models to identify, parse, and extract tables from PDFs.
## Key Features

- **AI-Powered Table Detection**: Automatically identifies tables within PDF documents
- **Data Extraction**: Extracts structured data from tables
- **Multiple Page Processing**: Process specific pages, page ranges, or entire documents
- **Customizable AI Instructions**: Provide specific extraction instructions for tailored results
- **Multiple Export Formats**: Download extracted tables in Excel or CSV formats
- **Table in Image Detection**: Optional feature to detect and extract tables embedded in images
- **Table and Page Information**: Option to add table name, position, and page number to extracted data
- **User-Friendly Interface**: Simple web-based UI built with Streamlit
- **Skips Pages With No Tables**: Pages without detected tables are automatically skipped
- **Text Validation Mode**: When enabled, values not present in the PDF text are replaced with "N/A"

## Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key 

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/Derrick015/PDF_Table_Extractor_AI.git
   cd PDF_Table_Extractor_AI
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Running the Application

Start the Streamlit web application:

```
streamlit run app.py
```

The application will be available at http://localhost:8501 in your web browser.

### Command Line (CLI) Usage

You can also run the extractor from the command line without the UI:

```
python main.py /absolute/path/to/file.pdf
```

Common options:

- Page selection:
  - Range (inclusive): `--range 1 10`
  - Specific pages: `--pages 1,3,5`
- Output:
  - Format: `--format xlsx` or `--format csv`
  - Layout style: `--style concatenated|by_page|sheet`
- Extraction behavior:
  - Instructions: `--instructions "Extract all data from the table(s)"`
  - Image & Inference Mode (bypass text validation): `--table-in-image`
  - Add table/page metadata: `--add-table-info`

Example:

```
python main.py /absolute/path/to/file.pdf --range 1 10 --format xlsx --style sheet --instructions "Extract all data from the table(s)"
```

### Extracting Tables

1. Upload a PDF file using the file uploader
2. Select pages to process (all pages, specific range, or custom selection)
3. Review page previews to confirm your selection
4. Optionally, customize the AI instructions in the sidebar
5. Enable additional options if needed:
   - Image & Inference Mode
   - Add table and page information to the output
6. Select a model of your choice
6. Choose your preferred output format (Excel or CSV)
7. Click "Process Selected Pages" to start extraction
8. Download the results in your preferred format:
   - Format 1: All tables concatenated on a single sheet
   - Format 2: All Tables on a page per sheet with spacing
   - Format 3: All tables on one sheet with spacing

### Defaults and Behavior

- Pages without tables are skipped:
  - When Image & Inference Mode is OFF, a fast pre-check skips pages with no detectable tables.
  - Regardless of mode, pages where the AI identifies 0 tables are skipped.
- Text validation (removing values not present in PDF text):
  - CLI and notebook helpers: ON by default (Image & Inference Mode is OFF).
  - Streamlit UI: Turn OFF "Image & Inference Mode" to enable this strict validation. When ON, the model may infer values and bypass text validation.

## How It Works

1. **Table Detection**: The AI vision model analyzes the PDF to identify and locate tables
2. **Structure Recognition**: The system determines table headers, rows, and columns
3. **Data Extraction**: Content is extracted based on available PDF text. In its absence the model will try to make an inference from the image. 
4. **Validation**: Multiple AI passes to mitigate hallucination and improve accuracy and consistency. With strict text validation enabled (default for CLI/notebooks, optional in Streamlit by turning OFF Image & Inference Mode), values not present in the PDF text are replaced with "N/A". When Image & Inference Mode is ON, the model may infer values and bypass this check.
5. **Export**: Data is formatted into Excel or CSV files for easy use

## Technical Details

- **PDF Processing**: Uses PyMuPDF for efficient PDF handling
- **AI Vision and Text Extration**: Leverages OpenAI's model for visual recognition and text extraction
- **Concurrent Processing**: Implements asyncio for parallel page processing
- **Data Handling**: Pandas for structured data manipulation
- **Web Interface**: Streamlit for an intuitive user experience

## Limitations

- Performance depends on the quality and complexity of the PDF
- Processing large documents may take time and consume API credits
- Very complex or highly stylised tables may require manual verification
- 200 MB max file size

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 Derrick Owusu Ofori

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact

For questions, support, or collaboration:

- GitHub Issues: Please use the [issue tracker](https://github.com/Derrick015/PDF_Table_Extractor_AI) for bug reports and feature requests
- LinkedIn: [Derrick Ofori](https://www.linkedin.com/in/derrickofori/)

Feel free to reach out through any of these channels. For bug reports and feature requests, please use GitHub Issues as the primary channel. 

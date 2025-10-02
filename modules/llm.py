from pydantic import BaseModel
import aiohttp

from openai import OpenAI
from pydantic import BaseModel

class TableInfo(BaseModel):
    num_tables: int
    table_headers_and_positions: list[str]
    # For each table (same order as headers), list the column names as they appear in the table
    columns_per_table: list[list[str]] | None = None


async def table_identification_llm(user_text, base64_image, openai_client, model, structure_output=TableInfo):


    response = await openai_client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": f"""
                    
                    You are an expert in PDF document analysis, table detection, and textual pattern recognition. 
                    Your expertise lies in extracting structured information from PDF and identifying tables

                    You will be provided with:
                    * A user's request
                    * An image of a page from a PDF document for reference.
            
                    Your task is to:
                    a. Determine and return the total number of tables present on the page in line with the user's request.

                    b. Identify and return table headers/names present on the page in line with the user's request (eg: Cable Type NEBU, etc). This is usually found on the top of the table. If a table lacks explicit headers, provide a descriptive label for the table based on near by or other content that distinguishes it. Additionally kindly add the position of the table on the page.

                    c. For each identified table, list the visible column names EXACTLY as they appear in the table, from left-to-right (or top-to-bottom if the table is vertical). If a column header is multi-row, concatenate parts with a single space. If a column has subcolumns, concatenate parent and subcolumn with a single space (e.g., "Diameter mm"). If a header is blank or not present, use a short descriptive placeholder like "Column 1", "Column 2", etc.
                    
                    Expected Output Format:

                    1. Number of Tables on the Page: [Total count]  

                    2. Table Header and Position: [Provide the table header or name, if present (this is the preferred option and is usually located at the top of the table), or use a descriptor if no header is available. Follow each header or descriptor with two dashes "--" and the table's position on the page. Each table must have one and only one header or descriptor. Example output: "Screw Machine -- Located at the bottom left of the page",  "Spanners 10mm -- Located at the bottom right of the page", ...]

                    3. Columns Per Table: [An array where each entry is a list of strings representing the exact column names for the corresponding table from step 2, in order in line with the user's request]

                    Note: If there are no tables on the page simply return "0" for the number of tables on the page and "no tables" for the table headers 
                    Note: If the user requests to extract from a specific table, ensure you only return the tables that match the user's request. 
                    Note: Ensure the columns are in the order and only extract the columns that the user has requested.
                    
                    
                    ``````````````````````          
                    User's request:    {user_text}
                    ``````````````````````
                    """ },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}", "detail": "auto"
                        },
                    },
                ],
            }
        ],
        response_format=structure_output,
    )
    response_1 = response.choices[0].message.parsed
    return response_1





## A. text and vision version
async def vision_llm_parser(user_text, text_input, table_to_target, base64_image, openai_client, model, structure_output):


    response = await openai_client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": f"""
                    
            You are a skilled data extraction and parsing specialist with expertise in analysing PDF text and image documents and extracting structured information from tables into a format that can be be converted to a dataframe and then exported to excel.

            You will be provided with:

            * A users request. 
            * A table header or descriptor along with its position on the page to guide you on which table to extract data from.
            * An image of the PDF page for reference.
            * All text data on the PDF page. 
            
            I need to extract the data and return it under the expected column names if they are provided.

            Provide the output for the following:

            ``````````````````````````
            User's request: {user_text}
            ``````````````````````````
            
            ``````````````````````````
            Table to extract data from: {table_to_target}
            ``````````````````````````

            ``````````````````````````
            Text document: {text_input}
            ``````````````````````````

            ``````````````````````````
            NOTE:
            - If a cell value spans multiple rows, duplicate the value for each row it covers so that every row has a complete set of values
            - Use EXACTLY and ONLY the expected column names, unless specified otherwise by the user. Preserve their spelling, casing, numbers, units, and punctuation.
            ``````````````````````````

                            """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}", "detail": "high"
                        },
                    },
                ],
            }
        ],
        response_format=structure_output,
    )
    response_1 = response.choices[0].message.parsed
    return response_1.output_dictionary


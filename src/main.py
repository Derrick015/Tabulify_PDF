import argparse
import asyncio
import logging
import os
import time
from dotenv import load_dotenv
import pymupdf
import itertools
from openai import AsyncOpenAI

from src.pdf_extraction import (
    get_page_pixel_data,
    get_validated_table_info,
    process_tables_to_df,
    write_output_final,
    write_output_to_csv,
    remove_duplicate_dfs,
)

# --------------------------------------------------
# Logging configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def parse_page_selection(total_pages: int, args: argparse.Namespace):
    """Translate CLI arguments into a list of zero-indexed page numbers."""

    if args.pages:
        page_nums = [int(p.strip()) for p in args.pages.split(",") if p.strip()]
        validated = [p for p in page_nums if 1 <= p <= total_pages]
        if not validated:
            raise ValueError("No valid page numbers supplied via --pages.")
        return [p - 1 for p in validated]

    if args.range:
        start, end = args.range
        if not (1 <= start <= end <= total_pages):
            raise ValueError("Invalid --range values. Ensure 1 <= start <= end <= total_pages")
        return list(range(start - 1, end))

    # default: all pages
    return list(range(total_pages))


async def process_pdf(args):
    """Main asynchronous workflow to extract tables from the PDF."""
    load_dotenv()
    open_api_key = os.getenv("OPENAI_API_KEY")
    if not open_api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not found.")
    
    # openai_client = OpenAI(api_key=open_api_key)
    openai_client = AsyncOpenAI(api_key=open_api_key)

    logging.info("Opening PDF: %s", args.pdf_path)
    doc = pymupdf.open(args.pdf_path)
    page_indices = parse_page_selection(doc.page_count, args)
    logging.info("Processing %d page(s): %s", len(page_indices), [i + 1 for i in page_indices])

    results_output = []

    async with asyncio.TaskGroup() as tg:
        tasks = []
        for page_no in page_indices:
            page = doc.load_page(page_no)

            if not args.table_in_image:
                # quick skip for pages without detectable tables
                if len(page.find_tables().tables) == 0:
                    logging.info("Page %d: no tables detected via PyMuPDF, skipping.", page_no + 1)
                    continue

            extracted_text = page.get_text()
            # Reduced DPI from 500 to 200 for faster processing
            base64_image = get_page_pixel_data(
                pdf_path=args.pdf_path,
                page_no=page_no,
                dpi=200,
                image_type="png",
            )


            # Validate via LLM
            num_tables, table_headers, columns_per_table, table_info_confidence = await get_validated_table_info(
                user_text=args.instructions,
                openai_client=openai_client,
                base64_image=base64_image,
                model=args.vision_model,
            )

            logging.info(f"table_info_confidence: {table_info_confidence}")
            logging.info(f"num_tables: {num_tables}")
            logging.info(f"table_headers: {table_headers}")
            logging.info(f"columns_per_table: {columns_per_table}")

            if num_tables == 0:
                logging.info("Page %d: LLM reported no tables, skipping.", page_no + 1)
                continue

            logging.info("Page %d: %d table(s) detected.", page_no + 1, num_tables)



            tasks.append(
                tg.create_task(
                    process_tables_to_df(
                        table_headers=table_headers,
                        expected_columns_per_table=columns_per_table,
                        user_text=args.instructions,
                        extracted_text=extracted_text,
                        base64_image=base64_image,
                        openai_client=openai_client,
                        page_number=page_no,
                        confidence_for_table_info=table_info_confidence,
                        table_in_image=args.table_in_image,
                        add_in_table_and_page_information=args.add_table_info,
                        append_raw_extracted_text=False,
                        model=args.model
                    )
                )
            )

        for t in tasks:
            results_output.append(await t)

    doc.close()
    return results_output


# --------------------------------------------------
# CLI
# --------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(description="PDF Table Extractor AI (CLI)")
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument(
        "-o",
        "--output-name",
        default="output_file",
        help="Base name for the output file(s) (without extension)",
    )
    parser.add_argument(
        "--format",
        choices=["xlsx", "csv"],
        default="xlsx",
        help="Output file format",
    )
    parser.add_argument(
        "--style",
        choices=["concatenated", "by_page", "sheet"],
        default="sheet",
        help="Layout style for output: \n"
        "concatenated -> all tables stacked vertically (Format 1);\n"
        "by_page -> separate tables per page (Format 2);\n"
        "sheet -> all tables on one sheet with gaps (Format 3).",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--range",
        metavar=("START", "END"),
        nargs=2,
        type=int,
        help="Process a continuous range of pages, inclusive (1-indexed)",
    )
    group.add_argument(
        "--pages",
        type=str,
        help="Comma-separated list of individual page numbers to process (1-indexed)",
    )

    parser.add_argument(
        "--instructions",
        default="Extract all data from the table(s)",
        help="Natural-language instructions passed to the LLM",
    )
    parser.add_argument(
        "--table-in-image",
        action="store_true",
        help="Enable Image & Inference Mode (bypass text-based validation)",
    )
    parser.add_argument(
        "--add-table-info",
        action="store_true",
        dest="add_table_info",
        help="Include table and page metadata as extra columns in the output",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="Text LLM model to use for extraction",
    )
    parser.add_argument(
        "--vision-model",
        default="gpt-4.1",
        help="Vision model to use for header detection",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    start_time = time.time()

    try:
        output_final = asyncio.run(process_pdf(args))
    except Exception as e:
        logging.error("Processing failed: %s", str(e))
        return

    if not output_final:
        logging.warning("No tables were extracted. Exiting.")
        return

    # Ensure output directory exists
    os.makedirs("output_files", exist_ok=True)
    base_path = os.path.join("output_files", args.output_name)

    # Map style to option
    style_map = {
        "concatenated": 1,
        "by_page": 2,
        "sheet": 3,
    }
    option = style_map[args.style]

    if args.format == "xlsx":
        excel_path = f"{base_path}.xlsx" if option == 1 else f"{base_path}_format_{option}.xlsx"
        write_output_final(output_final, excel_path=excel_path, option=option)
        logging.info("Excel written to %s", excel_path)
    else:
        csv_files = write_output_to_csv(output_final, csv_base_path=base_path, option=option)
        if isinstance(csv_files, list):
            logging.info("CSV files written: %s", ", ".join(csv_files))
        else:
            logging.info("CSV file written to %s", csv_files)

    # Deduplicate & log summary
    all_dfs = remove_duplicate_dfs(list(itertools.chain.from_iterable(output_final)))
    logging.info("Extraction complete. %d unique table(s) found.", len(all_dfs))

    elapsed = time.time() - start_time
    logging.info("Total runtime: %.2f seconds", elapsed)


if __name__ == "__main__":
    main() 
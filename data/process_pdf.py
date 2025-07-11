import os
import pathlib
from dotenv import load_dotenv

from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TesseractOcrOptions
from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormatOption
from docling.datamodel.base_models import InputFormat

# Load environment variables
load_dotenv()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def convert_pdf_to_md():
    for file in os.listdir("raw"):
        if not file.endswith('.pdf'):
            continue
        input_path = os.path.join("raw", file)
        print(f"Processing: {input_path}")
        try:
            ocr_options = TesseractOcrOptions(
                lang=["vie"],
                force_full_page_ocr=False
            )
            pipeline_options = PdfPipelineOptions(
                do_ocr=True, 
                do_table_structure=True,
                ocr_options=ocr_options
            )
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
            print(f"Converting: {file}")
            result = converter.convert(input_path)
            text = result.document.export_to_markdown()
            output_path = os.path.join("processed", file.replace(".pdf", ".md"))
            pathlib.Path(output_path).write_bytes(text.encode())
            print(f"✓ Done: {input_path}")
        except Exception as e:
            print(f"✗ Error: {file} -> {str(e)}")

if __name__ == "__main__":
    print("\n--- PDF Processing ---")
    print("Converting PDFs to Markdown files...")
    convert_pdf_to_md()
    print("All PDF processing completed.")

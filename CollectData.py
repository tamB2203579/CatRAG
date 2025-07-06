import os
import re
import json
import pathlib
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_text_splitters import MarkdownTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from embedding import Embedding

from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TesseractOcrOptions
from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormatOption
from docling.datamodel.base_models import InputFormat

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Labels for classification
labels = ["Dao_tao", "Hoc_tap_ren_luyen", "Khen_thuong_ky_luat", "Tot_nghiep", "KTX", "Khac"]

# ----- Helper Functions -----

def preprocess(text):
    words = text.split()
    processed_words = [word if word.isupper() else word.lower() for word in words]
    text = " ".join(processed_words)
    text = re.sub(r'[^\w\s,.?%+/\\\-]', '', text)
    return text.strip()

def vietnamese_spelling(text):
    with open("prompt/vietnamese_spelling.txt", "r", encoding="utf-8") as f:
        template = f.read()
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = {"text": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return chain.invoke(text)

def chunking(text):
    text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.create_documents([text])

def semantic_chunking(text):
    text_splitter = SemanticChunker(
        embeddings=Embedding(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=100
    )
    return text_splitter.create_documents([text])

def data_augment(varr, name):
    with open("prompt/augment.txt", "r", encoding="utf-8") as f:
        template = f.read()
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = (
        {"category": RunnablePassthrough(), "num_variations": RunnablePassthrough(), "text": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    df = pd.read_excel("dataset.xlsx", sheet_name=name)
    augmented_data = []
    category = df['label'].iloc[0]
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            result = chain.invoke({"category": category, "num_variations": varr, "text": row['text']})
            augmented_texts = json.loads(result)
            augmented_data.append({"text": row['text'], "label": row['label']})
            for variant in augmented_texts[0]["variants"]:
                augmented_data.append({"text": variant, "label": row['label']})
        except Exception as e:
            print(f"Error at index {index}: {e}")
    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_excel(f"augmented_dataset_{name}.xlsx", index=False)
    print(f"Original: {len(df)}, Augmented: {len(augmented_df)}")

def convert_pdf_to_md():
    for file in os.listdir("content"):
        if not file.endswith('.pdf'):
            continue
        input_path = os.path.join("content", file)
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
            text = result.document.export_to_text()
            output_path = os.path.join("result", file.replace(".pdf", ".txt"))
            pathlib.Path(output_path).write_bytes(text.encode())
            print(f"✓ Done: {input_path}")
        except Exception as e:
            print(f"✗ Error: {file} -> {str(e)}")

def process_md_files(use_semantic=False):
    for file in tqdm(os.listdir("result")):
        if not file.endswith(".md"):
            continue
        try:
            with open(os.path.join("result", file), "r", encoding="utf-8") as f:
                content = f.read()
            content = vietnamese_spelling(content)
            content = preprocess(content)
            chunks = semantic_chunking(content) if use_semantic else chunking(content)
            data = [{"text": chunk.page_content} for chunk in chunks]
            df = pd.DataFrame(data)
            path = os.path.join("result", file.replace(".md", ".xlsx"))
            df.to_excel(path, index=False, engine="openpyxl")
            print(f"Processed: {file} → {len(chunks)} chunks")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

# ----- Main Entry -----

def main():
    while True:
        print("\n--- Menu ---")
        print("1. Convert PDFs to Text Files")
        print("2. Process Text Files using Recursive Chunking")
        print("3. Process Text Files using Semantic Chunking")
        print("4. Augment Classification Data")
        print("5. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            convert_pdf_to_md()
        elif choice == '2':
            process_md_files(use_semantic=False)
        elif choice == '3':
            process_md_files(use_semantic=True)
        elif choice == '4':
            name = input("Enter class for augment: ")
            varr = input("Enter number of variations to generate: ")
            data_augment(varr, name)
        elif choice == '5':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
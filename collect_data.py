import os
import re
import json
import random
import pathlib
import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import List, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import MarkdownTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormatOption
from docling.datamodel.base_models import InputFormat

from embedding import Embedding

load_dotenv()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")


# ========== PDF TO MARKDOWN ==========
def convert_pdf_to_md():
    for file in os.listdir("data/raw"):
        if file.endswith(".pdf"):
            input_path = os.path.join("data/raw", file)
            try:
                ocr_options = TesseractOcrOptions(
                    lang=["vie"], force_full_page_ocr=False
                )
                pipeline_options = PdfPipelineOptions(
                    do_ocr=True, do_table_structure=True, ocr_options=ocr_options
                )
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
                converter = DocumentConverter(
                    {
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options
                        )
                    }
                )
                result = converter.convert(input_path)
                markdown = result.document.export_to_markdown()
                output_path = os.path.join("data/processed", file.replace(".pdf", ".md"))
                pathlib.Path(output_path).write_bytes(markdown.encode())
                print(f"✓ Converted: {input_path}")
            except Exception as e:
                print(f"✗ Error processing {file}: {e}")


# ========== TEXT CLEANING / CHUNKING ==========
def preprocess_text(text):
    words = text.split()
    processed = [w if w.isupper() else w.lower() for w in words]
    text = " ".join(processed)
    return re.sub(r"[^\w\s,.?%+/\\\-]", "", text).strip()


def apply_vietnamese_spelling_correction(text):
    with open("./prompt/vietnamese_spelling.txt", "r", encoding="utf-8") as f:
        template = f.read()
    chain = (
        {"text": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(template)
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        | StrOutputParser()
    )
    return chain.invoke(text)


def chunk_text(text, use_semantic=False):
    if use_semantic:
        splitter = SemanticChunker(
            embeddings=Embedding(),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            min_chunk_size=100,
        )
    else:
        splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.create_documents([text])


def process_md_files(use_semantic=False):
    for file in tqdm(os.listdir("data/processed")):
        if file.endswith(".md"):
            with open(os.path.join("data/processed", file), "r", encoding="utf-8") as f:
                content = f.read()
            content = preprocess_text(apply_vietnamese_spelling_correction(content))
            chunks = chunk_text(content, use_semantic)
            df = pd.DataFrame([{"text": chunk.page_content} for chunk in chunks])
            output_path = os.path.join("data/processed", file.replace(".md", ".xlsx"))
            df.to_excel(output_path, index=False, engine="openpyxl")
            print(f"Processed: {file} → {len(chunks)} chunks")


# ========== QUESTION GENERATION ==========
def load_question_prompt():
    with open("prompt/question_generate.txt", "r", encoding="utf-8") as f:
        return ChatPromptTemplate.from_template(f.read())


def load_chunks_from_excel(folder="./data/processed"):
    all_chunks = []
    for path in glob(os.path.join(folder, "*.xlsx")):
        try:
            df = pd.read_excel(path, engine="openpyxl")
            for _, row in df.iterrows():
                all_chunks.append(
                    {"text": row.get("text", ""), "source_file": os.path.basename(path)}
                )
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    return all_chunks


def generate_questions(text, model_name="gpt-4o-mini"):
    llm = ChatOpenAI(model=model_name, temperature=0.7)
    chain = load_question_prompt() | llm | StrOutputParser()
    try:
        response = chain.invoke({"context": text}).strip()
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end != -1:
            json_data = json.loads(response[start:end])
            return [
                {"question": qa["question"].strip(), "answer": qa["answer"].strip()}
                for qa in json_data
                if "question" in qa
                and "answer" in qa
                and len(qa["question"]) > 10
                and len(qa["answer"]) > 5
            ]
    except Exception as e:
        print(f"Error generating questions: {e}")
    return []


def save_questions_to_excel(data: List[Dict], path="dataset.xlsx"):
    df = pd.DataFrame(data)
    df.rename(columns={"answer": "ground_truth"}, inplace=True)
    df.to_excel(path, index=False, engine="openpyxl")
    print(f"Saved {len(df)} questions to {path}")


def generate_question_dataset(num_samples=50, output_excel="dataset.xlsx"):
    chunks = load_chunks_from_excel()
    if not chunks:
        print("No chunks found.")
        return
    sample_chunks = random.sample(chunks, min(num_samples, len(chunks)))
    results = []
    for i, chunk in enumerate(sample_chunks):
        print(f"Generating from chunk {i + 1}/{len(sample_chunks)}")
        qa_pairs = generate_questions(chunk["text"])
        for qa in qa_pairs:
            results.append(
                {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "context": chunk["text"],
                    "source_file": chunk["source_file"],
                }
            )
    if output_excel:
        save_questions_to_excel(results, output_excel)


def preview_generated_questions(num=5):
    chunks = load_chunks_from_excel()
    if not chunks:
        return print("No data found.")
    for chunk in random.sample(chunks, min(num, len(chunks))):
        print(f"\n=== From: {chunk['source_file']} ===")
        print(f"Text: {chunk['text'][:200]}...")
        qas = generate_questions(chunk["text"])
        for i, qa in enumerate(qas):
            print(f"{i + 1}. Q: {qa['question']}\n   A: {qa['answer']}\n")


# ========== CLI MAIN ==========
def main():
    while True:
        print("\n--- MENU ---")
        print("1. Convert PDFs to Markdown")
        print("2. Process Markdown → XLSX (Recursive Chunking)")
        print("3. Process Markdown → XLSX (Semantic Chunking)")
        print("4. Preview Questions")
        print("5. Generate Excel Dataset")
        print("6. Exit")

        choice = input("Enter choice: ").strip()
        if choice == "1":
            convert_pdf_to_md()
        elif choice == "2":
            process_md_files(use_semantic=False)
        elif choice == "3":
            process_md_files(use_semantic=True)
        elif choice == "4":
            preview_generated_questions()
        elif choice == "5":
            num = int(input("Number of samples [50]: ") or "50")
            excel_file = input("Excel file [dataset.xlsx]: ") or "dataset.xlsx"
            generate_question_dataset(num, excel_file)
        elif choice == "6":
            print("Exiting.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()

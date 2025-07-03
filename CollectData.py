from multilingual_pdf2text.models.document_model.document import Document
from multilingual_pdf2text.pdf2text import PDF2Text
from langchain_text_splitters import MarkdownTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from embedding import Embedding
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import pathlib
import json
import os
import re

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize labels for classification task
labels = ["Dao_tao", "Hoc_tap_ren_luyen", "Khen_thuong_ky_luat", "Tot_nghiep", "KTX", "Khac"]

def load_stop_words():
    # Load stopwords.txt
    with open("./lib/stopwords.txt", "r", encoding="utf-8") as f:
        stop_words = f.read().splitlines()
    
    return stop_words

def preprocess(text):
    # Convert non-uppercase words to lowercase
    words = text.split()
    processed_words = [word if word.isupper() else word.lower() for word in words]

    # Join words back into a string
    text = " ".join(processed_words)

    # Remove unwanted special characters (keep letters, numbers, whitespace, , ? . - % + / \)
    text = re.sub(r'[^\w\s,.?%+/\\\-]', '', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text

def convert_pdf_to_markdown_with_llm(pdf_path, output_path=None, use_cache=True):
    # Initialize document object with Vietnamese language setting
    pdf_document = Document(document_path=pdf_path, language="vie")
    pdf2text = PDF2Text(document=pdf_document)

    # Extract text from PDF
    extracted_text = ""
    for words in pdf2text.extract():
        extracted_text += words["text"]

    # Load prompt template for PDF to Markdown conversion
    with open("prompt/pdf_to_markdown.txt", mode="r", encoding="utf-8") as f:
        template = f.read()

    # Set up the LLM chain
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create processing chain: pass PDF content through the prompt to the model
    chain = (
        {"pdf_content": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Process the extracted text through the LLM chain
    markdown_text = chain.invoke(extracted_text)

    # Save to file if output path is provided
    if output_path:
        pathlib.Path(output_path).write_text(markdown_text, encoding="utf-8")
        print(f"Markdown saved to {output_path}")

    return markdown_text

def chunking(text, use_cache=True):
    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.create_documents([text])

def semantic_chunking(text, use_cache=True):
    text_splitter = SemanticChunker(breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95, embeddings=Embedding(), min_chunk_size=100)
    return text_splitter.create_documents([text])

def labelling():
    choice = input("Enter label for this data: ")
    return "__label__" + labels[int(choice)]

def data_augment(varr, use_cache=True):
    with open("prompt/augment.txt", mode="r", encoding="utf-8") as f:
        template = f.read()
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = (
        {"category": RunnablePassthrough(), "num_variations": RunnablePassthrough(), "text": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    df = pd.read_excel("classed_dataset.xlsx", sheet_name="KTX")
    augmented_data = []

    category = df['label'].iloc[0]

    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        
        chain_input = {
            "category": category,
            "num_variations": varr,
            "text": text
        }
        
        try:
            result = chain.invoke(chain_input)
            
            augmented_texts = json.loads(result)
            
            augmented_data.append({
                "text": text,
                "label": row['label']
            })
            
            for variant in augmented_texts[0]["variants"]:
                augmented_data.append({
                    "text": variant,
                    "label": row['label']
                })
            
        except Exception as e:
            print(f"Error processing text at index {index}: {e}")
    
    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_excel("augmented_dataset_TN.xlsx", index=False)
    print(f"Original dataset size: {len(df)}, Augmented dataset size: {len(augmented_df)}")
    print(f"Cached augmented data with {varr} variations")

def main():
    while True:
        print("---Collect Data Menu---")
        print("1. Convert PDF to CSV file using Recursive chunking.")
        print("2. Convert PDF to CSV file using Semantic chunking.")
        print("3. Augment data for classification.")
        print("4. Exit")
        print("----------------------")
        choice = int(input("Enter your choice: "))
        if choice == 1:
           for file in tqdm(os.listdir("content")):
            if file.endswith(".pdf"):
                content = convert_pdf_to_markdown_with_llm(os.path.join("content", file), os.path.join("result", file.replace(".pdf", ".md")))
                content = preprocess(content)
                chunks = chunking(content)
                data = []
                label = labelling()
                for chunk in chunks:
                    data.append({"text": chunk.page_content, "label": label})
                df = pd.DataFrame(data)
                path = os.path.join("result", file.replace(".pdf", ".csv"))
                df.to_csv(path, index=False, encoding="utf-8-sig", sep=";")
        elif choice == 2:
            for file in tqdm(os.listdir("content")):
                if file.endswith(".pdf"):
                    content = convert_pdf_to_markdown_with_llm(os.path.join("content", file), os.path.join("result", file.replace(".pdf", ".md")))
                    content = preprocess(content)
                    chunks = semantic_chunking(content)
                    data = []
                    # label = labelling()
                    for chunk in chunks:
                        data.append({"text": chunk.page_content})
                    df = pd.DataFrame(data)
                    path = os.path.join("result", file.replace(".pdf", ".csv"))
                    df.to_csv(path, index=False, encoding="utf-8-sig", sep=";")
        elif choice == 3:
            variations = input("Enter number of variations to paraphrase: ")
            data_augment(variations)
        elif choice == 4:
            print("Exiting the program.")
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
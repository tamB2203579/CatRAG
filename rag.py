from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from llama_index.core import Document
from dotenv import load_dotenv
import os
import re

from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TesseractOcrOptions
from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormatOption
from docling.datamodel.base_models import InputFormat

from embedding import Embedding
from vector_store import VectorStore

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

class RAG:
    def __init__(self, model_name="gpt-4o-mini"):
        # Initialize LLM
        if model_name == "gpt-4o-mini":
            self.llm = ChatOpenAI(model=model_name, temperature=0)
        else:
            self.llm = ChatMistralAI(model=model_name, temperature=0)
            
        # Initialize embedding model
        self.embedded_model = Embedding()
        
        # Initialize vector store
        self.vector_store = VectorStore(storage_dir="./storage")
        
        # Load stopwords
        with open("./lib/stopwords.txt", mode="r", encoding="utf-8") as f:
            self.stopwords = f.read().splitlines()

    def preprocess(self, text):
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
    
    def read_pdf_to_string(self, path):
        # Initialize document converter with proper options
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
        
        # Convert PDF to text
        result = converter.convert(path)
        extracted_text = result.document.export_to_text()
        
        return extracted_text
    
    def chunking(self, text):
        """
        Split text into chunks using RecursiveCharacterTextSplitter.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        chunks = text_splitter.create_documents([text])
        
        # Convert to llama-index Document format
        documents = [Document(text=chunk.page_content) for chunk in chunks]
        return documents
    
    def create_vector_stores(self, documents):
        """
        Create a vector store index from document chunks and save it to disk.
        """
        # Use the VectorStore class to create and save the vector index
        vector_index = self.vector_store.create_vector_store(documents)
        return vector_index

    def generate_response(self, query):
        """
        Generate a response using the RAG system.
        """
        # Get vector results
        vector_results = self.vector_store.get_vector_results(query, top_k=5)
        
        # Format vector context
        vector_context = "\n\n".join([
            f"Đoạn {i+1} (Điểm tương đồng: {result['score']:.4f}):\n{result['text']}"
            for i, result in enumerate(vector_results)
        ])

        # Load prompt template
        with open("prompt/rag_query.txt", "r", encoding="utf-8") as f:
            template = f.read()
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = chain.invoke({
            "query": query,
            "vector_context": vector_context,
        })
        
        return {
            "query": query,
            "response": response,
            "vector_context": vector_context,
        }
    
    def interactive_query(self):
        """
        Run an interactive query loop for the RAG system.
        """
        print("\nRAG Query System")
        print("Type 'q' to exit")
        
        while True:
            query = input("\nNhập câu hỏi của bạn: ")
            if query.lower() == "q":
                break
            
            result = self.generate_response(query)
            print("\nAnswer: ", result["response"])
            print("\nVector context: ", result["vector_context"])

from multilingual_pdf2text.models.document_model.document import Document
from multilingual_pdf2text.pdf2text import PDF2Text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from embedding import Embedding
from dotenv import load_dotenv
import os
import re

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

class RAG:
    def __init__(self, model_name):
        if model_name == "gpt-4o-mini":
            self.llm = ChatOpenAI(model=model_name, temperature=0)
        else:
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.embedded_model = Embedding()
        
        # Create directories if they don't exist
        os.makedirs("./faiss", exist_ok=True)
        
        # Load stopwords
        with open("./lib/stopwords.txt", mode="r", encoding="utf-8") as f:
            self.stopwords = f.read().splitlines()

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
    
    def read_pdf_to_string(self, path):
        # Initialize document objects
        pdf_document = Document(document_path=path, language="vie")
        pdf2text = PDF2Text(document=pdf_document)

        # Extract text from document objects
        extracted_text = ""
        for words in pdf2text.extract():
            extracted_text += words["text"]

        return extracted_text
    
    def chunking(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        return text_splitter.create_documents([text])
    
    def create_vector_stores(self, documents):
        # Use the embedding object properly
        vector_db = FAISS.from_documents(documents=documents, embedding=self.embedded_model)
        FAISS.save_local(vector_db, folder_path="./faiss/")
        return vector_db

    def generate_response(self, query):
        # Load the vector store
        vector_db = FAISS.load_local(folder_path="./faiss/", embeddings=self.embedded_model, allow_dangerous_deserialization=True)
        
        # Search for similar documents
        retrieved_docs = vector_db.similarity_search(query, k=5)

        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Create prompt for the LLM
        prompt = f"""
            Based only on the following information, please answer the query.
        
            Information: {context}
            Query: {query}
        
            Answer:
        """
        
        # Generate response using the LLM
        response = self.llm.invoke(prompt).content
        
        return response

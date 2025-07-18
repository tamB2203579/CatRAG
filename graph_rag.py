from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from llama_index.core import Document
from dotenv import load_dotenv
from uuid import uuid4
from glob import glob
import pandas as pd
import os

from knowledge_graph import KnowledgeGraph
from vector_store import VectorStore

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

class GraphRAG:
    def __init__(self, model_name="gpt-4o-mini"):
        # Initialize components
        self.knowledge_graph = KnowledgeGraph()
        self.vector_store = VectorStore()
        
        # Initialize LLM
        if model_name == "gpt-4o-mini":
            self.llm = ChatOpenAI(model=model_name, temperature=0)
        else:
            self.llm = ChatMistralAI(model=model_name, temperature=0)
            
    def load_excel_data(self, dir="data/processed"):
        """
        Load Excel files from a directory and combine them into a DataFrame.
        Each row gets a unique ID.
        """
        excel_files = glob(f"{dir}/*.xlsx")
        print(f"Found {len(excel_files)} Excel files in {dir}")

        all_dfs = []

        for file in excel_files:
            df = pd.read_excel(file, engine="openpyxl")
            print(f" - {file}: {len(df)} rows")
            all_dfs.append(df)

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"Combined dataset: {len(combined_df)} rows")

            combined_df["id"] = [str(uuid4()) for _ in range(len(combined_df))]

            return combined_df
        else:
            print("No data loaded.")
            return None
            
    def initialize_system(self, force_reinit=False):
        """
        Initialize the GraphRAG system:
        1. Load documents
        2. Split them into chunks
        3. Create vector store
        4. Build knowledge graph with entities and relationships
        """
        print("Initializing GraphRAG system...")
        
        # Load CSV data
        df = self.load_excel_data()
        
        if df is not None:
            documents = [Document(text=row['text'], id_=row['id'], metadata={"label": row['label']}) for _, row in df.iterrows()]
            print(f"Created {len(documents)} documents for indexing")
            
            if not documents:
                print("No documents created due to data loading issues")
                return
            
            # Create vector store
            self.vector_store.create_vector_store(documents)
            
            # Build knowledge graph
            # self.knowledge_graph.clear_database()
            # self.knowledge_graph.create_constraints()
            # self.knowledge_graph.build_knowledge_graph(documents)
            
            print("GraphRAG initialization completed successfully!")
        else:
            print("Skipping GraphRAG initialization due to missing data")
            
    def generate_response(self, query, label=None):
        """
        Generate a response using the GraphRAG system.
        """
        # Get vector results
        vector_results = self.vector_store.get_vector_results(query, top_k=5)
        
        # Format vector context
        vector_context = "\n\n".join([
            f"Đoạn {i+1} (Điểm tương đồng: {result['score']:.4f}):\n{result['text']}"
            for i, result in enumerate(vector_results)
        ])

        # Get graph context
        graph_context = self.knowledge_graph.get_graph_context(query, label=label)

        if not graph_context:
            graph_context = ""
        
        # Load prompt template
        with open("prompt/graphrag_query.txt", "r", encoding="utf-8") as f:
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
            "graph_context": graph_context,
            "label": label
        })
        
        return {
            "query": query,
            "response": response,
            "vector_context": vector_context,
            "graph_context": graph_context
        }

    def interactive_query(self, classifier=None):
        """
        Run an interactive query loop for the GraphRAG system.
        """
        print("\nGraphRAG Query System")
        print("Type 'q' to exit")
        
        while True:
            query = input("\nNhập câu hỏi của bạn: ")
            if query.lower() == "q":
                break
            
            label = None
            if classifier:
                try:
                    label = classifier.classify_text(query)
                    print(f"Classified as: {label}")
                except Exception as e:
                    print(f"Classification error: {e}")
            
            result = self.generate_response(query, label)
            print(result["response"])
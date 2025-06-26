from knowledge_graph import KnowledgeGraph
from vector_store import VectorStore
from history_manager import HistoryManager
from embedding import Embedding
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from llama_index.core import Document
import pandas as pd
import os
from uuid import uuid4

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

class GraphRAG:
    def __init__(self, model_name="gpt-4o-mini"):
        # Initialize components
        self.knowledge_graph = KnowledgeGraph()
        self.vector_store = VectorStore()
        self.history_manager = HistoryManager()
        
        # Initialize LLM
        if model_name == "gpt-4o-mini":
            self.llm = ChatOpenAI(model=model_name, temperature=0)
        else:
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
            
    def load_csv_data(self, dir="result"):
        """
        Load CSV files from a directory and combine them into a DataFrame.
        Each row gets a unique ID.
        """
        from glob import glob
        csv_files = glob(f"{dir}/*.csv")
        print(f"Found {len(csv_files)} CSV files in {dir}")

        all_dfs = []

        for csv_file in csv_files:
            df = pd.read_csv(csv_file, encoding="utf-8-sig", sep=";")
            print(f" - {csv_file}: {len(df)} rows")
            all_dfs.append(df)

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"Combined dataset: {len(combined_df)} rows")

            combined_df["id"] = [str(uuid4()) for _ in range(len(combined_df))]

            return combined_df
        else:
            print("No data loaded.")
            return None
            
    def initialize_system(self):
        """
        Initialize the GraphRAG system:
        1. Load documents
        2. Split them into chunks
        3. Create vector store
        4. Build knowledge graph with entities and relationships
        """
        print("Initializing GraphRAG system...")
        
        # Load CSV data
        df = self.load_csv_data()
        
        if df is not None:
            documents = [Document(text=row['text'], id_=row['id'], metadata={"label": row['label']}) for _, row in df.iterrows()]
            print(f"Created {len(documents)} documents for indexing")
            
            if not documents:
                print("No documents created due to data loading issues")
                return
            
            # Create vector store
            self.vector_store.create_vector_store(documents)
            
            # Build knowledge graph
            self.knowledge_graph.clear_database()
            self.knowledge_graph.create_constraints()
            self.knowledge_graph.build_knowledge_graph(documents)
            
            print("GraphRAG initialization completed successfully!")
        else:
            print("Skipping GraphRAG initialization due to missing data")
            
    def generate_response(self, query, session_id, label=None):
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
        
        # Get conversation history
        history = self.history_manager.get_history(session_id)
        past_query = " ".join(entry["query"] for entry in history)
        past_responses = " ".join(entry["response"] for entry in history)
        
        # Load prompt template
        with open("prompt/query.txt", "r", encoding="utf-8") as f:
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
            "past_query": past_query,
            "past_response": past_responses,
            "label": label
        })
        
        # Add to history
        self.history_manager.add_to_history(session_id, query, response)
        
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
        
        # Generate a new session ID
        session_id = self.history_manager.generate_session_id()
        
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
            
            result = self.generate_response(query, session_id, label)
            print("\nAnswer:")
            print(result["response"])

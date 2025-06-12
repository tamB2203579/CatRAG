from llama_index.core import Document
from helper_functions import *
import os

def initialize_graphrag():
    """
    Initialize the GraphRAG system:
    1. Load documents
    2. Split them into chunks
    3. Create vector store
    4. Build knowledge graph with entities and relationships
    """
    print("Initializing GraphRAG system...")
    
    # Load CSV data
    df = load_csv_data()
    
    if df is not None:
        documents = [Document(text=row['text'], id_=row['id'], metadata={"label": row['label']}) for _, row in df.iterrows()]
        print(f"Created {len(documents)} documents for indexing")
        
        if not documents:
            print("No documents created due to data loading issues")
            return
        
        # create_vector_store(documents)
        
        build_knowledge_graph(documents)
        
        print("GraphRAG initialization completed successfully!")
    else:
        print("Skipping GraphRAG initialization due to missing data")

def interactive_query():
    """
    Run an interactive query loop for the GraphRAG system.
    """
    print("\nGraphRAG Query System")
    print("Type 'q' to exit")
    
    while True:
        query = input("\nNhập câu hỏi của bạn: ")
        if query.lower() == "q":
            break
        
        result = graphrag_chatbot(query)
        print("\nAnswer:")
        print(result["response"])

def main_menu():
    """
    Display the main menu for the GraphRAG system.
    """
    while True:
        print("\nGraphRAG System Menu")
        print("1. Interactive Query (Question Answering)")
        print("2. Similarity Search")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            interactive_query()
        elif choice == "2":
            print("Exiting GraphRAG system. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    if not os.path.exists("storage") or input("Do you want to reinitialize the GraphRAG system? (y/n): ").lower() == "y":
        initialize_graphrag()
    else:
        print("Using existing GraphRAG system.")
    
    main_menu()

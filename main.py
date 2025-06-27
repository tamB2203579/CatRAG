from classify import main as classify_module
from history_manager import HistoryManager
from graph_rag import GraphRAG
import os

def main():
    """
    Main entry point for the GraphRAG system.
    """
    # Initialize the GraphRAG system
    name = input("Enter the LLM model you want to use: ")
    graph_rag = GraphRAG(model_name=name)
    history_manager = HistoryManager()
    
    # Check if we need to initialize the system
    if not os.path.exists("storage") or input("Do you want to reinitialize the GraphRAG system? (y/n): ").lower() == "y":
        graph_rag.initialize_system()
    else:
        print("Using existing GraphRAG system.")
    
    # Display the main menu
    while True:
        print("\nGraphRAG System Menu")
        print("1. Interactive Query (Question Answering)")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ")
        
        if choice == "1":
            graph_rag.interactive_query(classifier=classify_module)
        elif choice == "2":
            print("Exiting GraphRAG system. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

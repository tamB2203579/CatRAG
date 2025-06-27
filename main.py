from classify import main as classify_module
from history_manager import HistoryManager
from graph_rag import GraphRAG
import os

def main():
    """
    Main entry point for the GraphRAG system.
    """
    # Initialize the GraphRAG system
    graph_rag = GraphRAG(model_name="gemini-2.5-flash")
    history_manager = HistoryManager()
    
    # Check if we need to initialize the system
    if not os.path.exists("storage") or input("Do you want to reinitialize the GraphRAG system? (y/n): ").lower() == "y":
        graph_rag.initialize_system()
    else:
        print("Using existing GraphRAG system.")
    
    # Load a specific session history if needed
    history_manager.load_history_from_file("6c61ff71-2ba5-42cb-a4e6-abd901d7ebf9")
    
    # Display the main menu
    while True:
        print("\nGraphRAG System Menu")
        print("1. Interactive Query (Question Answering)")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ")
        
        if choice == "1":
            try:
                # Attempt to import and use the classifier
                graph_rag.interactive_query(classifier=classify_module)
            except ImportError:
                # Fall back to not using classification if the module isn't available
                print("Classification module not available. Proceeding without classification.")
                graph_rag.interactive_query()
        elif choice == "2":
            print("Exiting GraphRAG system. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

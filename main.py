from classify import classify_text
from rag import RAG
import os

def main():
    """
    Main entry point for the GraphRAG system.
    """
    # Initialize the GraphRAG system
    print("1. gpt-4o-mini")
    print("2. mistral-small-2506")
    choice = input("Enter the LLM model you want to use (1/2): ")
    
    name = "gpt-4o-mini" if choice == "1" else "mistral-small-2506"
    rag = RAG(model_name=name)
    
    # Display the main menu
    while True:
        print("\nGraphRAG System Menu")
        print("1. Interactive Query (Question Answering)")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ")
        
        if choice == "1":
            rag.interactive_query()
        elif choice == "2":
            print("Exiting GraphRAG system. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

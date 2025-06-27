from rag import RAG
import os

def main():
    print("=" * 50)
    print("PDF Question Answering System")
    print("=" * 50)
    
    # Get PDF path
    while True:
        pdf_path = input("Enter path to PDF file: ")
        if os.path.exists(pdf_path):
            break
        else:
            print(f"Error: PDF file not found at {pdf_path}")
    
    # Select model
    print("\nAvailable models:")
    print("1. gpt-4o-mini")
    print("2. gemini-2.5-flash (default)")
    
    model_choice = input("Select model (1/2): ").strip() or "2"
    model = "gpt-4o-mini" if model_choice == "1" else "gemini-2.5-flash"
    
    # Initialize the RAG system with the selected model
    rag = RAG(model)
    
    print(f"\nProcessing PDF: {pdf_path}")
    
    # Read PDF content
    pdf_text = rag.read_pdf_to_string(pdf_path)
    
    # Chunk the document
    chunks = rag.chunking(pdf_text)
    
    # Create vector store
    rag.create_vector_stores(chunks)
    
    # Query loop
    while True:
        print("\n" + "-" * 50)
        query = input("Enter your question (or 'exit' to quit): ")
        
        if query.lower() == 'exit':
            break
            
        if query.strip():
            response = rag.generate_response(query)
            print("\nResponse:")
            print(response)
        else:
            print("Please enter a valid question.")

if __name__ == "__main__":
    main()
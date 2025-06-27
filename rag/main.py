from rag import RAG
import os
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run RAG on PDF documents")
    parser.add_argument("--pdf", help="Path to PDF file", required=True)
    parser.add_argument("--model", help="Model name (gpt-4o-mini or gemini-pro)", 
                        default="gemini-2.5-flash", choices=["gpt-4o-mini", "gemini-2.5-flash"])
    parser.add_argument("--query", help="Query to ask", required=True)
    args = parser.parse_args()

    # Initialize the RAG system with the specified model
    rag = RAG(args.model)
    
    # Process PDF if it exists
    if os.path.exists(args.pdf):
        print(f"Processing PDF: {args.pdf}")
        # Read PDF content
        pdf_text = rag.read_pdf_to_string(args.pdf)
        
        # Chunk the document
        chunks = rag.chunking(pdf_text)
        
        # Create vector store
        rag.create_vector_stores(chunks)
        
        # Generate response to query
        print(f"\nQuery: {args.query}")
        response = rag.generate_response(args.query)
        print("\nResponse:")
        print(response)
    else:
        print(f"Error: PDF file not found at {args.pdf}")

if __name__ == "__main__":
    main()
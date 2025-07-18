from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
import os

import logging
logging.getLogger("llama_index").setLevel(logging.ERROR)

class VectorStore:
    def __init__(self, storage_dir="./storage"):
        self.storage_dir = storage_dir
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
    def create_vector_store(self, docs):
        """
        Create a vector store index from document chunks and save it to disk.
        """
        # Create vector store index
        print("Creating vector index...")
        vector_index = VectorStoreIndex.from_documents(
            docs,
            show_progress=True
        )

        # Save the index to disk
        print("Saving vector index to storage...")
        vector_index.storage_context.persist(persist_dir=self.storage_dir)
        print("Vector store created and saved successfully!")
        
        return vector_index
        
    def load_vector_store(self):
        """
        Load a previously saved vector store index from disk.
        """
        if not os.path.exists(self.storage_dir):
            raise FileNotFoundError(f"Storage directory '{self.storage_dir}' does not exist.")
            
        storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
        loaded_index = load_index_from_storage(storage_context)
        return loaded_index
        
    def get_vector_results(self, query, top_k=5):
        """
        Get relevant results from the vector store based on the query.
        """
        index = self.load_vector_store()
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        results = []
        for i, node in enumerate(nodes):
            results.append({
                "text": node.node.text,
                "score": node.score,
                "id": node.node.id_,
                "metadata": node.node.metadata
            })
        
        return results

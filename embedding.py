from transformers import AutoModel, AutoTokenizer
from langchain.embeddings.base import Embeddings
from typing import List
import torch.nn as nn
import numpy as np

class Embedding(Embeddings):
    def __init__(self, model_name="vinai/phobert-base", output_dim=1536):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # # Get the original embedding dimension
        # original_dim = self.model.config.hidden_size
        
        # # Create a projection layer to transform to 1536 dimensions
        # self.projection = nn.Linear(original_dim, output_dim)
        
    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        outputs = self.model(**inputs)

        # # Get the original embeddings
        # original_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # # Project to 1536 dimensions
        # projected_embeddings = self.projection(original_embeddings).detach().numpy()
        # return projected_embeddings

        embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
        return embeddings

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        embeddings = self.encode(docs)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.encode([text])[0]
        return embedding.tolist()
import os
import torch
import warnings
from transformers import AutoModel, AutoTokenizer
from langchain.embeddings.base import Embeddings

warnings.filterwarnings("ignore", message="Some weights of.*pooler.*")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

class Embedding(Embeddings):
    def __init__(self, model_name="vinai/phobert-base-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False).to(self.device)
        print(f"▶ Embedding model loaded on device: {self.device}")

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        return embeddings

    def embed_query(self, text):
        embedding = self.encode([text])[0]
        return embedding.tolist()
    
    def embed_documents(self, docs):
        embeddings = self.encode(docs)
        return embeddings.tolist()
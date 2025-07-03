from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Document
from pydantic import BaseModel
from typing import List, Dict
from fastapi import FastAPI
import os

from graph_rag import GraphRAG 
from classify.main import classify_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize GraphRAG instance
graph_rag = GraphRAG()
responses: List[Dict] = []

class ResponseData(BaseModel):
    query: str
    response: str = ""
    session_id: str

def answer_query(query: str, session_id: str) -> str:
    label = classify_text(query)
    result = graph_rag.generate_response(query=query, label=label, session_id=session_id)
    return result["response"]

@app.post("/ask")
async def ask_question(data: ResponseData):
    response_text = answer_query(data.query, data.session_id)

    result = {
        "query": data.query,
        "response": response_text,
        "session_id": data.session_id
    }

    responses.append(result)
    return {"status": "success", "data": result}

@app.get("/responses")
async def get_responses():
    return {"status": "success", "responses": responses}

if __name__ == "__main__":
    import uvicorn
    # graph_rag.initialize_system()
    uvicorn.run(app="api:app", host="localhost", port=8000)
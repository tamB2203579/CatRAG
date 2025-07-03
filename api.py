from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os
from llama_index.core import Document
from helper_functions import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

responses: List[Dict] = []


class ResponseData(BaseModel):
    query: str
    response: str = ""
    session_id: str

def initialize_graphrag(force_reinit=False):
    if os.path.exists("storage") and not force_reinit:
        print("Using existing GraphRAG system.")
        return

    print("Initializing GraphRAG system...")

    df = load_csv_data()
    if df is not None:
        documents = [
            Document(text=row['text'], id_=row['id'], metadata={"label": row['label']})
            for _, row in df.iterrows()
        ]
        print(f"Created {len(documents)} documents for indexing")

        if not documents:
            print("No documents created due to data loading issues")
            return

        # Uncomment nếu bạn muốn xử lý
        # create_vector_store(documents)
        # build_knowledge_graph(documents)

        print("GraphRAG initialization completed successfully!")
    else:
        print("Skipping GraphRAG initialization due to missing data")

def answer_query(query: str, session_id: str) -> str:
    load_history_from_file(session_id)
    result = graphrag_chatbot(query, session_id)
    return result["response"]


@app.on_event("startup")
def startup_event():
    initialize_graphrag(force_reinit=False)


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


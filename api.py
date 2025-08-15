from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict
from fastapi import FastAPI
import time
import os

from graph_rag import GraphRAG 
from rag import RAG
from classify.main import classify_text
import rag

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the frontend
static_dir = os.path.join(os.path.dirname(__file__), "web_application", "dist")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


# Initialize GraphRAG instance
# graphrag = GraphRAG(model_name="gpt-4o-mini")
rag = RAG(model_name="mistral-small-2506")
responses: List[Dict] = []
first_query_processed = False

class ResponseData(BaseModel):
    query: str
    response: str = ""
    session_id: str

def answer_query(query: str) -> str:
    start_time = time.time()
    # label = classify_text(query)
    # result = graphrag.generate_response(query=query, label=label)
    result = rag.generate_response(query=query)
    print(time.time() - start_time)
    return result["response"]

@app.post("/ask")
async def ask_question(data: ResponseData):
    response_text = answer_query(data.query)

    result = {
        "query": data.query,
        "response": response_text,
    }

    responses.append(result)
    return {"status": "success", "data": result}

@app.get("/responses")
async def get_responses():
    return {"status": "success", "responses": responses}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="api:app", host="localhost", port=8000)
from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv
from uuid import uuid4
from glob import glob
from tqdm import tqdm
import pandas as pd
import warnings
import datetime
import spacy
import json
import os
import re

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")

# Model Setup
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Neo4j Setup
neo4j_url = "bolt://localhost:7687"
neo4j_username = "neo4j"
neo4j_password = "123123123"
driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_password))

# Llamaindex Setting
Settings.embed_model = embed_model

# Load Vietnamese spaCy model
nlp = spacy.load("vi_core_news_lg")

history = {}

def load_csv_data(dir="result"):
    csv_files = glob(f"{dir}/*.csv")
    print(f"Found {len(csv_files)} CSV files in {dir}")

    all_dfs = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding="utf-8-sig", sep=";")
        print(f" - {csv_file}: {len(df)} rows")
        all_dfs.append(df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Combined dataset: {len(combined_df)} rows")

        combined_df["id"] = [str(uuid4()) for _ in range(len(combined_df))]

        return combined_df
    else:
        print("No data loaded.")
        return None
    
def create_vector_store(documents):
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    
    # Create vector store index
    print("Creating vector index...")
    vector_index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
        transformations=[text_splitter]
    )

    # Save the index to disk
    print("Saving vector index to storage...")
    vector_index.storage_context.persist(persist_dir="./storage")

    print("Vector store created and saved successfully!")

def load_vector_store():
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    loaded_index = load_index_from_storage(storage_context)
    return loaded_index

def clear_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared")

def create_constraints():
    with driver.session() as session:
        # Create chunk constrains
        session.run(
            """
            CREATE CONSTRAINT chunk_id IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """
        )
        # Create entitiy contraints
        session.run(
            """
            CREATE CONSTRAINT entity_name IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.name IS UNIQUE
            """
        )
        # Create category constraints
        session.run(
            """
            CREATE CONSTRAINT category_name IF NOT EXISTS
            FOR (c:Category) REQUIRE c.name IS UNIQUE
            """
        )

        print("Constraints created")

def normalize(text: str):
    replacements = {
        'àáảãạăằắẳẵặâầấẩẫậ': 'a',
        'èéẻẽẹêềếểễệ': 'e',
        'ìíỉĩị': 'i',
        'òóỏõọôồốổỗộơờớởỡợ': 'o',
        'ùúủũụưừứửữự': 'u',
        'ỳýỷỹỵ': 'y',
        'đ': 'd'
    }

    result = text.lower()
    for chars, replacement in replacements.items():
        for c in chars:
            result = result.replace(c, replacement)

    result = re.sub(r'[^\w\s]', '', result)
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def extract_entities(text: str):
    doc = nlp(text)
    entities = [ent.text for ent in doc]
    entities = list(set([e.strip() for e in entities]))
    return entities

def extract_relationships(text, entities):
    with open("prompt/extract_relationships.txt", mode="r", encoding="utf-8") as f:
        template = f.read()
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({
        "text": text,
        "entities": ", ".join(entities) 
    })

    try:
        cleaned_response = response
        if "```json" in cleaned_response:
            cleaned_response = cleaned_response.replace("```json", "").replace("```", "")
        cleaned_response = cleaned_response.strip()

        relationships = json.loads(cleaned_response)
        return relationships
    except Exception as e:
        print(f"Failed to parse relationships JSON: {e}")
        return []
    
def populate_graph(df: pd.DataFrame):
    clear_database()
    create_constraints()

    with driver.session() as session:
        categories = df["label"].str.replace("__label__", "").unique()

        for category in categories:
            session.run("""
                MERGE (c:Category {name: $name})
            """, name=category)
        
        print(f"Created {len(categories)} category nodes")

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Building knowledge graph in Neo4j..."):
            text = row["text"]
            chunk_id = row["id"]
            category = row["label"].replace("__label__", "")

            entities = extract_entities(text)
            relationships = extract_relationships(text, entities)

            if relationships:
                rels_dict = {
                    "chunk_id": chunk_id,
                    "text": text,
                    "category": category,
                    "relationships": relationships
                }

                if not os.path.exists("relationships"):
                    os.makedirs("relationships")
                
                with open(f"relationships/{chunk_id}.json", "w", encoding="utf-8") as f:
                    json.dump(rels_dict, f, ensure_ascii=False, indent=2)
                
            # Create chunk nodes:
            session.run("""
                MERGE (c:Chunk {id: $id})
                SET c.text = $text, c.category = $category
            """, id=chunk_id, text=text, category=category)

            # Connect chunk with cateogory
            session.run("""
                MATCH (chunk:Chunk {id: $chunk_id})
                MATCH (category:Category {name: $category})
                MERGE (chunk)-[:BELONGS_TO]->(category)
            """, chunk_id=chunk_id, category=category)

            # Create entity nodes and connect to chunk
            for entity in entities:
                # Create entity nodes
                session.run("""
                    MERGE (e:Entity {name: $name})
                """, name=entity)

                # Connection entity to chunk
                session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (e:Entity {name: $entity_name})
                    MERGE (c)-[:CONTAINS]->(e)
                """, chunk_id=chunk_id, entity_name=entity)

            # Create relationships between entities
            for rel in relationships:
                source = rel["source"]
                target = rel["target"]
                relationship = normalize(rel["relationship"].upper().replace(" ", "_"))
            
                if source != target:
                    # Create relationship in Neo4j
                    try:
                        session.run(f"""
                            MATCH (s:Entity {{name: $source}})
                            MATCH (t:Entity {{name: $target}})
                            MERGE (s)-[r:{relationship}]->(t)
                        """)
                    except Exception as e:
                        session.run("""
                            MATCH (s:Entity {name: $source})
                            MATCH (t:Entity {name: $target})
                            MERGE (s)-[r:RELATED_TO]->(t)
                            SET r.type = $relationship
                        """, source=source, target=target, relationship=rel["relationship"])

def get_graph_context(query: str, limit=5):
    query_terms = query.split()

    with driver.session() as session:
        # First approach: Find entities mentioned in the query
        entities = session.run("""
            MATCH (e:Entity)
            WHERE any(term IN $query_terms WHERE e.name CONTAINS term OR $query_text CONTAINS e.name)
            RETURN e.name AS entity
            LIMIT $limit
        """, query_terms=query_terms, query_text=query, limit=limit).values()

        if not entities:
            return "No relevant graph context found."
        
        context_parts = []

        for entity_tuple in entities:
            entity_name = entity_tuple[0]

            # Find relationships for this entity
            relationships = session.run("""
                MATCH (e:Entity {name: $entity_name})-[r]-(other:Entity)
                RETURN type(r) AS relationship, 
                    e.name AS source, 
                    other.name AS target,
                    r.type AS rel_type
                LIMIT 5
            """, entity_name=entity_name).values()

            # Initialize entity_context
            entity_context = f"Entity: {entity_name}\n"

            if relationships:
                entity_context += "Relationships:\n"
                for rel_type, source, target, rel_name in relationships:
                    relationship = rel_name if rel_type == "RELATED_TO" else rel_type.lower().replace("_", " ")
                    entity_context += f" - {source} {relationship} {target}\n"
            
            context_parts.append(entity_context)
        
    return "\n\n".join(context_parts)

def get_vector_results(query, top_k=5):
    index = load_vector_store()

    retriever = index.as_retriever(kwargs=top_k)

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

def generate_session_id():
    return str(uuid4())

def add_to_history(session_id, query, response):
    global history
    max_entries = 10
    
    if session_id not in history:
        history[session_id] = []
    
    entry = {
        "query": query,
        "response": response,
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    if len(history[session_id]) >= max_entries:
        history[session_id].pop(0)
    history[session_id].append(entry)
    save_history_to_file(session_id)  
    
def save_history_to_file(session_id):
    
    filename = f"history/chat_history_{session_id}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(history.get(session_id, []), file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save history to {filename}: {e}")
    
def load_history_from_file(session_id):
    
    filename = f"history/chat_history_{session_id}.json"
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            
            loaded_history = json.load(file)
            history[session_id] = loaded_history[-10:]
            
        print(f"History loaded from {filename} successfully!")
        
    except FileNotFoundError:
        print(f"No history file found at {filename}. Starting with empty history.")
        history[session_id] = []
    except Exception as e:
        print(f"Failed to load history from {filename}: {e}")
        history[session_id] = []

def clear_history(session_id):
    filename = f"history/chat_history_{session_id}.json"
    if session_id in history:
        history[session_id].clear()
        
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump([], file, indent=4)
        print(f"History and file {filename} have been cleared successfully!")

def graphrag_chatbot(query,session_id):
    with open("prompt/query.txt", "r", encoding="utf-8") as f:
        template = f.read()
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    vector_results = get_vector_results(query, top_k=5)
    # Format vector context
    vector_context = "\n\n".join([
        f"Đoạn {i+1} (Điểm tương đồng: {result['score']:.4f}):\n{result['text']}"
        for i, result in enumerate(vector_results)
    ])

    graph_context = get_graph_context(query)
  
    past_query = " ".join(entry["query"] for entry in history[session_id])
    past_responses = " ".join(entry["response"] for entry in history[session_id])

    response = chain.invoke({
        "query": query,
        "vector_context": vector_context,
        "graph_context": graph_context,
        "past_query": past_query,
        "past_response": past_responses
    })
    
    add_to_history(session_id, query, response)
    

    return {
        "query": query,
        "response": response,
        # "vector_results": vector_results,
        "graph_context": graph_context
    }
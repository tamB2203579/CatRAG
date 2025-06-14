from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from neo4j import GraphDatabase
from dotenv import load_dotenv
from uuid import uuid4
from glob import glob
from tqdm import tqdm
import pandas as pd
import warnings
import spacy
import json
import os
import re

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")

# Model Setup
embed_model = OpenAIEmbeddings()

# Neo4j Setup
neo4j_url = "bolt://localhost:7687"
neo4j_username = "neo4j"
neo4j_password = "123123123"
driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_password), database="graphrag")

# Load Vietnamese spaCy model
nlp = spacy.load("vi_core_news_lg")

def load_csv_data(dir="semantic_result"):
    """
    Load CSV files from a directory and combine them into a DataFrame.
    Each row gets a unique ID.
    """
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

def create_vector_store(docs):
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
    vector_index.storage_context.persist(persist_dir="./storage")
    print("Vector store created and saved successfully!")

def load_vector_store():
    """
    Load a previously saved vector store index from disk.
    """
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    loaded_index = load_index_from_storage(storage_context)
    return loaded_index

def clear_database():
    """
    Clear all nodes and relationships from the Neo4j database.
    """
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared")

def create_constraints():
    """
    Create Neo4j constraints for the graph database.
    """
    with driver.session() as session:
        # Create chunk constraints
        session.run(
            """
            CREATE CONSTRAINT chunk_id IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """
        )
        
        # Create entity constraints
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
    """
    Normalize Vietnamese text by removing diacritics and special characters.
    """
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
    """
    Extract entities from text using spaCy.
    """
    doc = nlp(text)
    entities = [ent.text for ent in doc]
    
    entities = list(set([e.strip() for e in entities]))
    return entities

def extract_relationships(text, entities):
    """
    Extract relationships between entities using LLM.
    """
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

def build_knowledge_graph(chunks):
    """
    Build a knowledge graph in Neo4j
    """
    # Clear database and create constraints
    # clear_database()
    # create_constraints()

    with driver.session() as session:
        print("Creating Chunk nodes")
        for chunk in tqdm(chunks, desc="Creating Chunk nodes"):
            # Create chunk node
            session.run("""
            CREATE (c:Chunk {
                id: $id,
                text: $text,
                label: $label
            })
            """, id=chunk.id_, text=chunk.text, label=chunk.metadata.get("label", ""))
            
            # Create category node if a label exists
            if chunk.metadata.get("label"):
                # Create category node
                session.run("""
                    MERGE (cat:Category {name: $label})
                """, label=chunk.metadata.get("label", ""))
            
            # Connect chunk to category
            session.run("""
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (cat:Category {name: $label})
                MERGE (c)-[:BELONGS_TO]->(cat)
            """, chunk_id=chunk.id_, label=chunk.metadata.get("label", ""))
        
        print("Extracting entities and relationships from chunks...")
        for chunk in tqdm(chunks, desc="Processing entities and relationships"):
            # Extract entities
            entities = extract_entities(chunk.text)
            
            # Create entity nodes and connect to chunk
            for entity in entities:
                embedding = embed_model.embed_query(entity)
                # Create entity node
                session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.embedding = $embedding
                """, name=entity, embedding=embedding)
                
                # Connect entity to chunk
                session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (e:Entity {name: $entity_name})
                    MERGE (c)-[:MENTIONS]->(e)
                """, chunk_id=chunk.id_, entity_name=entity)
            
            # Extract relationships between entities
            relationships = extract_relationships(chunk.text, entities)
            
            # Save relationships to file for later reference
            if relationships:
                rels_dict = {
                    "chunk_id": chunk.id_,
                    "text": chunk.text,
                    "entities": entities,
                    "relationships": relationships
                }
                
                if not os.path.exists("relationships"):
                    os.makedirs("relationships")
                
                with open(f"relationships/{chunk.id_}.json", "w", encoding="utf-8") as f:
                    json.dump(rels_dict, f, ensure_ascii=False, indent=2)
            
            # Create relationships in Neo4j
            for rel in relationships:
                source = rel["source"]
                target = rel["target"]
                relationship = normalize(rel["relationship"].upper().replace(" ", "_"))
                
                if source != target:
                    try:
                        # Try to create a typed relationship
                        session.run(f"""
                            MATCH (s:Entity {{name: $source}})
                            MATCH (t:Entity {{name: $target}})
                            MERGE (s)-[r:{relationship}]->(t)
                            SET r.chunk_id = $chunk_id
                        """, source=source, target=target, chunk_id=chunk.id_)
                    except Exception as e:
                        # Fall back to generic relationship
                        session.run("""
                            MATCH (s:Entity {name: $source})
                            MATCH (t:Entity {name: $target})
                            MERGE (s)-[r:RELATED_TO]->(t)
                            SET r.relationship = $relationship,
                                r.chunk_id = $chunk_id
                        """, source=source, target=target, relationship=rel["relationship"], chunk_id=chunk.id_)

def get_graph_context(query: str, limit=5):
    """
    Get relevant context from the knowledge graph based on the query.
    """
    # Generate embedding for the query
    query_embedding = embed_model.embed_query(query)

    with driver.session() as session:
        # Use vector similarity to find relevant entities
        entities = session.run("""
            MATCH (e:Entity)
            WHERE e.embedding IS NOT NULL
            WITH e, gds.similarity.cosine(e.embedding, $query_embedding) AS score
            WHERE score > 0.75
            RETURN e.name AS entity, score
            ORDER BY score DESC
            LIMIT $limit
        """, query_embedding=query_embedding, limit=limit).values()
        
        # If no entities found with vector search, fall back to text matching
        if not entities:
            query_terms = query.split()
            entities = session.run("""
                MATCH (e:Entity)
                WHERE any(term IN $query_terms WHERE e.name CONTAINS term OR $query_text CONTAINS e.name)
                RETURN e.name AS entity, 1.0 AS score
                LIMIT $limit
            """, query_terms=query_terms, query_text=query, limit=limit).values()
        
        if not entities:
            return "No relevant graph context found."
        
        context_parts = []

        for entity_tuple in entities:
            entity_name = entity_tuple[0]
            entity_score = entity_tuple[1]

            # Find relationships for this entity
            relationships = session.run("""
                MATCH (e:Entity {name: $entity_name})-[r]-(other:Entity)
                RETURN type(r) AS relationship, 
                    e.name AS source, 
                    other.name AS target,
                    CASE WHEN type(r) = 'RELATED_TO' THEN r.relationship ELSE null END AS rel_type
                LIMIT 5
            """, entity_name=entity_name).values()

            # Find chunks mentioning this entity
            chunks = session.run("""
                MATCH (e:Entity {name: $entity_name})<-[:MENTIONS]-(c:Chunk)
                RETURN c.text AS chunk_text
                LIMIT 2
            """, entity_name=entity_name).values()

            # Initialize entity_context with similarity score
            entity_context = f"Entity: {entity_name} (Similarity: {entity_score:.4f})\n"

            if relationships:
                entity_context += "Relationships:\n"
                for rel_type, source, target, rel_name in relationships:
                    relationship = rel_name if rel_type == "RELATED_TO" else rel_type.lower().replace("_", " ")
                    entity_context += f" - {source} {relationship} {target}\n"
            
            if chunks:
                entity_context += "\nRelevant Context:\n"
                for chunk_tuple in chunks:
                    entity_context += f" - {chunk_tuple[0][:150]}...\n"
            
            context_parts.append(entity_context)
        
    return "\n\n".join(context_parts)

def get_vector_results(query, top_k=5):
    """
    Get relevant results from the vector store based on the query.
    """
    index = load_vector_store()
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

def graphrag_chatbot(query):
    """
    A chatbot that combines vector search and graph context for answering queries.
    """
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

    response = chain.invoke({
        "query": query,
        "vector_context": vector_context,
        "graph_context": graph_context
    })

    return {
        "query": query,
        "response": response,
        "graph_context": graph_context
    }

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from embedding import Embedding
from neo4j import GraphDatabase
from underthesea import ner
from tqdm import tqdm
import re
import json
import os

class KnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="123123123", database="graphrag"):  # Change the datbase name please
        self.driver = GraphDatabase.driver(uri, auth=(username, password), database=database)
        self.embed_model = Embedding()
    
    def close(self):
        self.driver.close()
        
    def clear_database(self):
        """Clear all nodes and relationships from the Neo4j database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared")
            
    def create_constraints(self):
        """Create Neo4j constraints for the graph database."""
        with self.driver.session() as session:
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
            
    def normalize(self, text: str):
        """Normalize Vietnamese text by removing diacritics and special characters."""
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
        
    def extract_entities(self, text: str):
        """Extract noun entities from text using underthesea."""
        docs = ner(text)
        noun_tags = ['N', 'Nb', 'Np', 'Ny']
        entities = [ent[0] for ent in docs if ent[1] in noun_tags]
        
        return list(set(entities))

    def extract_relationships(self, text, entities, use_cache=True):
        """Extract relationships between entities using LLM."""
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
            relationships = json.loads(response)
            return relationships
        except Exception as e:
            print(f"Failed to parse relationships JSON: {e}")
            return []
            
    def build_knowledge_graph(self, chunks, use_cache=True):
        """Build a knowledge graph in Neo4j"""
        with self.driver.session() as session:
            print("Creating chunks in Neo4j...")
            for chunk in tqdm(chunks, desc="Creating chunk nodes"):
                # Get embedding for the chunk
                embeddings = self.embed_model.embed_query(chunk.text)

                # Extract and clean label from metadata - handle non-string types
                label_value = chunk.metadata.get("label", "")
                label = str(label_value).replace("__label__", "").strip() if label_value is not None else ""

                # Create chunk node with embedding
                session.run("""
                    MERGE (c:Chunk {id: $id})
                    SET c.text = $text,
                    c.label = $label,
                    c.embeddings = $embedding
                """, id=chunk.id_, text=chunk.text, label=label, embedding=embeddings)
                
                # Create category node if a label exists
                if label:
                    # Create category node
                    session.run("""
                        MERGE (cat:Category {name: $label})
                    """, label=label)
                
                # Connect chunk to category
                session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (cat:Category {name: $label})
                    MERGE (c)-[:BELONGS_TO]->(cat)
                """, chunk_id=chunk.id_, label=label)
            
            print("Extracting entities and relationships from chunks...")
            for chunk in tqdm(chunks, desc="Processing entities and relationships"):
                # Extract entities
                entities = self.extract_entities(chunk.text)
                
                # Create entity nodes and connect to chunk
                for entity in entities:
                    embedding = self.embed_model.embed_query(entity)
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
                relationships = self.extract_relationships(chunk.text, entities)
                
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
                    source = rel["subject"]
                    target = rel["object"]
                    relationship = self.normalize(rel["predicate"].upper())
                    
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
                            """, source=source, target=target, relationship=rel["predicate"], chunk_id=chunk.id_)

    def get_graph_context(self, query: str, limit=5, entity_limit=10, label=None):
        """Get relevant context from the knowledge graph based on the query."""
        # Generate embedding for the query
        query_embedding = self.embed_model.embed_query(query)

        with self.driver.session() as session:
            # 1. Retrieve the most relevant chunks based on embedding similarity
            if label:
                chunks = session.run("""
                    MATCH (c:Chunk)-[:BELONGS_TO]->(cat:Category {name: $label})
                    WHERE c.embeddings IS NOT NULL
                    WITH c, gds.similarity.cosine(c.embeddings, $query_embedding) AS score
                    WHERE score > 0.7
                    RETURN c.id AS chunk_id, c.text AS chunk_text, score
                    ORDER BY score DESC
                    LIMIT $limit
                """, query_embedding=query_embedding, label=label, limit=limit).values()
            else:
                chunks = session.run("""
                    MATCH (c:Chunk)
                    WHERE c.embeddings IS NOT NULL
                    WITH c, gds.similarity.cosine(c.embeddings, $query_embedding) AS score
                    WHERE score > 0.7
                    RETURN c.id AS chunk_id, c.text AS chunk_text, score
                    ORDER BY score DESC
                    LIMIT $limit
                """, query_embedding=query_embedding, limit=limit).values()
            
            # Format chunks context
            chunks_context = "Relevant chunks context: "
            for _, text, score in chunks:
                chunks_context += f"\n- {text} (relevance: {score:.2f})"
            
            # 2. Get entities from retrieved chunks
            chunk_ids = [chunk_id for chunk_id, _, _ in chunks]
            entities = []
            
            if chunk_ids:
                entities_result = session.run("""
                    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                    WHERE c.id IN $chunk_ids
                    RETURN DISTINCT e.name AS entity_name
                """, chunk_ids=chunk_ids).values()
                
                entities = [name[0] for name in entities_result]
            
            if  not entities:
                rels_context = "\nNo entities found for the query."
            else:
                relationships = session.run("""
                    MATCH (e1:Entity)-[r]-(e2:Entity)
                    WHERE e1.name IN $entity_names
                    RETURN DISTINCT type(r) AS relationship, e1.name AS from, e2.name AS to
                """, entity_names=entities).values()
                    
                rels_context = "Relationships between entities in those 5 relevant chunks: "
                if relationships:
                    for rel, from_entity, to_entity in relationships:
                        rels_context += f"\n- {from_entity} -- {rel} --> {to_entity}"
                else:
                    rels_context += "\nNo relationships found for the entities."
            
            return chunks_context + "\n\n" + rels_context
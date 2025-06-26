from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from embedding import Embedding
from neo4j import GraphDatabase
from tqdm import tqdm
import spacy
import json
import os
import re

class KnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="123123123", database="graphrag"):
        self.driver = GraphDatabase.driver(uri, auth=(username, password), database=database)
        self.embed_model = Embedding()
        self.nlp = spacy.load("vi_core_news_lg")
        
        # Ensure relationships directory exists
        os.makedirs("relationships", exist_ok=True)
        
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
        """Extract entities from text using spaCy."""
        doc = self.nlp(text)
        entities = [ent.text for ent in doc]
        
        entities = list(set([e.strip() for e in entities]))
        return entities
        
    def extract_relationships(self, text, entities):
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
            cleaned_response = response
            if "```json" in cleaned_response:
                cleaned_response = cleaned_response.replace("```json", "").replace("```", "")
            cleaned_response = cleaned_response.strip()

            relationships = json.loads(cleaned_response)
            return relationships
        except Exception as e:
            print(f"Failed to parse relationships JSON: {e}")
            return []
            
    def build_knowledge_graph(self, chunks):
        """Build a knowledge graph in Neo4j"""
        with self.driver.session() as session:
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
                    source = rel["source"]
                    target = rel["target"]
                    relationship = self.normalize(rel["relationship"].upper().replace(" ", "_"))
                    
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

    def get_graph_context(self, query: str, limit=5, label=None):
        """Get relevant context from the knowledge graph based on the query."""
        # Generate embedding for the query
        query_embedding = self.embed_model.embed_query(query)

        with self.driver.session() as session:
            if label:
                entities = session.run("""
                MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)-[:BELONGS_TO]->(cat:Category {name: $label})
                WHERE e.embedding IS NOT NULL
                WITH e, gds.similarity.cosine(e.embedding, $query_embedding) AS score
                WHERE score > 0.75
                RETURN e.name AS entity, score
                ORDER BY score DESC
                LIMIT $limit
                """, query_embedding=query_embedding, label=label, limit=limit).values()
            else:
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

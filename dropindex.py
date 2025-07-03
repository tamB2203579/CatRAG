# Assuming the Neo4j driver is already set up from your previous code
from neo4j import GraphDatabase

neo4j_url = "bolt://localhost:7687"
neo4j_username = "neo4j"
neo4j_password = "123123123"

# Set up the Neo4j driver
driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_password))

def drop_all_indexes():
    try:
        with driver.session() as session:
            # Query all indexes
            result = session.run("SHOW INDEXES YIELD name, type")
            indexes = list(result)  # Collect all index records
            
            if not indexes:
                print("No indexes found in the database.")
                return
            
            # Drop each index
            for record in indexes:
                index_name = record["name"]
                index_type = record["type"]
                try:
                    session.run(f"DROP INDEX {index_name}")
                    print(f"Dropped index: {index_name} (Type: {index_type})")
                except Exception as e:
                    print(f"Failed to drop index {index_name}: {e}")
            
            print("All indexes dropped successfully!")
            
    except Exception as e:
        print(f"Error while accessing Neo4j: {e}")
        raise
    finally:
        driver.close()  # Ensure the driver is closed after use

# Run the function
if __name__ == "__main__":
    drop_all_indexes()
import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
new_db_name=os.environ['DB_NAME']

try:
    # Connect as admin to create new database
    with psycopg2.connect(
        host=os.environ['AURORA_ENDPOINT'],
        user=os.environ['AURORA_USERNAME'],
        password=os.environ['AURORA_PWD'],
        
    ) as admin_conn:
        admin_conn.autocommit = True
        
    with admin_conn.cursor() as cursor:
        # Create database
        cursor.execute(sql.SQL("CREATE DATABASE IF NOT EXISTS {}").format(sql.Identifier(new_db_name)))
            
            
    print(f"Database created successfully!")

except psycopg2.Error as e:
    print(f"Error creating database: {e}")


with admin_conn.cursor() as cursor:
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cursor.execute("SELECT * FROM pg_available_extensions WHERE name = 'vector'")
    result = cursor.fetchone()
    if result:
        print("pgvector extension installed successfully!")
    else:
        print("Failed to install pgvector extension")


with admin_conn.cursor() as cursor:    
    cursor.execute("""
                    CREATE TABLE new_embeddings (
                        id TEXT PRIMARY KEY,
                        embedding VECTOR(768),  
                        metadata TEXT
                    )
                """)

with admin_conn.cursor() as cursor:    
    cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_hnsw_embeddings
                    ON divya_new_embeddings
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (
                        m = 40,
                        ef_construction = 200
                    );
            """)

# Langchain_RAG_variants

Implementation of a simple rag - with query or search to run .

RDS PGVector Extension- By default , pgvector with langchain stores all embeddings in individual collections defined by user. 
1. langchain_pg_collection
2. langchain_pg_embedding

Sample query - SELECT * FROM langchain_pg_embedding where collection_id='082dbe65-f29b-4296-960a-ff1748cb3c15';


Query to check column types -
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'langchain_pg_embedding';

Query to create vector on embedding column -
ALTER TABLE langchain_pg_embedding 
ALTER COLUMN embedding TYPE vector(1536)  -- Use your actual dimension
USING embedding::vector;

Query to create IVF index on column -
CREATE INDEX idx_langchain_embedding_ivflat 
ON langchain_pg_embedding 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100)
WHERE collection_id = '082dbe65-f29b-4296-960a-ff1748cb3c15';


Verifying index - 
-- Check existing indexes
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'langchain_pg_embedding';

HNSW indexing - 
CREATE INDEX idx_langchain_embedding_hnsw 
ON langchain_pg_embedding 
USING hnsw (embedding vector_cosine_ops)
WITH (m= , ef_construction=)
WHERE collection_id='082dbe65-f29b-4296-960a-ff1748cb3c15';





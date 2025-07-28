-- Database Setup for RAG Movie Chatbot with pgvector
-- Run these commands in pgAdmin Query Tool

-- Step 1: Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: Create movies table
CREATE TABLE IF NOT EXISTS movies (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    year INTEGER,
    subtitle_file_path TEXT,
    total_chunks INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Step 3: Create movie_chunks table with vector embeddings
CREATE TABLE IF NOT EXISTS movie_chunks (
    id SERIAL PRIMARY KEY,
    movie_id INTEGER REFERENCES movies(id) ON DELETE CASCADE,
    chunk_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    timestamp_range VARCHAR(50),
    start_time FLOAT,
    end_time FLOAT,
    subtitle_count INTEGER,
    subtitle_numbers INTEGER[],
    embedding VECTOR(384), -- 384 dimensions for all-MiniLM-L6-v2 model
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Step 4: Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_movie_chunks_movie_id ON movie_chunks(movie_id);
CREATE INDEX IF NOT EXISTS idx_movie_chunks_chunk_id ON movie_chunks(chunk_id);

-- Step 5: Create vector similarity search index
-- Note: This index is created after inserting data for better performance
-- CREATE INDEX IF NOT EXISTS idx_movie_chunks_embedding ON movie_chunks USING ivfflat (embedding vector_cosine_ops);

-- Step 6: Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Step 7: Create trigger for movies table
CREATE TRIGGER update_movies_updated_at BEFORE UPDATE ON movies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Step 8: Insert sample movie (optional)
INSERT INTO movies (title, year, subtitle_file_path) 
VALUES ('Little Miss Sunshine', 2006, 'data/Little.Miss.Sunshine.2006.720p.BluRay.x264.YIFY-en.srt')
ON CONFLICT DO NOTHING;

-- Step 9: Verify setup
SELECT 'pgvector extension installed' as status 
WHERE EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector');

SELECT 'Tables created successfully' as status 
WHERE EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'movies')
  AND EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'movie_chunks');

-- Step 10: Show table structure
\d movies
\d movie_chunks 
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import os
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    """Database configuration class"""
    host: str = "localhost"
    port: int = 5432
    database: str = "postgres"
    user: str = "postgres"
    password: str = ""

class PostgreSQLEmbeddingStore:
    """
    PostgreSQL-based embedding store using pgvector for similarity search.
    Replaces FAISS with a production-ready database solution.
    """
    
    def __init__(self, db_config: DatabaseConfig, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the PostgreSQL embedding store.
        
        Args:
            db_config (DatabaseConfig): Database connection configuration
            model_name (str): Name of the sentence transformer model
        """
        self.db_config = db_config
        self.model_name = model_name
        self.model = None
        self.connection = None
        
        print(f"Initializing PostgreSQL Embedding Store with model: {model_name}")
        self._load_model()
        self._connect_database()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            print("Loading sentence transformer model...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded! Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _connect_database(self):
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(
                host=self.db_config.host,
                port=self.db_config.port,
                database=self.db_config.database,
                user=self.db_config.user,
                password=self.db_config.password
            )
            
            # Register pgvector
            register_vector(self.connection)
            print("✅ Connected to PostgreSQL with pgvector support")
            
            # Test the connection
            self._test_connection()
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
    
    def _test_connection(self):
        """Test database connection and pgvector extension."""
        try:
            cur = self.connection.cursor()
            
            # Test pgvector extension
            cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
            if not cur.fetchone():
                raise Exception("pgvector extension not installed")
            
            # Test tables exist
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_name IN ('movies', 'movie_chunks')
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            if 'movies' not in tables or 'movie_chunks' not in tables:
                raise Exception("Required tables not found. Run database_setup.sql first.")
            
            cur.close()
            print("✅ Database and tables verified")
            
        except Exception as e:
            print(f"❌ Database test failed: {e}")
            raise
    
    def store_movie_and_chunks(self, movie_title: str, movie_year: int, 
                              subtitle_path: str, chunks: List[Dict]) -> int:
        """
        Store movie and its chunks with embeddings in the database.
        
        Args:
            movie_title (str): Title of the movie
            movie_year (int): Year of the movie
            subtitle_path (str): Path to subtitle file
            chunks (List[Dict]): List of subtitle chunks from parser
            
        Returns:
            int: Movie ID
        """
        if not chunks:
            raise ValueError("No chunks provided")
        
        print(f"Storing movie '{movie_title}' with {len(chunks)} chunks...")
        
        try:
            cur = self.connection.cursor()
            
            # Step 1: Insert or get movie
            cur.execute("""
                INSERT INTO movies (title, year, subtitle_file_path, total_chunks)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING id
            """, (movie_title, movie_year, subtitle_path, len(chunks)))
            
            result = cur.fetchone()
            if result:
                movie_id = result[0]
                print(f"✅ Created new movie record with ID: {movie_id}")
            else:
                # Movie already exists, get its ID
                cur.execute("SELECT id FROM movies WHERE title = %s AND year = %s", 
                           (movie_title, movie_year))
                movie_id = cur.fetchone()[0]
                print(f"✅ Using existing movie record with ID: {movie_id}")
                
                # Clear existing chunks for this movie
                cur.execute("DELETE FROM movie_chunks WHERE movie_id = %s", (movie_id,))
                print("🗑️ Cleared existing chunks")
            
            # Step 2: Generate embeddings for all chunks
            print("Generating embeddings...")
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Step 3: Prepare data for batch insert
            chunk_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_data.append((
                    movie_id,
                    chunk['chunk_id'],
                    chunk['text'],
                    chunk['timestamp_range'],
                    chunk['start_time'],
                    chunk['end_time'],
                    chunk['subtitle_count'],
                    chunk['subtitle_numbers'],
                    embedding.tolist()  # Convert numpy array to list for pgvector
                ))
            
            # Step 4: Batch insert chunks with embeddings
            print("Inserting chunks into database...")
            execute_values(
                cur,
                """
                INSERT INTO movie_chunks 
                (movie_id, chunk_id, text, timestamp_range, start_time, end_time, 
                 subtitle_count, subtitle_numbers, embedding)
                VALUES %s
                """,
                chunk_data,
                template=None,
                page_size=100
            )
            
            # Step 5: Create vector index if it doesn't exist (for better search performance)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_movie_chunks_embedding 
                ON movie_chunks USING ivfflat (embedding vector_cosine_ops)
            """)
            
            # Commit transaction
            self.connection.commit()
            cur.close()
            
            print(f"✅ Successfully stored {len(chunks)} chunks for movie ID {movie_id}")
            return movie_id
            
        except Exception as e:
            self.connection.rollback()
            print(f"❌ Error storing movie and chunks: {e}")
            raise
    
    def similarity_search(self, query: str, movie_id: Optional[int] = None, 
                         k: int = 5) -> List[Dict]:
        """
        Search for similar chunks using pgvector similarity.
        
        Args:
            query (str): Search query
            movie_id (Optional[int]): Limit search to specific movie
            k (int): Number of results to return
            
        Returns:
            List[Dict]: Similar chunks with similarity scores
        """
        print(f"Searching for: '{query}' (top {k} results)")
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
            
            cur = self.connection.cursor()
            
            # Build query with optional movie filter
            if movie_id:
                sql = """
                    SELECT mc.id, mc.chunk_id, mc.text, mc.timestamp_range,
                           mc.start_time, mc.end_time, mc.subtitle_count,
                           mc.subtitle_numbers, m.title, m.year,
                           (mc.embedding <=> %s) as distance
                    FROM movie_chunks mc
                    JOIN movies m ON mc.movie_id = m.id
                    WHERE mc.movie_id = %s
                    ORDER BY mc.embedding <=> %s
                    LIMIT %s
                """
                cur.execute(sql, (query_embedding.tolist(), movie_id, query_embedding.tolist(), k))
            else:
                sql = """
                    SELECT mc.id, mc.chunk_id, mc.text, mc.timestamp_range,
                           mc.start_time, mc.end_time, mc.subtitle_count,
                           mc.subtitle_numbers, m.title, m.year,
                           (mc.embedding <=> %s) as distance
                    FROM movie_chunks mc
                    JOIN movies m ON mc.movie_id = m.id
                    ORDER BY mc.embedding <=> %s
                    LIMIT %s
                """
                cur.execute(sql, (query_embedding.tolist(), query_embedding.tolist(), k))
            
            results = []
            for row in cur.fetchall():
                result = {
                    'id': row[0],
                    'chunk_id': row[1],
                    'text': row[2],
                    'timestamp_range': row[3],
                    'start_time': row[4],
                    'end_time': row[5],
                    'subtitle_count': row[6],
                    'subtitle_numbers': row[7],
                    'movie_title': row[8],
                    'movie_year': row[9],
                    'distance': float(row[10]),
                    'similarity_score': 1 - float(row[10])  # Convert distance to similarity
                }
                results.append(result)
            
            cur.close()
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            raise
    
    def get_movie_stats(self, movie_id: int) -> Dict:
        """Get statistics for a specific movie."""
        try:
            cur = self.connection.cursor()
            
            cur.execute("""
                SELECT m.title, m.year, m.total_chunks, COUNT(mc.id) as stored_chunks,
                       m.created_at, m.updated_at
                FROM movies m
                LEFT JOIN movie_chunks mc ON m.id = mc.movie_id
                WHERE m.id = %s
                GROUP BY m.id, m.title, m.year, m.total_chunks, m.created_at, m.updated_at
            """, (movie_id,))
            
            result = cur.fetchone()
            cur.close()
            
            if result:
                return {
                    'movie_id': movie_id,
                    'title': result[0],
                    'year': result[1],
                    'total_chunks': result[2],
                    'stored_chunks': result[3],
                    'created_at': result[4],
                    'updated_at': result[5]
                }
            else:
                return {'error': 'Movie not found'}
                
        except Exception as e:
            print(f"❌ Error getting movie stats: {e}")
            return {'error': str(e)}
    
    def list_movies(self) -> List[Dict]:
        """List all movies in the database."""
        try:
            cur = self.connection.cursor()
            
            cur.execute("""
                SELECT m.id, m.title, m.year, m.total_chunks, COUNT(mc.id) as stored_chunks
                FROM movies m
                LEFT JOIN movie_chunks mc ON m.id = mc.movie_id
                GROUP BY m.id, m.title, m.year, m.total_chunks
                ORDER BY m.created_at DESC
            """)
            
            movies = []
            for row in cur.fetchall():
                movies.append({
                    'id': row[0],
                    'title': row[1],
                    'year': row[2],
                    'total_chunks': row[3],
                    'stored_chunks': row[4]
                })
            
            cur.close()
            return movies
            
        except Exception as e:
            print(f"❌ Error listing movies: {e}")
            return []
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            print("🔌 Database connection closed")


# Test function
def test_postgres_embedder():
    """Test the PostgreSQL embedding system."""
    from parse_subtitles import SubtitleParser
    
    print("=== Testing PostgreSQL Embedding System ===\n")
    
    # Database configuration - UPDATE THESE VALUES
    db_config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="postgres",  # Change to your database name
        user="postgres",      # Change to your username
        password=""           # Add your password
    )
    
    try:
        # Step 1: Parse subtitles
        parser = SubtitleParser(chunk_size=5, overlap=1)
        subtitles, chunks = parser.process_subtitle_file(
            'data/Little.Miss.Sunshine.2006.720p.BluRay.x264.YIFY-en.srt'
        )
        
        if not chunks:
            print("❌ No chunks found. Check subtitle file.")
            return
        
        # Step 2: Initialize PostgreSQL store
        store = PostgreSQLEmbeddingStore(db_config)
        
        # Step 3: Store movie and chunks
        movie_id = store.store_movie_and_chunks(
            movie_title="Little Miss Sunshine",
            movie_year=2006,
            subtitle_path="data/Little.Miss.Sunshine.2006.720p.BluRay.x264.YIFY-en.srt",
            chunks=chunks
        )
        
        # Step 4: Test search
        test_queries = [
            "What does the family say about beauty pageants?",
            "Who talks about winning?",
            "What happens with the van?",
            "What does Olive say?"
        ]
        
        print(f"\n=== Testing Search (Movie ID: {movie_id}) ===")
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = store.similarity_search(query, movie_id=movie_id, k=3)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['similarity_score']:.3f}")
                print(f"     Time: {result['timestamp_range']}")
                print(f"     Text: {result['text'][:150]}...")
        
        # Step 5: Show stats
        stats = store.get_movie_stats(movie_id)
        print(f"\n=== Movie Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Step 6: List all movies
        movies = store.list_movies()
        print(f"\n=== All Movies ===")
        for movie in movies:
            print(f"ID: {movie['id']}, Title: {movie['title']} ({movie['year']}), "
                  f"Chunks: {movie['stored_chunks']}/{movie['total_chunks']}")
        
        store.close()
        return store
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    test_postgres_embedder() 
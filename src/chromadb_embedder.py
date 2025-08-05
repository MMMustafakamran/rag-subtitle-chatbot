import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import os
import json

class ChromaDBEmbeddingStore:
    """
    ChromaDB-based embedding store for RAG applications.
    """
    
    def __init__(self, 
                 persist_directory: str = "data/chromadb", 
                 model_name: str = "all-MiniLM-L6-v2",
                 collection_name: str = "movie_subtitles"):
       
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.collection_name = collection_name
        self.model = None
        self.client = None
        self.collection = None
        
        print(f" Initializing ChromaDB Embedding Store")
        print(f" Data directory: {persist_directory}")
        print(f" Model: {model_name}")
        
        self._setup_chromadb()
        self._load_model()
    
    def _setup_chromadb(self):
        """Set up ChromaDB client and collection."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Movie subtitle chunks for RAG"}
            )
            
            print(f" ChromaDB initialized successfully!")
            print(f"Collection '{self.collection_name}' ready")
            
        except Exception as e:
            print(f" ChromaDB setup failed: {e}")
            raise
    
    def _load_model(self):
        """Load sentence transformer model."""
        try:
            print(" Loading sentence transformer model...")
            self.model = SentenceTransformer(self.model_name)
            print(f" Model loaded! Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f" Model loading failed: {e}")
            raise
    
    def store_movie_chunks(self, movie_title: str, movie_year: int, chunks: List[Dict]) -> str:
        """
        Store movie chunks with embeddings in ChromaDB.
        """
        if not chunks:
            raise ValueError("No chunks provided")
        
        movie_id = f"{movie_title}_{movie_year}".replace(" ", "_").lower()
        print(f"🎬 Storing '{movie_title}' ({movie_year}) with {len(chunks)} chunks...")
        
        try:
            # Clear existing chunks for this movie
            existing_ids = self._get_movie_chunk_ids(movie_id)
            if existing_ids:
                print(f" Removing {len(existing_ids)} existing chunks...")
                self.collection.delete(ids=existing_ids)
            
            # Prepare data for ChromaDB
            texts = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                chunk_id = f"{movie_id}_chunk_{chunk['chunk_id']}"
                
                texts.append(chunk['text'])
                ids.append(chunk_id)
                
                # Store all chunk metadata
                metadata = {
                    'movie_id': movie_id,
                    'movie_title': movie_title,
                    'movie_year': movie_year,
                    'chunk_id': chunk['chunk_id'],
                    'timestamp_range': chunk['timestamp_range'],
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'subtitle_count': chunk['subtitle_count'],
                    'subtitle_numbers': json.dumps(chunk['subtitle_numbers'])  # Store as JSON string
                }
                metadatas.append(metadata)
            
            # Generate embeddings (ChromaDB can do this automatically, but we use our model for consistency)
            print(" Generating embeddings...")
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Add to ChromaDB
            print("💾 Storing in ChromaDB...")
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )
            
            print(f" Successfully stored {len(chunks)} chunks for '{movie_title}'")
            return movie_id
            
        except Exception as e:
            print(f" Error storing chunks: {e}")
            raise
    
    def _get_movie_chunk_ids(self, movie_id: str) -> List[str]:
        """Get all chunk IDs for a specific movie."""
        try:
            results = self.collection.get(
                where={"movie_id": movie_id},
                include=[]  # Only get IDs
            )
            return results['ids']
        except Exception:
            return []
    
    def similarity_search(self, query: str, movie_id: Optional[str] = None, k: int = 5) -> List[Dict]:
        """
        Search for similar chunks using ChromaDB.
        
        Args:
            query (str): Search query
            movie_id (Optional[str]): Limit search to specific movie
            k (int): Number of results to return
            
        Returns:
            List[Dict]: Similar chunks with similarity scores
        """
        print(f"🔍 Searching for: '{query}' (top {k} results)")
        
        try:
            # Build where clause for movie filtering
            where_clause = {"movie_id": movie_id} if movie_id else None
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            search_results = []
            if results['documents'] and results['documents'][0]:  # Check if we have results
                for i in range(len(results['documents'][0])):
                    metadata = results['metadatas'][0][i]
                    
                    # Parse subtitle_numbers back from JSON
                    subtitle_numbers = json.loads(metadata['subtitle_numbers'])
                    
                    result = {
                        'text': results['documents'][0][i],
                        'movie_title': metadata['movie_title'],
                        'movie_year': metadata['movie_year'],
                        'chunk_id': metadata['chunk_id'],
                        'timestamp_range': metadata['timestamp_range'],
                        'start_time': metadata['start_time'],
                        'end_time': metadata['end_time'],
                        'subtitle_count': metadata['subtitle_count'],
                        'subtitle_numbers': subtitle_numbers,
                        'distance': results['distances'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'rank': i + 1
                    }
                    search_results.append(result)
            
            print(f"✅ Found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            print(f" Search error: {e}")
            raise
    
    def get_movie_stats(self, movie_id: str) -> Dict:
        """Get statistics for a specific movie."""
        try:
            results = self.collection.get(
                where={"movie_id": movie_id},
                include=['metadatas']
            )
            
            if not results['metadatas']:
                return {'error': 'Movie not found'}
            
            # Get movie info from first chunk
            first_metadata = results['metadatas'][0]
            
            return {
                'movie_id': movie_id,
                'movie_title': first_metadata['movie_title'],
                'movie_year': first_metadata['movie_year'],
                'total_chunks': len(results['metadatas']),
                'stored_chunks': len(results['metadatas'])  # Same as total for ChromaDB
            }
            
        except Exception as e:
            print(f"❌ Error getting movie stats: {e}")
            return {'error': str(e)}
    
    def list_movies(self) -> List[Dict]:
        """List all movies in the database."""
        try:
            # Get all data
            results = self.collection.get(include=['metadatas'])
            
            if not results['metadatas']:
                return []
            
            # Group by movie
            movies = {}
            for metadata in results['metadatas']:
                movie_id = metadata['movie_id']
                if movie_id not in movies:
                    movies[movie_id] = {
                        'id': movie_id,
                        'title': metadata['movie_title'],
                        'year': metadata['movie_year'],
                        'chunk_count': 0
                    }
                movies[movie_id]['chunk_count'] += 1
            
            return list(movies.values())
            
        except Exception as e:
            print(f"❌ Error listing movies: {e}")
            return []
    
    def delete_movie(self, movie_id: str) -> bool:
        """Delete all chunks for a specific movie."""
        try:
            chunk_ids = self._get_movie_chunk_ids(movie_id)
            if chunk_ids:
                self.collection.delete(ids=chunk_ids)
                print(f"🗑️ Deleted {len(chunk_ids)} chunks for movie: {movie_id}")
                return True
            else:
                print(f"⚠️ No chunks found for movie: {movie_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error deleting movie: {e}")
            return False
    
    def get_collection_info(self) -> Dict:
        """Get information about the ChromaDB collection."""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'total_chunks': count,
                'persist_directory': self.persist_directory,
                'model_name': self.model_name
            }
        except Exception as e:
            return {'error': str(e)}


# Test function
def test_chromadb_system():
    """Test the ChromaDB embedding system."""
    from parse_subtitles import SubtitleParser
    
    print("=== Testing ChromaDB Embedding System ===\n")
    
    try:
        # Step 1: Parse subtitles
        parser = SubtitleParser(chunk_size=5, overlap=1)
        subtitles, chunks = parser.process_subtitle_file(
            'data/Little.Miss.Sunshine.2006.720p.BluRay.x264.YIFY-en.srt'
        )
        
        if not chunks:
            print("❌ No chunks found. Check subtitle file.")
            return None
        
        print(f"📝 Parsed {len(chunks)} chunks from subtitle file")
        
        # Step 2: Initialize ChromaDB store
        store = ChromaDBEmbeddingStore()
        
        # Step 3: Store movie and chunks
        movie_id = store.store_movie_chunks(
            movie_title="Little Miss Sunshine",
            movie_year=2006,
            chunks=chunks
        )
        
        # Step 4: Test search
        test_queries = [
            "What does the family say about beauty pageants?",
            "Who talks about winning?",
            "What happens with the van?",
            "What does Olive say about dancing?"
        ]
        
        print(f"\n=== Testing Search ===")
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            results = store.similarity_search(query, movie_id=movie_id, k=3)
            
            for result in results:
                print(f"  📍 Score: {result['similarity_score']:.3f}")
                print(f"     Time: {result['timestamp_range']}")
                print(f"     Text: {result['text'][:150]}...")
                print()
        
        # Step 5: Show stats
        print("=== Collection Statistics ===")
        collection_info = store.get_collection_info()
        for key, value in collection_info.items():
            print(f"{key}: {value}")
        
        movie_stats = store.get_movie_stats(movie_id)
        print(f"\n=== Movie Statistics ===")
        for key, value in movie_stats.items():
            print(f"{key}: {value}")
        
        # Step 6: List all movies
        movies = store.list_movies()
        print(f"\n=== All Movies ===")
        for movie in movies:
            print(f"• {movie['title']} ({movie['year']}) - {movie['chunk_count']} chunks")
        
        print(f"\n🎉 ChromaDB system test completed successfully!")
        print(f"📁 Data saved to: {store.persist_directory}")
        
        return store
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    test_chromadb_system()
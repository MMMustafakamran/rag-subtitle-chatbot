import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss

class EmbeddingIndexer:
    """
    A class to handle embedding generation and vector indexing for subtitle chunks.
    Uses Sentence Transformers for embeddings and FAISS for efficient similarity search.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_type: str = "flat"):
        """
        Initialize the embedding indexer.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
            index_type (str): Type of FAISS index ('flat' or 'ivf')
        """
        self.model_name = model_name
        self.index_type = index_type
        self.model = None
        self.index = None
        self.chunks = []
        self.embeddings = None
        
        print(f"Initializing EmbeddingIndexer with model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            print("Loading sentence transformer model...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully! Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for subtitle chunks.
        
        Args:
            chunks (List[Dict]): List of subtitle chunks from parser
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if not chunks:
            raise ValueError("No chunks provided for embedding")
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better similarity search
        )
        
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def create_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create a FAISS index from embeddings.
        
        Args:
            embeddings (np.ndarray): Array of embeddings
            
        Returns:
            faiss.Index: FAISS index for similarity search
        """
        dimension = embeddings.shape[1]
        n_embeddings = embeddings.shape[0]
        
        print(f"Creating FAISS index for {n_embeddings} embeddings with dimension {dimension}")
        
        if self.index_type == "flat":
            # Simple flat index - good for small datasets
            index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity with normalized vectors)
        elif self.index_type == "ivf":
            # IVF index - better for larger datasets
            nlist = min(100, n_embeddings // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Train the index
            print("Training IVF index...")
            index.train(embeddings.astype(np.float32))
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Add embeddings to index
        index.add(embeddings.astype(np.float32))
        
        print(f"Index created successfully with {index.ntotal} vectors")
        return index
    
    def build_index(self, chunks: List[Dict]) -> Tuple[faiss.Index, np.ndarray]:
        """
        Complete pipeline: embed chunks and build index.
        
        Args:
            chunks (List[Dict]): List of subtitle chunks
            
        Returns:
            Tuple[faiss.Index, np.ndarray]: (FAISS index, embeddings array)
        """
        # Store chunks for later retrieval
        self.chunks = chunks
        
        # Generate embeddings
        embeddings = self.embed_chunks(chunks)
        self.embeddings = embeddings
        
        # Create index
        index = self.create_index(embeddings)
        self.index = index
        
        return index, embeddings
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar chunks given a query.
        
        Args:
            query (str): Search query
            k (int): Number of top results to return
            
        Returns:
            List[Dict]: List of similar chunks with similarity scores
        """
        if self.index is None or not self.chunks:
            raise ValueError("Index not built yet. Call build_index() first.")
        
        print(f"Searching for: '{query}' (top {k} results)")
        
        # Embed the query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search in the index
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):  # Valid index
                result = self.chunks[idx].copy()
                result['similarity_score'] = float(score)
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def save_index(self, save_dir: str = "data/index"):
        """
        Save the index, embeddings, and chunks to disk.
        
        Args:
            save_dir (str): Directory to save the index files
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(save_dir, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        # Save chunks and embeddings
        data_path = os.path.join(save_dir, "index_data.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'model_name': self.model_name,
                'index_type': self.index_type
            }, f)
        
        print(f"Index saved to {save_dir}")
    
    def load_index(self, save_dir: str = "data/index"):
        """
        Load a previously saved index.
        
        Args:
            save_dir (str): Directory containing the saved index files
        """
        index_path = os.path.join(save_dir, "faiss_index.bin")
        data_path = os.path.join(save_dir, "index_data.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(data_path):
            raise FileNotFoundError(f"Index files not found in {save_dir}")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.embeddings = data['embeddings']
            
            # Verify model compatibility
            if data['model_name'] != self.model_name:
                print(f"Warning: Loaded index was created with {data['model_name']}, "
                      f"but current model is {self.model_name}")
        
        print(f"Index loaded from {save_dir} with {len(self.chunks)} chunks")
    
    def get_stats(self) -> Dict:
        """Get statistics about the current index."""
        if not self.chunks or self.embeddings is None:
            return {"status": "No index built"}
        
        return {
            "num_chunks": len(self.chunks),
            "embedding_dimension": self.embeddings.shape[1],
            "model_name": self.model_name,
            "index_type": self.index_type,
            "index_size": self.index.ntotal if self.index else 0
        }


# Utility functions for testing
def test_embedding_system():
    """Test the embedding and indexing system."""
    # Import the subtitle parser
    from parse_subtitles import SubtitleParser
    
    print("=== Testing Embedding & Indexing System ===\n")
    
    # Step 1: Parse subtitles (you need a subtitle file in data/ folder)
    parser = SubtitleParser(chunk_size=3, overlap=1)
    
    try:
        subtitles, chunks = parser.process_subtitle_file('data/sample.srt')
        
        if not chunks:
            print("No chunks found. Please add a subtitle file to data/ folder.")
            return
        
        # Step 2: Create embeddings and index
        indexer = EmbeddingIndexer(model_name="all-MiniLM-L6-v2")
        
        # Build the index
        index, embeddings = indexer.build_index(chunks)
        
        # Step 3: Test search
        test_queries = [
            "What did the character say about love?",
            "Who is the main character?",
            "What happened in the beginning?",
            "How does the movie end?"
        ]
        
        print("\n=== Testing Search Functionality ===")
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = indexer.search(query, k=3)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['similarity_score']:.3f}")
                print(f"     Time: {result['timestamp_range']}")
                print(f"     Text: {result['text'][:100]}...")
        
        # Step 4: Save index
        indexer.save_index()
        print(f"\n=== Index Statistics ===")
        stats = indexer.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        return indexer
        
    except FileNotFoundError:
        print("Please add a subtitle file (.srt) to the 'data/' folder named 'sample.srt'")
        return None


if __name__ == "__main__":
    # Run the test
    test_embedding_system()

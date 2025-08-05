#!/usr/bin/env python3
"""
RAG Movie Chatbot - Development Main File
========================================

This file runs the complete RAG pipeline with detailed debugging statements.
Perfect for understanding how all components work together.

Usage: python main_dev.py
"""

import os
import time
import traceback
from typing import List, Dict, Optional
from datetime import datetime

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print a colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_step(step_num: int, text: str):
    """Print a step with numbering"""
    print(f"{Colors.OKBLUE}{Colors.BOLD}STEP {step_num}: {text}{Colors.ENDC}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def print_debug(text: str):
    """Print debug message"""
    print(f"{Colors.ENDC}🔍 DEBUG: {text}{Colors.ENDC}")

class RAGMovieChatbotDev:
    """Development version of RAG Movie Chatbot with extensive debugging"""
    
    def __init__(self):
        self.parser = None
        self.embedder = None
        self.subtitle_file = None
        self.chunks = []
        self.movie_id = None
        
        print_header("RAG MOVIE CHATBOT - DEVELOPMENT MODE")
        print_info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_info(f"Working directory: {os.getcwd()}")
        
    def step1_check_environment(self):
        """Step 1: Check environment and dependencies"""
        print_step(1, "ENVIRONMENT CHECK")
        
        try:
            # Check Python version
            import sys
            print_debug(f"Python version: {sys.version}")
            
            # Check required directories
            required_dirs = ['src', 'data']
            for dir_name in required_dirs:
                if os.path.exists(dir_name):
                    print_success(f"Directory '{dir_name}' exists")
                else:
                    print_error(f"Directory '{dir_name}' missing")
                    os.makedirs(dir_name, exist_ok=True)
                    print_info(f"Created directory '{dir_name}'")
            
            # Check for subtitle files
            data_files = os.listdir('data') if os.path.exists('data') else []
            srt_files = [f for f in data_files if f.endswith('.srt')]
            
            if srt_files:
                print_success(f"Found {len(srt_files)} subtitle files:")
                for srt_file in srt_files:
                    file_size = os.path.getsize(f"data/{srt_file}") / 1024  # KB
                    print_debug(f"  - {srt_file} ({file_size:.1f} KB)")
                self.subtitle_file = f"data/{srt_files[0]}"  # Use first file
            else:
                print_warning("No .srt files found in data/ directory")
                print_info("Please add a subtitle file to continue")
                return False
            
            # Check imports
            print_debug("Checking imports...")
            try:
                import numpy
                print_success(f"numpy: {numpy.__version__}")
            except ImportError:
                print_error("numpy not installed")
                return False
                
            try:
                import chromadb
                print_success(f"chromadb: {chromadb.__version__}")
            except ImportError:
                print_error("chromadb not installed")
                return False
                
            try:
                import sentence_transformers
                print_success(f"sentence-transformers: {sentence_transformers.__version__}")
            except ImportError:
                print_error("sentence-transformers not installed")
                return False
            
            return True
            
        except Exception as e:
            print_error(f"Environment check failed: {e}")
            traceback.print_exc()
            return False
    
    def step2_initialize_parser(self):
        """Step 2: Initialize subtitle parser"""
        print_step(2, "SUBTITLE PARSER INITIALIZATION")
        
        try:
            from src.parse_subtitles import SubtitleParser
            
            # Initialize parser with debug info
            chunk_size = 5
            overlap = 1
            print_debug(f"Initializing parser with chunk_size={chunk_size}, overlap={overlap}")
            
            self.parser = SubtitleParser(chunk_size=chunk_size, overlap=overlap)
            print_success("SubtitleParser initialized successfully")
            
            # Show parser configuration
            print_debug(f"Parser config: chunk_size={self.parser.chunk_size}, overlap={self.parser.overlap}")
            
            return True
            
        except Exception as e:
            print_error(f"Failed to initialize parser: {e}")
            traceback.print_exc()
            return False
    
    def step3_parse_subtitles(self):
        """Step 3: Parse subtitle file"""
        print_step(3, "SUBTITLE PARSING")
        
        if not self.subtitle_file:
            print_error("No subtitle file available")
            return False
        
        try:
            print_debug(f"Processing file: {self.subtitle_file}")
            
            # Time the parsing process
            start_time = time.time()
            subtitles, chunks = self.parser.process_subtitle_file(self.subtitle_file)
            parse_time = time.time() - start_time
            
            print_success(f"Parsing completed in {parse_time:.2f} seconds")
            print_info(f"Raw subtitles: {len(subtitles)} entries")
            print_info(f"Text chunks: {len(chunks)} chunks")
            
            # Store chunks for later use
            self.chunks = chunks
            
            # Show sample chunk
            if chunks:
                sample_chunk = chunks[0]
                print_debug("Sample chunk structure:")
                for key, value in sample_chunk.items():
                    if key == 'text':
                        preview = value[:100] + "..." if len(value) > 100 else value
                        print_debug(f"  {key}: {preview}")
                    else:
                        print_debug(f"  {key}: {value}")
            
            # Calculate statistics
            total_text_length = sum(len(chunk['text']) for chunk in chunks)
            avg_chunk_length = total_text_length / len(chunks) if chunks else 0
            print_debug(f"Average chunk length: {avg_chunk_length:.1f} characters")
            
            return True
            
        except Exception as e:
            print_error(f"Failed to parse subtitles: {e}")
            traceback.print_exc()
            return False
    
    def step4_initialize_embedder(self):
        """Step 4: Initialize ChromaDB embedder"""
        print_step(4, "CHROMADB EMBEDDER INITIALIZATION")
        
        try:
            from src.chromadb_embedder import ChromaDBEmbeddingStore
            
            print_debug("Initializing ChromaDB embedder...")
            print_warning("This may take time on first run (downloading model ~90MB)")
            
            start_time = time.time()
            self.embedder = ChromaDBEmbeddingStore(
                persist_directory="data/chromadb",
                model_name="all-MiniLM-L6-v2",
                collection_name="movie_subtitles"
            )
            init_time = time.time() - start_time
            
            print_success(f"ChromaDB embedder initialized in {init_time:.2f} seconds")
            
            # Show embedder info
            collection_info = self.embedder.get_collection_info()
            print_debug("Embedder configuration:")
            for key, value in collection_info.items():
                print_debug(f"  {key}: {value}")
            
            return True
            
        except Exception as e:
            print_error(f"Failed to initialize embedder: {e}")
            traceback.print_exc()
            return False
    
    def step5_store_embeddings(self):
        """Step 5: Store movie chunks as embeddings"""
        print_step(5, "EMBEDDING STORAGE")
        
        if not self.chunks:
            print_error("No chunks available for embedding")
            return False
        
        try:
            movie_title = "Little Miss Sunshine"  # Default, can be made configurable
            movie_year = 2006
            
            print_debug(f"Storing movie: {movie_title} ({movie_year})")
            print_debug(f"Number of chunks to embed: {len(self.chunks)}")
            
            start_time = time.time()
            self.movie_id = self.embedder.store_movie_chunks(
                movie_title=movie_title,
                movie_year=movie_year,
                chunks=self.chunks
            )
            embedding_time = time.time() - start_time
            
            print_success(f"Embeddings stored in {embedding_time:.2f} seconds")
            print_info(f"Movie ID: {self.movie_id}")
            
            # Show storage statistics
            movie_stats = self.embedder.get_movie_stats(self.movie_id)
            print_debug("Storage statistics:")
            for key, value in movie_stats.items():
                print_debug(f"  {key}: {value}")
            
            return True
            
        except Exception as e:
            print_error(f"Failed to store embeddings: {e}")
            traceback.print_exc()
            return False
    
    def step6_test_search(self):
        """Step 6: Test similarity search"""
        print_step(6, "SIMILARITY SEARCH TESTING")
        
        if not self.movie_id:
            print_error("No movie available for search")
            return False
        
        # Test queries
        test_queries = [
            "What does the family say about beauty pageants?",
            "Who talks about winning?",
            "What happens with the van?",
            "What does Olive say about dancing?",
            "How does the movie end?"
        ]
        
        try:
            print_debug(f"Testing {len(test_queries)} queries...")
            
            for i, query in enumerate(test_queries, 1):
                print_info(f"\nQuery {i}: '{query}'")
                
                start_time = time.time()
                results = self.embedder.similarity_search(
                    query=query,
                    movie_id=self.movie_id,
                    k=3
                )
                search_time = time.time() - start_time
                
                print_debug(f"Search completed in {search_time:.3f} seconds")
                print_debug(f"Found {len(results)} results")
                
                # Show top result
                if results:
                    top_result = results[0]
                    print_success(f"Top match (score: {top_result['similarity_score']:.3f}):")
                    print_debug(f"  Time: {top_result['timestamp_range']}")
                    preview = top_result['text'][:150] + "..." if len(top_result['text']) > 150 else top_result['text']
                    print_debug(f"  Text: {preview}")
                else:
                    print_warning("No results found")
            
            return True
            
        except Exception as e:
            print_error(f"Search testing failed: {e}")
            traceback.print_exc()
            return False
    
    def step7_interactive_mode(self):
        """Step 7: Interactive query mode"""
        print_step(7, "INTERACTIVE MODE")
        
        if not self.movie_id:
            print_error("No movie available for interactive mode")
            return False
        
        print_info("Enter your questions about the movie (type 'quit' to exit)")
        print_info("Example: 'What does the main character say about success?'")
        
        try:
            while True:
                print(f"\n{Colors.OKCYAN}🎬 Ask about the movie:{Colors.ENDC} ", end="")
                user_query = input().strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print_info("Exiting interactive mode...")
                    break
                
                if not user_query:
                    print_warning("Please enter a question")
                    continue
                
                print_debug(f"Processing query: '{user_query}'")
                
                start_time = time.time()
                results = self.embedder.similarity_search(
                    query=user_query,
                    movie_id=self.movie_id,
                    k=5
                )
                search_time = time.time() - start_time
                
                print_debug(f"Search completed in {search_time:.3f} seconds")
                
                if results:
                    print_success(f"Found {len(results)} relevant scenes:")
                    
                    for i, result in enumerate(results, 1):
                        score = result['similarity_score']
                        timestamp = result['timestamp_range']
                        text = result['text']
                        
                        print(f"\n{Colors.OKGREEN}📍 Result {i} (Score: {score:.3f}){Colors.ENDC}")
                        print(f"{Colors.OKCYAN}⏰ {timestamp}{Colors.ENDC}")
                        print(f"💬 {text}")
                else:
                    print_warning("No relevant scenes found")
            
            return True
            
        except KeyboardInterrupt:
            print_info("\nInteractive mode interrupted by user")
            return True
        except Exception as e:
            print_error(f"Interactive mode failed: {e}")
            traceback.print_exc()
            return False
    
    def run_complete_pipeline(self):
        """Run the complete RAG pipeline"""
        print_info("Starting complete RAG pipeline...")
        
        steps = [
            self.step1_check_environment,
            self.step2_initialize_parser,
            self.step3_parse_subtitles,
            self.step4_initialize_embedder,
            self.step5_store_embeddings,
            self.step6_test_search,
            self.step7_interactive_mode
        ]
        
        for step_func in steps:
            try:
                if not step_func():
                    print_error(f"Pipeline stopped at {step_func.__name__}")
                    return False
                print_success(f"Completed {step_func.__name__}")
            except KeyboardInterrupt:
                print_warning("\nPipeline interrupted by user")
                return False
            except Exception as e:
                print_error(f"Unexpected error in {step_func.__name__}: {e}")
                traceback.print_exc()
                return False
        
        print_header("PIPELINE COMPLETED SUCCESSFULLY")
        print_success("All components working correctly!")
        print_info("You can now run the Streamlit app: python -m streamlit run app_simple.py")
        
        return True

def main():
    """Main function"""
    try:
        chatbot = RAGMovieChatbotDev()
        chatbot.run_complete_pipeline()
    except KeyboardInterrupt:
        print_warning("\nProgram interrupted by user")
    except Exception as e:
        print_error(f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        print_info(f"Program ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
import streamlit as st
import os
from typing import List, Dict
import time
##python -m streamlit run app.py
# Page configuration
st.set_page_config(
    page_title="🎬 Movie Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-chunk {
        background-color: #e8f4f8;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 3px solid #1f77b4;
    }
    .timestamp {
        color: #666;
        font-size: 0.9rem;
        font-style: italic;
    }
    .similarity-score {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_chromadb():
    """Initialize ChromaDB store with error handling"""
    try:
        with st.spinner("🔄 Initializing ChromaDB (this may take a moment on first run)..."):
            from src.chromadb_embedder import ChromaDBEmbeddingStore
            store = ChromaDBEmbeddingStore()
            st.success("✅ ChromaDB initialized successfully!")
            return store
    except Exception as e:
        st.error(f"❌ Failed to initialize ChromaDB: {e}")
        st.info("💡 This might be the first run. The sentence transformer model needs to download (~90MB)")
        return None

def main():
    """Main application function"""
    
    # Main header
    st.markdown('<h1 class="main-header">🎬 Movie Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### Ask questions about your favorite movies using their subtitle files!")
    
    # Initialize ChromaDB
    if 'chromadb_store' not in st.session_state:
        st.session_state.chromadb_store = None
    
    # Sidebar
    st.sidebar.markdown("## 🎬 Movie Chatbot")
    st.sidebar.markdown("---")
    
    # Initialize button
    if st.sidebar.button("🚀 Initialize ChromaDB", type="primary"):
        st.session_state.chromadb_store = initialize_chromadb()
    
    # Show initialization status
    if st.session_state.chromadb_store is None:
        st.warning("⚠️ ChromaDB not initialized yet. Click the button in the sidebar to start!")
        st.info("""
        **First time setup:**
        1. Click "🚀 Initialize ChromaDB" in the sidebar
        2. Wait for the sentence transformer model to download (~90MB)
        3. Once initialized, you can upload movies and start chatting!
        """)
        return
    
    # Main content tabs
    tab1, tab2 = st.tabs(["💬 Chat with Movies", "📁 Upload New Movie"])
    
    with tab1:
        movie_chat_interface()
    
    with tab2:
        upload_and_process_subtitle()

def movie_chat_interface():
    """Main chat interface for querying movies"""
    st.subheader("💬 Chat with Your Movies")
    
    if st.session_state.chromadb_store is None:
        st.warning("Please initialize ChromaDB first!")
        return
    
    try:
        # Get available movies
        movies = st.session_state.chromadb_store.list_movies()
        
        if not movies:
            st.warning("📽️ No movies available. Please upload a subtitle file first!")
            return
        
        # Movie selection
        movie_options = {f"{movie['title']} ({movie['year']})": movie['id'] for movie in movies}
        selected_movie_display = st.selectbox(
            "🎬 Select a movie to chat about:",
            options=list(movie_options.keys()),
            key="selected_movie"
        )
        
        selected_movie_id = movie_options[selected_movie_display] if selected_movie_display else None
        
        # Show movie stats
        if selected_movie_id:
            movie_stats = st.session_state.chromadb_store.get_movie_stats(selected_movie_id)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Chunks", movie_stats.get('total_chunks', 0))
            with col2:
                st.metric("🎭 Movie Year", movie_stats.get('movie_year', 'N/A'))
            with col3:
                st.metric("💾 Stored Chunks", movie_stats.get('stored_chunks', 0))
        
        # Chat interface
        st.markdown("---")
        
        # Query input
        query = st.text_input(
            "🤔 Ask a question about the movie:",
            placeholder="e.g., What does the main character say about love?",
            key="user_query"
        )
        
        # Search parameters
        with st.expander("🔧 Search Settings"):
            col1, col2 = st.columns(2)
            with col1:
                num_results = st.slider("Number of results", min_value=1, max_value=10, value=5)
            with col2:
                min_score = st.slider("Minimum similarity score", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        
        # Process query
        if query and selected_movie_id:
            with st.spinner("🔍 Searching movie dialogue..."):
                try:
                    # Search for relevant chunks
                    results = st.session_state.chromadb_store.similarity_search(
                        query, 
                        movie_id=selected_movie_id, 
                        k=num_results
                    )
                    
                    # Filter by minimum score
                    filtered_results = [r for r in results if r['similarity_score'] >= min_score]
                    
                    if filtered_results:
                        st.success(f"✅ Found {len(filtered_results)} relevant dialogue chunks!")
                        
                        # Display results
                        for i, result in enumerate(filtered_results, 1):
                            with st.container():
                                st.markdown(f"""
                                <div class="source-chunk">
                                    <h4>📍 Result #{i} 
                                        <span class="similarity-score">(Score: {result['similarity_score']:.3f})</span>
                                    </h4>
                                    <p class="timestamp">⏰ {result['timestamp_range']}</p>
                                    <p><strong>Dialogue:</strong> {result['text']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Show search summary
                        st.markdown("---")
                        st.info(f"🎯 **Search Summary:** Found dialogue from {len(set([r['timestamp_range'] for r in filtered_results]))} different scenes")
                        
                    else:
                        st.warning(f"⚠️ No results found with similarity score >= {min_score}")
                        
                except Exception as e:
                    st.error(f"❌ Search failed: {e}")
        
        elif query and not selected_movie_id:
            st.warning("Please select a movie first!")
            
    except Exception as e:
        st.error(f"❌ Error in chat interface: {e}")

def upload_and_process_subtitle():
    """Handle subtitle file upload and processing"""
    st.subheader("📁 Upload New Movie Subtitle")
    
    if st.session_state.chromadb_store is None:
        st.warning("Please initialize ChromaDB first!")
        return
    
    uploaded_file = st.file_uploader(
        "Choose a subtitle file (.srt)", 
        type=['srt'],
        help="Upload a .srt subtitle file to add a new movie to the chatbot"
    )
    
    if uploaded_file is not None:
        # Get movie details
        col1, col2 = st.columns(2)
        with col1:
            movie_title = st.text_input("Movie Title", value="", key="movie_title")
        with col2:
            movie_year = st.number_input("Release Year", min_value=1900, max_value=2030, value=2020, key="movie_year")
        
        if st.button("Process Movie", type="primary"):
            if movie_title.strip():
                with st.spinner("Processing subtitle file..."):
                    try:
                        from src.parse_subtitles import SubtitleParser
                        
                        # Save uploaded file temporarily
                        temp_path = f"data/temp_{uploaded_file.name}"
                        os.makedirs("data", exist_ok=True)
                        
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Parse subtitles
                        parser = SubtitleParser(chunk_size=5, overlap=1)
                        subtitles, chunks = parser.process_subtitle_file(temp_path)
                        
                        # Store in ChromaDB
                        movie_id = st.session_state.chromadb_store.store_movie_chunks(movie_title, movie_year, chunks)
                        
                        # Clean up temp file
                        os.remove(temp_path)
                        
                        st.success(f"✅ Successfully processed '{movie_title}' ({movie_year})")
                        st.info(f"📊 Created {len(chunks)} searchable chunks from {len(subtitles)} subtitle entries")
                        
                        # Refresh the page to show new movie
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error processing file: {e}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
            else:
                st.warning("Please enter a movie title")

if __name__ == "__main__":
    main()
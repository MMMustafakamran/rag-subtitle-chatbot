"""
Test script to verify PostgreSQL and pgvector setup
Run this after setting up the database and updating config.py
"""

import psycopg2
from pgvector.psycopg2 import register_vector
from config import DatabaseConfig

def test_database_setup():
    """Test database connection and pgvector setup"""
    
    # Update config.py with your credentials first!
    db_config = DatabaseConfig()
    
    print("=== Testing Database Setup ===\n")
    print(f"Connecting to: {db_config.host}:{db_config.port}/{db_config.database}")
    print(f"User: {db_config.user}")
    
    try:
        # Test connection
        conn = psycopg2.connect(
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            user=db_config.user,
            password=db_config.password
        )
        
        # Register pgvector
        register_vector(conn)
        print("✅ Connected to PostgreSQL successfully!")
        
        cur = conn.cursor()
        
        # Test pgvector extension
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        vector_ext = cur.fetchone()
        if vector_ext:
            print("✅ pgvector extension is installed!")
        else:
            print("❌ pgvector extension not found!")
            print("   Run: CREATE EXTENSION vector; in pgAdmin")
            return False
        
        # Test tables
        cur.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_name IN ('movies', 'movie_chunks')
        """)
        tables = [row[0] for row in cur.fetchall()]
        
        if 'movies' in tables and 'movie_chunks' in tables:
            print("✅ Required tables exist!")
        else:
            print("❌ Required tables missing!")
            print("   Run the database_setup.sql script in pgAdmin")
            return False
        
        # Test vector operations
        cur.execute("SELECT '[1,2,3]'::vector;")
        result = cur.fetchone()
        if result:
            print("✅ Vector operations working!")
        else:
            print("❌ Vector operations failed!")
            return False
        
        # Show table info
        cur.execute("SELECT COUNT(*) FROM movies;")
        movie_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM movie_chunks;")
        chunk_count = cur.fetchone()[0]
        
        print(f"📊 Database Status:")
        print(f"   Movies: {movie_count}")
        print(f"   Chunks: {chunk_count}")
        
        cur.close()
        conn.close()
        
        print("\n🎉 Database setup is complete and working!")
        print("You can now run the PostgreSQL embedding system!")
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        print("\nPlease check:")
        print("1. PostgreSQL is running")
        print("2. Database name, username, and password in config.py")
        print("3. Database exists")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_database_setup()
    if success:
        print("\n✅ Ready to proceed with the embedding system!")
    else:
        print("\n❌ Please fix the issues above before continuing.") 
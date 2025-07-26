from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
import time
from controller.ask_controller import router as ask_router
from utils.build_faiss_index import parse_xml_to_chunks, build_and_save_faiss_index
from rag import globals as rag_globals
from rag.vector_store import load_faiss_index
from rag.content_retriever import load_documents
from rag.embedding import load_embedding_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application startup with cache initialization"""
    startup_time = time.time()
    
    try:
        print("🚀 Starting RAG Literature Chatbot...")
        
        # Initialize cache system
        print("📦 Initializing cache system...")
        from utils.cache_manager import cache_manager
        cache_manager.cleanup_expired()  # Clean up expired entries on startup
        
        # Check if FAISS index exists
        if not os.path.exists("data/faiss_index.bin"):
            print("🛠️ Building FAISS index on app startup...")
            
            # Parse all XML files
            docs = []
            xml_files = [f for f in os.listdir("data") if f.endswith(".xml")]
            
            if not xml_files:
                print("⚠️ No XML files found in data directory!")
                raise FileNotFoundError("No XML files found")
            
            for xml_file in xml_files:
                xml_path = os.path.join("data", xml_file)
                print(f"📄 Processing {xml_file}...")
                file_docs = parse_xml_to_chunks(xml_path)
                docs.extend(file_docs)
            
            print(f"📊 Total documents processed: {len(docs)}")
            build_and_save_faiss_index(docs, "shibing624/text2vec-base-chinese")
        else:
            print("✅ FAISS index already exists, skipping build...")
        
        # Load all components
        print("📦 Loading FAISS index...")
        rag_globals.index = load_faiss_index()
        
        print("🤖 Loading embedding model...")
        rag_globals.embeddings = load_embedding_model()
        
        print("📚 Loading documents...")
        rag_globals.documents = load_documents()
        
        startup_duration = time.time() - startup_time
        print(f"✅ Application startup completed in {startup_duration:.2f} seconds")
        print(f"📊 Loaded {rag_globals.index.ntotal} vectors and {len(rag_globals.documents)} documents")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    print("🔄 Shutting down application...")
    try:
        from utils.cache_manager import cache_manager
        cache_manager.shutdown()
        print("💾 Cache saved on shutdown")
    except Exception as e:
        print(f"⚠️ Error during cache shutdown: {e}")

app = FastAPI(
    title="RAG Literature Chatbot",
    description="A Vietnamese literature chatbot using RAG (Retrieval-Augmented Generation)",
    version="2.0.0",
    lifespan=lifespan
)

app.include_router(ask_router)
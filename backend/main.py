from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
from controller.ask_controller import router as ask_router
from utils.build_faiss_index import parse_xml_to_chunks, build_and_save_faiss_index
from rag import globals as rag_globals
from rag.vector_store import load_faiss_index
from rag.content_retriever import load_documents

from rag.embedding import load_embedding_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists("data/faiss_index.bin"):
        print("🛠️ Building FAISS index on app startup...")
        docs = parse_xml_to_chunks("data/result_07.xml")
        build_and_save_faiss_index(docs, "shibing624/text2vec-base-chinese")
        # build_and_save_faiss_index(docs, "BAAI/bge-large-zh")

    rag_globals.index = load_faiss_index()
    rag_globals.embeddings = load_embedding_model()
    rag_globals.documents = load_documents()
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(ask_router)
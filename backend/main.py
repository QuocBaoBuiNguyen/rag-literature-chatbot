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
        
    #Cách phèn
        # docs1 = parse_xml_to_chunks("data/result_05.xml")
        # docs3 = parse_xml_to_chunks("data/result_07.xml")
        # docs4 = parse_xml_to_chunks("data/result_12.xml")
        # docs6 = parse_xml_to_chunks("data/result_14.xml")
        # docs2 = parse_xml_to_chunks("data/result_06.xml")  
        # docs5 = parse_xml_to_chunks("data/result_13.xml")
        
        # docs = docs1  + docs3 + docs4  + docs6 + docs2 + docs5
        
    #Xịn hơn tí
        ## Gọi parse_xml_to_chunks cho từng file và nối các kết quả list lại
        docs = []
        for f in os.listdir("data"):
            if f.endswith(".xml"):
                docs.extend(parse_xml_to_chunks(os.path.join("data", f)))

        build_and_save_faiss_index(docs, "shibing624/text2vec-base-chinese")
        # build_and_save_faiss_index(docs, "BAAI/bge-large-zh")

    rag_globals.index = load_faiss_index()
    rag_globals.embeddings = load_embedding_model()
    rag_globals.documents = load_documents()
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(ask_router)
import faiss
import pickle
import numpy as np

FAISS_INDEX_PATH = "data/faiss_index.bin"
DOCS_PKL_PATH = "data/doc_chunks.pkl"

def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    return index

def search_similar_chunks(index, query_vec, chunks, k=5, min_score=0.3):
    query_vec = np.array([query_vec]).astype("float32")

    # ✅ Normalize vector nếu dùng cosine similarity
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)

    distances, indices = index.search(query_vec, k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if score >= min_score:  # score là cosine similarity nếu dùng IndexFlatIP
            results.append(chunks[idx]["text"])
    return results
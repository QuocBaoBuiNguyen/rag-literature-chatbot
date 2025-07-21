from sentence_transformers import SentenceTransformer

def load_embedding_model():
    return SentenceTransformer('shibing624/text2vec-base-chinese')

def embed_query(query: str, model) -> list[float]:
    return model.encode([query])[0]

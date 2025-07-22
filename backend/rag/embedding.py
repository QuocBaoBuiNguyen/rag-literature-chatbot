from sentence_transformers import SentenceTransformer

def load_embedding_model():
    return SentenceTransformer('shibing624/text2vec-base-chinese')
    # return SentenceTransformer('BAAI/bge-large-zh')

def embed_query(query: str, model) -> list[float]:
    return model.encode([query])[0]
    # return model.encode("为这个句子生成表示以用于检索：" + query)
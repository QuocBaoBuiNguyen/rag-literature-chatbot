import pickle

def load_documents(path="data/doc_chunks.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

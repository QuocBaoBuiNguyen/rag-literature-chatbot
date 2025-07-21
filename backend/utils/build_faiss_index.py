import os
import xml.etree.ElementTree as ET
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "shibing624/text2vec-base-chinese"  # hoặc "bge-base-vi"
FAISS_INDEX_PATH = "data/faiss_index.bin"
DOCS_PKL_PATH = "data/doc_chunks.pkl"

# Chia STC thành các chunk nhỏ (2-4 câu) với sliding window = 1
def split_into_chunks(sentences, chunk_size=3, stride=1):
    chunks = []
    for i in range(0, len(sentences) - chunk_size + 1, stride):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def parse_xml_to_chunks(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    chunks = []
    chunk_id = 0

    for page in root.findall(".//PAGE"):
        page_id = page.attrib.get("ID", "")
        stcs = [stc.text.strip() for stc in page.findall(".//STC") if stc.text]
        
        if not stcs:
            continue

        # Chia thành các đoạn nhỏ
        chunk_texts = split_into_chunks(stcs, chunk_size=3, stride=1)

        for text in chunk_texts:
            chunks.append({
                "text": text,
                "metadata": {
                    "page_id": page_id,
                    "chunk_id": f"{page_id}_{chunk_id}"
                }
            })
            chunk_id += 1

    print(f"✅ Đã tạo tổng cộng {len(chunks)} chunk từ file XML.")
    return chunks

def build_and_save_faiss_index(docs, model_name):
    model = SentenceTransformer(model_name)
    texts = [doc["text"] for doc in docs]

    print(f"📄 Đang nhúng {len(texts)} đoạn văn bản...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)  # ✅ normalize
    vectors = np.array(embeddings).astype("float32")

    print(f"📦 Vector nhúng có shape: {vectors.shape}")
    index = faiss.IndexFlatIP(vectors.shape[1])  # ✅ cosine similarity
    index.add(vectors)

    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save docs for later retrieval
    with open(DOCS_PKL_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"✅ FAISS index saved to: {FAISS_INDEX_PATH}")
    print(f"📦 Saved {len(docs)} chunks with metadata to: {DOCS_PKL_PATH}")
import os
import xml.etree.ElementTree as ET
import pickle
import numpy as np
import faiss
import openai
import json
from sentence_transformers import SentenceTransformer

# MODEL_NAME = "BAAI/bge-large-zh"  # hoặc "bge-base-vi"
MODEL_NAME = "shibing624/text2vec-base-chinese"  # hoặc "bge-base-vi"
FAISS_INDEX_PATH = "data/faiss_index.bin"
DOCS_PKL_PATH = "data/doc_chunks.pkl"

# Phân câu dài thành các câu nhỏ có ngữ nghĩa
def split_chinese_sentence_openrouter(text):
    """
    Sử dụng Qwen qua OpenRouter (free) để tách câu tiếng Trung
    """
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_AI_KEY")  # Đăng ký tại openrouter.ai
    )

    prompt = f"""
    请将下面这段中文文本分解成多个有完整语义的短句。要求：
    1. 每个短句都要有完整的语法结构和语义
    2. 保持原文的所有信息不丢失
    3. 用换行符分隔每个短句
    4. 如果文本没有明确语义，请说明并尝试合理分段

    原文：{text}
    """

    try:
        response = client.chat.completions.create(
            model="qwen/qwen-2.5-coder-32b-instruct",  # Free model
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenRouter API Error: {e}")
        return None
    
# Lấy overlap các đoạn để bao quát ngữ nghĩa: ví dụ chunk1chunk2chunk3, chunk2chunk3chunk4
def split_overlap_sematic_chunks(full_text):
    split_txt = full_text.split('\n')
    # Bỏ phần tử đầu và cuối
    trimmed = split_txt[1:-1]

    # Loại bỏ các phần tử rỗng
    result = [item for item in trimmed if item]
    result = [''.join(result[i:i+3]).replace(" ", "") for i in range(len(result) - 2)]
    return result

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
        
        
        # Call API của QWEN để tạo sematic riêng
        full_text = "".join(stcs)
        sematic_chunk_list = split_chinese_sentence_openrouter(full_text)
        
        
        
        if sematic_chunk_list != None and len(sematic_chunk_list) > 3:
            # Chia thành các đoạn nhỏ
            overlap_chunks_list = split_overlap_sematic_chunks(sematic_chunk_list)
            chunk_texts = overlap_chunks_list
        else: chunk_texts = split_into_chunks(stcs, chunk_size=3, stride=1)
        
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
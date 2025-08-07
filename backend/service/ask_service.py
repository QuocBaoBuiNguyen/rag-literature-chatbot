from rag.embedding import embed_query
from rag.vector_store import load_faiss_index, search_similar_chunks
from rag.content_retriever import load_documents
from rag import globals as rag_globals
from llm.chatbot_llm import generate_answer  # giả sử bạn dùng LLM local

# Load FAISS index và documents 1 lần khi import
# index, embeddings = load_faiss_index()
# documents = load_documents()

def ask_llm_with_rag(question: str) -> str:
    # 1. Embed câu hỏi
    query_vec = embed_query(question, rag_globals.embeddings)
    print(f"🔍 Câu hỏi đã được nhúng: {query_vec[:10]}...")  # In ra 10 giá trị đầu tiên của vector

    # 2. Tìm đoạn tương tự
    print(f"🔍 Tìm kiếm đoạn tương tự trong FAISS index.... Index {rag_globals.index}, Documents {rag_globals.documents[:5]}")
    top_chunks = search_similar_chunks(rag_globals.index, query_vec, rag_globals.documents, k=3)
    print(f"🔍 Tìm thấy {len(top_chunks)} đoạn tương tự cho câu hỏi...")

    # 3. Tạo prompt cho LLM
    context = "\n\n".join(top_chunks)
    prompt = f"""Dựa trên các đoạn sau:

            {context}

            Câu hỏi: {question}
            Trả lời ngắn gọn, bằng tiếng Việt dễ hiểu."""

    print(f"🔍 Prompt cho LLM: {prompt[:500]}...")  # In ra 100 ký tự đầu tiên của prompt
    # 4. Gọi LLM local sinh câu trả lời
    return generate_answer(prompt)

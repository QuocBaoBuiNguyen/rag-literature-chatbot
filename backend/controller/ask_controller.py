from fastapi import APIRouter
from pydantic import BaseModel
import pickle

from service.ask_service import ask_llm_with_rag

router = APIRouter()

class AskRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask(req: AskRequest):
    answer, context = ask_llm_with_rag(req.question)
    return {"answer": answer, "context": context}

@router.get("/debug/chunks")
def get_chunks_preview():
    try:
        with open("data/doc_chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        preview = [
            {
                "index": i,
                "page_id": chunk["metadata"].get("page_id", ""),
                "text_preview": chunk["text"][:200]  # chỉ lấy 200 ký tự đầu
            }
            for i, chunk in enumerate(chunks[:30])
        ]
        return {"preview": preview, "total_chunks": len(chunks)}

    except Exception as e:
        return {"error": str(e)}

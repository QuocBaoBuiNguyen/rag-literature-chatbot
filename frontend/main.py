import chainlit as cl
import requests

BACKEND_URL = "http://localhost:8000/ask"

@cl.on_chat_start
async def start():
    await cl.Message("Chào bạn! Hãy hỏi tôi về văn học.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    response = requests.post(BACKEND_URL, json={"question": message.content})
    try:
        answer = response.json()["answer"]
        await cl.Message(answer).send()
    except Exception as e:
        await cl.Message("Có lỗi xảy ra khi trả lời. Vui lòng thử lại sau.").send()
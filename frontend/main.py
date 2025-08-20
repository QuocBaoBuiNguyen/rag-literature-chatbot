import chainlit as cl
import requests

BACKEND_URL = "http://localhost:8000/ask"

@cl.on_chat_start
async def start():
    await cl.Message("Chào bạn! Hãy hỏi tôi về văn học.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    try:
        response = requests.post(BACKEND_URL, json={"question": message.content})
        answer = response.json()["answer"]
        await cl.Message(answer).send()
    except Exception as e:
        await cl.Message("Có lỗi xảy ra, vui lòng thử lại sau.").send()
        
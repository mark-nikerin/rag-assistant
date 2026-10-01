from dotenv import load_dotenv
from pathlib import Path

root = Path(__file__).resolve().parents[3]
env_path = root / ".env"
load_dotenv(env_path)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from rag_shared.db import create_async_session_factory
from assistant_service.retrieval import retrieve_context

app = FastAPI(title="RAG Assistant", version="0.1.0")

client = AsyncOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",  # LM Studio не требует ключ
)

session_factory = create_async_session_factory()


class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    async with session_factory() as session:

        context_chunks = await retrieve_context(request.query, session, top_k=5)

        if context_chunks:
            context_text = "\n\n".join(context_chunks)
            prompt = f"""Контекст из документов:
        {context_text}

        Вопрос: {request.query}

        Ответь на вопрос, используя контекст выше. Если контекст не содержит информации для ответа, скажи об этом."""
        else:
            prompt = f"""Вопрос: {request.query}

        Ответь на вопрос. Если не знаешь ответа, скажи об этом."""

        response = await client.responses.create(
            model="prism-ml/bonsai-27b",
            input=prompt
        )

        final_answer = ""
        for item in response.output:
            if hasattr(item, "type") and item.type == "reasoning":
                if hasattr(item, "content") and item.content:
                    for content_item in item.content:
                        if hasattr(content_item, "text") and content_item.text:
                            print(f"Reasoning: {content_item.text}")

            if hasattr(item, "type") and item.type == "message":
                if hasattr(item, "content") and item.content:
                    for content_item in item.content:
                        if hasattr(content_item, "text") and content_item.text:
                            final_answer = content_item.text
                            break
                if final_answer:
                    break

        conversation_id = request.conversation_id or "stub"
        return ChatResponse(answer=final_answer or "No answer generated", conversation_id=conversation_id)
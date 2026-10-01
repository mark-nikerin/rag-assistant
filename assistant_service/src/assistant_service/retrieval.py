from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from rag_shared.models import Chunk
from openai import AsyncOpenAI


EMBEDDING_MODEL = "text-embedding-bge-m3"

async def retrieve_context(query: str, session: AsyncSession, top_k: int = 5) -> list[str]:
    """
    Поиск топ-K чанков по косинусному сходству через pgvector.
    Возвращает список текстов чанков для включения в промпт.
    """

    client = AsyncOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio"
    )

    embedding_response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = embedding_response.data[0].embedding

    stmt = (
        select(Chunk)
        .order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )

    result = await session.execute(stmt)
    chunks = result.scalars().all()

    return [chunk.content for chunk in chunks]
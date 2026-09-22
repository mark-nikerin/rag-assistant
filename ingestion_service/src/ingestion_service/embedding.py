from openai import OpenAI


_client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
)

EMBEDDING_MODEL = "text-embedding-bge-m3"


def get_embedding(text: str) -> list[float]:
    cleaned = text.replace("\n", " ").strip()
    response = _client.embeddings.create(input=[cleaned], model=EMBEDDING_MODEL)
    return response.data[0].embedding
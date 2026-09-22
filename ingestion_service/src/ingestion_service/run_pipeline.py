import asyncio
from dotenv import load_dotenv
from pathlib import Path

root = Path(__file__).resolve().parents[3]
env_path = root / ".env"
load_dotenv(env_path)

from rag_shared.db import create_async_session_factory
from ingestion_service.pipeline import run_pipeline


async def main():
    session_factory = create_async_session_factory()
    await run_pipeline(Path("source"), session_factory)


if __name__ == "__main__":
    asyncio.run(main())
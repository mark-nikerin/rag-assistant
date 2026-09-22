import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_shared.models import SourceFile, Document, Chunk

@dataclass
class FileInfo:
    content_hash: str
    size_bytes: int
    mtime: datetime
    source_path: Path

@dataclass
class FileDecision:
    file: FileInfo
    action: str  # "inserted", "reindexed", "skipped"
    existing_file: SourceFile | None = None


async def upsert_source_file(
    session: AsyncSession,
    file_info: FileInfo,
    embedding_model: str = "text-embedding-3-large",
    chunking_strategy: str = "semantic_v1",
) -> SourceFile:
    stmt = select(SourceFile).where(
        SourceFile.content_hash == file_info.content_hash
    )
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        existing.file_size_bytes = file_info.size_bytes
        existing.mtime = file_info.mtime
        existing.status = "pending"
        existing.last_indexed_at = None
        await session.flush()
        return existing
    else:
        new_file = SourceFile(
            content_hash=file_info.content_hash,
            file_size_bytes=file_info.size_bytes,
            mtime=file_info.mtime,
            mime_type="text/plain",
            status="pending",
            embedding_model=embedding_model,
            chunking_strategy=chunking_strategy,
        )
        session.add(new_file)
        await session.flush()
        return new_file


async def create_document_and_chunks(
        session: AsyncSession,
        source_file: SourceFile,
        content: str,
        embedding_model: str = "text-embedding-3-large",
) -> list[Chunk]:
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    document = Document(
        source_file_id=source_file.id,
        title=source_file.content_hash[:16],  # временное название
        doc_type="text",
        metadata_={},
    )
    session.add(document)
    await session.flush()

    chunks: list[Chunk] = []
    for idx, paragraph in enumerate(paragraphs):
        chunk = Chunk(
            document_id=document.id,
            chunk_index=idx,
            content=paragraph,
            content_hash=hashlib.sha256(paragraph.encode()).hexdigest(),
            token_count=len(paragraph.split()),
            embedding=None,
            embedding_model=embedding_model,
            metadata_={},
            is_active=True,
        )
        session.add(chunk)
        chunks.append(chunk)

    await session.flush()
    return chunks


def compute_file_hash(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_source_folder(source_root: Path) -> list[FileInfo]:
    supported_extensions = {".txt", ".md", ".pdf"}
    files: list[FileInfo] = []

    for path in source_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported_extensions:
            stat = path.stat()
            files.append(
                FileInfo(
                    content_hash=compute_file_hash(path),
                    size_bytes=stat.st_size,
                    mtime=datetime.fromtimestamp(stat.st_mtime).replace(tzinfo=None),
                    source_path=path,
                )
            )
    return files


async def decide_actions(session: AsyncSession, files: list[FileInfo]) -> list[FileDecision]:
    decisions: list[FileDecision] = []

    for file_info in files:
        stmt = select(SourceFile).where(
            SourceFile.content_hash == file_info.content_hash
        )

        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing is None:
            decisions.append(
                FileDecision(file=file_info, action="inserted")
            )
        elif existing.content_hash != file_info.content_hash:
            decisions.append(
                FileDecision(
                    file=file_info,
                    action="reindexed",
                    existing_file=existing,
                )
            )
        else:
            decisions.append(
                FileDecision(
                    file=file_info,
                    action="skipped",
                    existing_file=existing,
                )
            )

    return decisions


async def run_pipeline(
        source_root: Path,
        session_factory: async_sessionmaker[AsyncSession],
):
    print(f"Scanning {source_root}...")
    files = scan_source_folder(source_root)
    print(f"Found {len(files)} files")

    async with session_factory() as session:
        decisions = await decide_actions(session, files)

        for d in decisions:
            if d.action == "skipped":
                print(f"{d.file.source_path.name} -> skipped")
                continue

            print(f"{d.file.source_path.name} -> {d.action}")

            # Записываем в БД
            source_file = await upsert_source_file(session, d.file)

            # Читаем содержимое файла
            content = d.file.source_path.read_text(encoding="utf-8")

            # Создаём документ и чанки
            chunks = await create_document_and_chunks(session, source_file, content)

            # Помечаем файл как проиндексированный
            source_file.status = "indexed"
            source_file.last_indexed_at = datetime.now(timezone.utc).replace(tzinfo=None)

            await session.commit()

            print(f"  Created {len(chunks)} chunks")

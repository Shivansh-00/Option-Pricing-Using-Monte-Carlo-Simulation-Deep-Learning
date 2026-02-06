from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    title: str
    content: str


def load_documents(directory: str | Path) -> list[Document]:
    documents: list[Document] = []
    for file in Path(directory).glob("*"):
        if file.is_file():
            documents.append(Document(title=file.name, content=file.read_text(errors="ignore")))
    return documents

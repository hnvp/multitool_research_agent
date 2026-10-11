from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from config import settings

DEFAULT_PDF_PATH = Path(__file__).parent / "data" / "documents" / "tom_tat.pdf"


def clean_source_name(filename: str) -> str:
    """Convert a filename into a stable ASCII source identifier."""
    text = unicodedata.normalize(
        "NFKD",
        filename,
    )

    text = text.encode(
        "ascii",
        "ignore",
    ).decode("ascii")

    text = re.sub(
        r"[^a-zA-Z0-9_.-]",
        "_",
        text,
    )

    return text


def extract_text(pdf_path: Path) -> str:
    """Extract text from a PDF using its existing text layer."""
    document = pymupdf.open(pdf_path)

    try:
        pages = [page.get_text() for page in document]

        return "\n".join(pages)

    finally:
        document.close()


def is_noisy(chunk: str) -> bool:
    chars = [c for c in chunk if not c.isspace()]

    return not chars or sum(c.isalpha() for c in chars) / len(chars) < 0.5


def ingest_pdf(pdf_path: Path) -> str:
    """Extract, chunk, embed, and index a PDF into Pinecone.

    Returns the source identifier used in Pinecone metadata.
    """
    pdf_path = pdf_path.resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured in .env")

    text = extract_text(pdf_path)

    if not text.strip():
        raise RuntimeError(
            "No text was extracted from the PDF."
            "The PDF may be image-based or scanned and require OCR."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    chunks = splitter.split_text(text)

    chunks = [chunk for chunk in chunks if not is_noisy(chunk)]

    if not chunks:
        raise RuntimeError("No usable chunks were created.")

    embedding_model = SentenceTransformer(settings.rag_embedding_model)

    embeddings = embedding_model.encode(
        ["passage: " + chunk for chunk in chunks],
        show_progress_bar=False,
    )

    pc = Pinecone(api_key=settings.pinecone_api_key)

    index = pc.Index(settings.pinecone_index)

    source_name = clean_source_name(pdf_path.name)

    vectors = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append(
            {
                "id": f"{source_name}_{i}",
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk,
                    "source": source_name,
                    "chunk_id": i,
                },
            }
        )

    batch_size = 50

    for start in range(
        0,
        len(vectors),
        batch_size,
    ):
        batch = vectors[start : start + batch_size]

        index.upsert(vectors=batch)

    return source_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Ingest a PDF into the project's " "Pinecone RAG index.")
    )

    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=DEFAULT_PDF_PATH,
        help=(
            "Path to the PDF to ingest. " "Defaults to the project's sample document."
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()

    print(f"PDF: {pdf_path}")
    print(f"Pinecone index: " f"{settings.pinecone_index}")
    print(f"Embedding model: " f"{settings.rag_embedding_model}")

    source_name = ingest_pdf(pdf_path)

    print()
    print("=" * 60)
    print("INGEST COMPLETE")
    print("=" * 60)
    print(f"Index: " f"{settings.pinecone_index}")
    print(f"Source: {source_name}")


if __name__ == "__main__":
    main()

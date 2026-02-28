"""
Initialize rulebook from PDF or text file.
"""

import argparse
import logging
import sys
from pathlib import Path

from core.rag_engine import RAGEngine
from core.chunking import TextChunker
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Initialize rulebook")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to rulebook file (PDF or text)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for text splitting"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap"
    )

    args = parser.parse_args()

    # Load file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loading file: {input_path}")

    # Read file based on extension
    suffix = input_path.suffix.lower()

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(input_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif suffix == ".docx":
        from docx import Document
        doc = Document(input_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    elif suffix in [".md", ".txt"]:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        logger.error(f"Unsupported file format: {suffix}")
        logger.info("Supported formats: .pdf, .docx, .md, .txt")
        sys.exit(1)

    logger.info(f"Loaded {len(text)} characters from file")

    # Initialize RAG engine
    logger.info("Initializing RAG engine...")
    rag_engine = RAGEngine(embedding_model=args.model)

    # Ingest document
    logger.info("Ingesting document into vector store...")
    chunks_count = rag_engine.ingest_document(text, metadata={"source": str(input_path), "document_name": input_path.name})

    # Save vector store
    logger.info("Saving vector store...")
    rag_engine.vector_store.save()

    logger.info("=" * 60)
    logger.info("Rulebook initialization complete!")
    logger.info(f"Chunks stored: {chunks_count}")
    logger.info(f"Vector store saved to: data/vector_store")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

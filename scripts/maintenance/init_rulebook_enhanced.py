"""
Initialize rulebook from PDF or text file (enhanced parser fallback).
"""

import argparse
import logging
import sys
from pathlib import Path

from core.rag_engine import RAGEngine
from loguru import logger


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def load_pdf_pymupdf(input_path: Path) -> str:
    try:
        import fitz
    except ImportError:
        logger.error("PyMuPDF not installed. Run: pip install pymupdf")
        return ""

    doc = fitz.open(str(input_path))
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def load_pdf_pypdf(input_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        return ""

    reader = PdfReader(str(input_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_text_file(input_path: Path) -> str:
    for encoding in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            return input_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return ""


def load_file(input_path: Path, use_pymupdf: bool = True) -> str:
    suffix = input_path.suffix.lower()

    if suffix == ".pdf":
        if use_pymupdf:
            text = load_pdf_pymupdf(input_path)
            if len(text.strip()) >= 50:
                return text
            logger.warning("PyMuPDF extraction too short; fallback to pypdf.")
        return load_pdf_pypdf(input_path)

    if suffix == ".docx":
        try:
            import docx
        except ImportError:
            logger.error("python-docx not installed. Run: pip install python-docx")
            return ""
        doc = docx.Document(str(input_path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    if suffix in {".txt", ".md", ".markdown"}:
        return load_text_file(input_path)

    logger.error(f"Unsupported file format: {suffix}")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize rulebook")
    parser.add_argument("--input", type=str, required=True, help="Path to rulebook file")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model name",
    )
    parser.add_argument("--no-pymupdf", action="store_true", help="Use pypdf only for PDFs")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        raise SystemExit(1)

    logger.info(f"Loading file: {input_path}")
    text = load_file(input_path, use_pymupdf=not args.no_pymupdf)
    if not text:
        logger.error("Failed to extract text from input file")
        raise SystemExit(1)

    logger.info(f"Extracted {len(text)} characters")
    rag_engine = RAGEngine(embedding_model=args.model)
    chunks_count = rag_engine.ingest_document(
        text,
        metadata={"source": str(input_path), "document_name": input_path.name},
    )
    rag_engine.vector_store.save()

    logger.info("=" * 60)
    logger.info("Rulebook initialization complete")
    logger.info(f"Chunks stored: {chunks_count}")
    logger.info("Vector store: data/vector_store")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

"""
Initialize rulebooks with OCR fallback for scanned PDFs.
"""

import argparse
import io
import logging
import shutil
from pathlib import Path

from core.rag_engine import RAGEngine
from loguru import logger


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _extract_pdf_text_pymupdf(input_path: Path) -> str:
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF not installed; cannot run PDF extraction")
        return ""

    doc = fitz.open(str(input_path))
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def _extract_pdf_text_ocr(input_path: Path) -> str:
    try:
        import fitz
        from PIL import Image
    except ImportError:
        logger.warning("Missing OCR prerequisites (pymupdf/Pillow)")
        return ""

    # Initialize OCR engine lazily; prefer PaddleOCR, fallback to pytesseract.
    use_paddle = False
    ocr = None
    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=True, lang="ch")
        use_paddle = True
    except ImportError:
        try:
            import pytesseract  # type: ignore

            ocr = pytesseract
        except ImportError:
            logger.warning("No OCR engine installed (paddleocr or pytesseract)")
            return ""

    doc = fitz.open(str(input_path))
    pages_text = []
    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        page_text = ""
        if use_paddle:
            result = ocr.ocr(img, cls=True)
            for line in result:
                for word_info in line:
                    page_text += word_info[1][0] + "\n"
        else:
            page_text = ocr.image_to_string(img, lang="chi_sim+eng")

        pages_text.append(page_text)
        logger.info(f"OCR processed page {page_num}/{len(doc)}")

    doc.close()
    return "\n".join(pages_text).strip()


def load_with_ocr(input_path: Path) -> str:
    suffix = input_path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
            try:
                return input_path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return ""

    if suffix != ".pdf":
        logger.warning(f"Unsupported file extension for OCR loader: {suffix}")
        return ""

    text = _extract_pdf_text_pymupdf(input_path)
    if len(text) >= 500:
        logger.info(f"PyMuPDF extracted {len(text)} chars from {input_path.name}")
        return text

    logger.info(f"PyMuPDF extraction too short ({len(text)} chars), trying OCR...")
    text = _extract_pdf_text_ocr(input_path)
    if text:
        logger.info(f"OCR extracted {len(text)} chars from {input_path.name}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize rulebooks with OCR")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=["data/rules/1.pdf", "data/rules/2.pdf", "data/rules/3.pdf"],
        help="Input file paths",
    )
    parser.add_argument("--clean", action="store_true", help="Clean vector store before indexing")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model",
    )
    args = parser.parse_args()

    if args.clean:
        vector_store_path = Path("data/vector_store")
        if vector_store_path.exists():
            shutil.rmtree(vector_store_path)
            logger.info("Vector store cleaned")

    rag_engine = RAGEngine(embedding_model=args.model)
    total_chunks = 0

    for input_str in args.input:
        input_path = Path(input_str)
        if not input_path.exists():
            logger.warning(f"File not found, skipping: {input_path}")
            continue

        logger.info(f"Processing: {input_path.name}")
        text = load_with_ocr(input_path)
        if not text or len(text) < 100:
            logger.warning(f"Insufficient text extracted, skipping: {input_path.name}")
            continue

        chunks_count = rag_engine.ingest_document(
            text,
            metadata={"source": str(input_path), "document_name": input_path.name},
        )
        total_chunks += chunks_count
        logger.info(f"Indexed {chunks_count} chunks from {input_path.name}")

    rag_engine.vector_store.save()
    logger.info(f"All done. Total chunks indexed: {total_chunks}")


if __name__ == "__main__":
    main()

"""
Initialize multiple rulebooks at once
"""
import argparse
import sys
from pathlib import Path

from core.rag_engine import RAGEngine
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
from loguru import logger


def load_pdf_pymupdf(input_path: Path) -> str:
    """使用 PyMuPDF (fitz) 读取 PDF"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF 未安装，请运�? pip install pymupdf")
        return None

    logger.info("使用 PyMuPDF 读取 PDF...")
    doc = fitz.open(input_path)
    text = ""

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        text += page_text + "\n"
        logger.info(f"  - 已读取第 {page_num}/{len(doc)} �?({len(page_text)} 字符)")

    doc.close()
    return text


def load_pdf_pypdf(input_path: Path) -> str:
    """使用 pypdf 读取 PDF"""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf 未安装，请运�? pip install pypdf")
        return None

    logger.info("使用 pypdf 读取 PDF...")
    reader = PdfReader(input_path)
    text = ""

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        text += page_text + "\n"
        logger.info(f"  - 已读取第 {page_num}/{len(reader.pages)} �?({len(page_text)} 字符)")

    return text


def load_pdf_pdfplumber(input_path: Path) -> str:
    """使用 pdfplumber 读取 PDF"""
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber 未安装，请运�? pip install pdfplumber")
        return None

    logger.info("使用 pdfplumber 读取 PDF...")
    text = ""

    with pdfplumber.open(input_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            logger.info(f"  - 已读取第 {page_num}/{len(pdf.pages)} �?({len(page_text) if page_text else 0} 字符)")

    return text


def load_file(input_path: Path, use_pymupdf: bool = True) -> str:
    """根据文件扩展名读取文�?""
    suffix = input_path.suffix.lower()

    if suffix == ".pdf":
        # 尝试多种方法，按优先�?        methods = []

        # 方法1: PyMuPDF (如果可用且启�?
        if use_pymupdf:
            methods.append(("PyMuPDF", load_pdf_pymupdf))

        # 方法2: pdfplumber (更强大的文本提取)
        methods.append(("pdfplumber", load_pdf_pdfplumber))

        # 方法3: pypdf (默认备�?
        methods.append(("pypdf", load_pdf_pypdf))

        for method_name, method_func in methods:
            logger.info(f"尝试 {method_name} 读取...")
            text = method_func(input_path)

            if text and len(text) > 50:
                logger.info(f"�?{method_name} 成功提取 {len(text)} 字符")
                return text
            else:
                logger.info(f"�?{method_name} 提取文本过少或失�?)

        # 所有方法都失败�?        logger.error("所有PDF读取方法都失败了")
        logger.error("这可能是扫描版PDF，需要使用OCR")
        logger.error("请运�? python init_with_ocr.py")
        return None

    else:
        # 读取文本文件
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        for encoding in encodings:
            try:
                with open(input_path, "r", encoding=encoding) as f:
                    text = f.read()
                logger.info(f"使用编码 {encoding} 读取文本文件")
                return text
            except UnicodeDecodeError:
                continue

        logger.error("无法用任何编码读取文本文�?)
        return None


def main():
    parser = argparse.ArgumentParser(description="Initialize multiple rulebooks")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="Paths to rulebook files (PDF or text)",
        default=["data/rules/1.pdf", "data/rules/2.pdf", "data/rules/3.pdf"]
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model name (use multilingual for Chinese+English)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing vector store first"
    )

    args = parser.parse_args()

    # Clean existing vector store if requested
    if args.clean:
        logger.info("=" * 60)
        logger.info("🗑�? 清理现有向量�?..")
        logger.info("=" * 60)
        import shutil
        vector_store_path = Path("data/vector_store")
        if vector_store_path.exists():
            shutil.rmtree(vector_store_path)
            logger.info("�?向量库已清理")
        else:
            logger.info("�?向量库不存在，无需清理")

    # Initialize RAG engine (will load existing vector store)
    logger.info("=" * 60)
    logger.info("🚀 初始�?RAG 引擎...")
    logger.info("=" * 60)
    rag_engine = RAGEngine(embedding_model=args.model)

    total_chunks = 0

    # Process each file
    for input_path_str in args.input:
        input_path = Path(input_path_str)

        if not input_path.exists():
            logger.warning(f"⚠️  文件不存在，跳过: {input_path}")
            continue

        logger.info("=" * 60)
        logger.info(f"📄 加载规则�? {input_path.name}")
        logger.info("=" * 60)

        # Read file
        text = load_file(input_path, use_pymupdf=True)

        if text is None:
            logger.error(f"�?文件读取失败: {input_path}")
            continue

        if len(text) < 100:
            logger.warning(f"⚠️  提取的文本过�?({len(text)} 字符)，可能是扫描�?PDF")
            logger.warning(f"   - 跳过此文�? {input_path}")
            logger.warning("   - 建议: 安装 PyMuPDF (pip install pymupdf) 或使�?OCR 工具")
            continue

        logger.info(f"�?成功加载 {len(text)} 个字�?)

        # Ingest document
        logger.info("📦 将文档添加到向量�?..")
        chunks_count = rag_engine.ingest_document(text, metadata={"source": str(input_path), "document_name": input_path.name})

        if chunks_count == 0:
            logger.warning(f"⚠️  未生成任何文本块，跳�? {input_path}")
            continue

        total_chunks += chunks_count

        logger.info(f"�?添加�?{chunks_count} 个文本块\n")

    # Save vector store
    logger.info("=" * 60)
    logger.info("💾 保存向量�?..")
    logger.info("=" * 60)
    rag_engine.vector_store.save()

    logger.info("=" * 60)
    logger.info("�?所有规则书初始化完成！")
    logger.info(f"   - 本次新增文本块数�? {total_chunks}")
    logger.info(f"   - 向量库位�? data/vector_store")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

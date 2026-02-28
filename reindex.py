"""
Re-index all rulebooks in data/rules/ into the vector store.
Supports: PDF, DOCX, TXT, MD, MARKDOWN.

Usage:
    python reindex.py
    python reindex.py --clean
"""

import argparse
import logging
import re
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from core.rag_engine import RAGEngine
from loguru import logger


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown"}


def load_pdf(path: Path) -> str:
    try:
        import fitz
    except ImportError:
        logger.error("PyMuPDF not installed: pip install pymupdf")
        return ""

    doc = fitz.open(str(path))
    text_parts = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(text_parts).strip()


def load_docx(path: Path) -> str:
    try:
        import docx
    except ImportError:
        logger.warning("python-docx not installed, falling back to ZIP/XML parsing")
        return load_docx_fallback(path)

    doc = docx.Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    if parts:
        return "\n".join(parts)
    return load_docx_fallback(path)


def _extract_docx_text_from_xml(xml_bytes: bytes) -> str:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for p in root.findall(".//w:p", ns):
        segs = []
        for t in p.findall(".//w:t", ns):
            if t.text:
                segs.append(t.text)
        if segs:
            paragraphs.append("".join(segs).strip())

    return "\n".join(x for x in paragraphs if x).strip()


def load_docx_fallback(path: Path) -> str:
    """Extract DOCX text without python-docx (best effort)."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.startswith("word/") and n.endswith(".xml")]
            if not names:
                return ""

            def order_key(name: str) -> tuple[int, str]:
                if name == "word/document.xml":
                    return (0, name)
                if "header" in name:
                    return (1, name)
                if "footer" in name:
                    return (2, name)
                return (3, name)

            text_parts = []
            for name in sorted(names, key=order_key):
                text = _extract_docx_text_from_xml(zf.read(name))
                if text:
                    text_parts.append(text)
            return "\n".join(text_parts).strip()
    except Exception as e:
        logger.error(f"DOCX fallback extraction failed for {path.name}: {e}")
        return ""


def load_text(path: Path) -> str:
    for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return ""


def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    if suffix in {".txt", ".md", ".markdown"}:
        return load_text(path)
    return ""


def detect_game_name(filepath: Path, text: str) -> str:
    # Prefer filename first.
    fname = filepath.stem.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\b(rules?|rulebook)\b", "", fname, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?:\u89c4\u5219|\u8bf4\u660e\u4e66?)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and not cleaned.isdigit():
        return cleaned

    sample = text[:3000]

    # Chinese title markers: 《Game》
    cn_title = re.search(r"[\u300a<](.+?)[\u300b>]", sample)
    if cn_title:
        return cn_title.group(1).strip()

    # Pattern like: 在XX这款游戏中...
    cn_game = re.search(r"\u5728(.+?)(?:\u8fd9\u6b3e|\u8fd9\u4e2a)?\u6e38\u620f\u4e2d", sample)
    if cn_game:
        return cn_game.group(1).strip()

    # Pattern like: GameName ~ Rules
    for line in sample.splitlines()[:30]:
        line = line.strip()
        match = re.search(r"^([A-Za-z][\w\s]{2,40}?)\s*~\s*Rules", line)
        if match:
            return match.group(1).strip()

    # Pattern like: GameName is a game of...
    en_match = re.search(r"\b([A-Z][A-Za-z\s]{2,40})\s+is\s+a\s+game\b", sample)
    if en_match:
        return en_match.group(1).strip()

    return filepath.stem


def normalize_rule_name(name: str) -> str:
    s = (name or "").strip().replace("_", " ").replace("-", " ")
    s = re.sub(r"\b(rules?|rulebook|manual|official|english|cn|zh|chs|cht)\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:规则|说明书?)", " ", s)
    s = re.sub(r"\b(19|20)\d{2}\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_expansion_name(name: str) -> bool:
    lower = (name or "").lower()
    if any(k in lower for k in ("expansion", "extension", "promo", "module")):
        return True
    return any(k in name for k in ("扩展", "拓展"))


def strip_expansion_suffix(name: str) -> str:
    stripped = re.sub(
        r"(?i)\s*(?:扩展|拓展|expansion|extension|promo|module)\b.*$",
        "",
        name or "",
    ).strip(" _-")
    return stripped


def infer_game_identity(filepath: Path, detected_name: str, all_rule_names: list[str]) -> dict:
    """
    Build expansion-aware identity metadata.

    - game_name: document-specific name (e.g. "折纸与船胡椒扩展")
    - base_game_name: base game (e.g. "折纸与船")
    - is_expansion: whether document looks like expansion
    - expansion_name: expansion-specific label when available
    """
    game_name = normalize_rule_name(detected_name) or normalize_rule_name(filepath.stem) or filepath.stem
    base_game_name = game_name
    expansion_name = ""
    is_expansion = is_expansion_name(game_name)

    if is_expansion:
        best_base = ""
        for candidate in all_rule_names:
            cand = normalize_rule_name(candidate)
            if not cand or cand == game_name:
                continue
            if len(cand) < 2:
                continue
            if game_name.startswith(cand) and len(cand) > len(best_base):
                best_base = cand

        if not best_base:
            stripped = strip_expansion_suffix(game_name)
            if stripped and stripped != game_name:
                best_base = stripped

        if best_base:
            base_game_name = best_base
            remain = game_name[len(best_base):].strip(" _-")
            expansion_name = remain if remain else game_name
        else:
            expansion_name = game_name

    return {
        "game_name": game_name,
        "base_game_name": base_game_name,
        "is_expansion": is_expansion,
        "expansion_name": expansion_name,
    }


def is_text_readable(text: str, min_ratio: float = 0.3) -> bool:
    if len(text) < 50:
        return False
    sample = text[:2000]
    readable = sum(
        1
        for c in sample
        if ("\u4e00" <= c <= "\u9fff") or c.isascii()
    )
    return readable / max(len(sample), 1) > min_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-index all rulebooks")
    parser.add_argument("--clean", action="store_true", help="Clean vector store first")
    parser.add_argument("--dir", type=str, default="data/rules", help="Rules directory")
    args = parser.parse_args()

    rules_dir = Path(args.dir)
    if not rules_dir.exists():
        logger.error(f"Rules directory not found: {rules_dir}")
        raise SystemExit(1)

    if args.clean:
        vs_path = Path("data/vector_store")
        if vs_path.exists():
            shutil.rmtree(vs_path)
            logger.info("Vector store cleaned")

    logger.info("Initializing RAG engine...")
    rag_engine = RAGEngine()

    files = sorted(
        f for f in rules_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES and not f.name.startswith("~$")
    )
    all_rule_names = [normalize_rule_name(f.stem) or f.stem for f in files]
    logger.info(f"Found {len(files)} files to index")

    total_chunks = 0
    for filepath in files:
        logger.info(f"--- Processing: {filepath.name} ---")

        text = load_file(filepath)
        if not text:
            logger.warning(f"SKIP: No text extracted from {filepath.name}")
            continue

        if not is_text_readable(text):
            logger.warning(f"SKIP: Text not readable in {filepath.name}")
            continue

        detected_game_name = detect_game_name(filepath, text)
        identity = infer_game_identity(filepath, detected_game_name, all_rule_names)
        game_name = identity["game_name"]
        logger.info(
            f"Detected game name: {game_name} | "
            f"base: {identity['base_game_name']} | "
            f"expansion: {identity['is_expansion']}"
        )
        logger.info(f"Extracted {len(text)} characters")

        chunks = rag_engine.ingest_document(
            text,
            metadata={
                "source": str(filepath),
                "document_name": filepath.name,
                "game_name": game_name,
                "base_game_name": identity["base_game_name"],
                "is_expansion": identity["is_expansion"],
                "expansion_name": identity["expansion_name"],
            },
        )
        total_chunks += chunks
        logger.info(f"Indexed {chunks} chunks")

    rag_engine.vector_store.save()
    logger.info(f"Done. Total chunks indexed: {total_chunks}")
    logger.info("Vector store saved to data/vector_store/")


if __name__ == "__main__":
    main()

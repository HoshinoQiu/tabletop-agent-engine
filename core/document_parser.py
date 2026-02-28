"""
PDF Structure Parser: Extracts text with structural information from PDFs.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path


from loguru import logger


class PDFStructureParser:
    """Parse PDF documents extracting structural information like headings and sections."""

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a PDF file and extract structured blocks with metadata.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of blocks with text, page_number, is_heading, font_size, section_title
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not installed. Falling back to plain text extraction.")
            return self._fallback_parse(file_path)

        blocks = []
        current_section = ""

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            return []

        for page_num, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # Skip non-text blocks
                    continue

                for line in block.get("lines", []):
                    line_text = ""
                    max_font_size = 0
                    is_bold = False

                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                        font_size = span.get("size", 12)
                        if font_size > max_font_size:
                            max_font_size = font_size
                        if "bold" in span.get("font", "").lower():
                            is_bold = True

                    line_text = line_text.strip()
                    if not line_text:
                        continue

                    # Detect headings: larger font or bold text that's short
                    is_heading = (max_font_size > 14 or (is_bold and len(line_text) < 80))

                    if is_heading:
                        current_section = line_text

                    blocks.append({
                        "text": line_text,
                        "page_number": page_num,
                        "font_size": max_font_size,
                        "is_bold": is_bold,
                        "is_heading": is_heading,
                        "section_title": current_section,
                    })

        doc.close()

        logger.info(f"Parsed {len(blocks)} text blocks from {file_path}")
        return blocks

    def _fallback_parse(self, file_path: str) -> List[Dict[str, Any]]:
        """Fallback parser using pypdf for plain text extraction."""
        blocks = []
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                for line in text.split("\n"):
                    line = line.strip()
                    if line:
                        blocks.append({
                            "text": line,
                            "page_number": page_num,
                            "font_size": 12,
                            "is_bold": False,
                            "is_heading": False,
                            "section_title": "",
                        })
        except Exception as e:
            logger.error(f"Fallback PDF parsing failed: {e}")
        return blocks

    def parse_to_text(self, file_path: str) -> str:
        """Parse a file and return full text. Supports PDF, TXT, and Markdown."""
        ext = Path(file_path).suffix.lower()

        # Plain text / Markdown â€?read directly
        if ext in (".txt", ".md", ".markdown"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                logger.info(f"Read {len(text)} chars from text file: {file_path}")
                return text
            except Exception as e:
                logger.error(f"Failed to read text file {file_path}: {e}")
                return ""

        # PDF â€?use structured parser
        blocks = self.parse(file_path)
        lines = []
        for block in blocks:
            if block["is_heading"]:
                lines.append(f"\n## {block['text']}\n")
            else:
                lines.append(block["text"])
        return "\n".join(lines)

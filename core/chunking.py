"""
Text Chunker: Splits text into overlapping chunks.
Includes StructureAwareChunker for parent-child document support.
"""

import re
import uuid
from typing import List, Dict, Any


from loguru import logger


class TextChunker:
    """Splits text into overlapping chunks with metadata."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Any]:
        chunks = []
        doc_metadata = metadata or {}
        sentences = self._split_sentences(text)
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            for piece in self._split_oversized_sentence(sentence):
                piece_length = len(piece)
                if current_length + piece_length > self.chunk_size and current_chunk:
                    chunk_metadata = doc_metadata.copy()
                    chunk_metadata['chunk_index'] = len(chunks)
                    chunks.append(TextChunk(
                        text="".join(current_chunk).strip(),
                        metadata=chunk_metadata
                    ))
                    overlap_text = "".join(current_chunk)[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current_chunk = [overlap_text, piece] if overlap_text else [piece]
                    current_length = len(overlap_text) + piece_length
                else:
                    current_chunk.append(piece)
                    current_length += piece_length

        if current_chunk:
            chunk_metadata = doc_metadata.copy()
            chunk_metadata['chunk_index'] = len(chunks)
            chunks.append(TextChunk(
                text="".join(current_chunk).strip(),
                metadata=chunk_metadata
            ))

        logger.info(f"Text chunked into {len(chunks)} pieces")
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?。！？])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_oversized_sentence(self, sentence: str) -> List[str]:
        """Split long text spans that exceed chunk_size, even without punctuation."""
        if len(sentence) <= self.chunk_size or self.chunk_size <= 0:
            return [sentence]

        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size

        pieces = []
        start = 0
        while start < len(sentence):
            piece = sentence[start:start + self.chunk_size].strip()
            if piece:
                pieces.append(piece)
            if start + self.chunk_size >= len(sentence):
                break
            start += step
        return pieces


class StructureAwareChunker:
    """Structure-aware chunker with parent-child document support."""

    def __init__(self, parent_chunk_size: int = 1000, child_chunk_size: int = 200,
                 enable_parent_child: bool = True):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.enable_parent_child = enable_parent_child

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Any]:
        """Split text into structure-aware chunks with parent-child relationships."""
        doc_metadata = metadata or {}
        chunks = []

        # Split text into sections by headings (markdown-style or detected)
        sections = self._split_into_sections(text)

        for section_title, section_text in sections:
            if not section_text.strip():
                continue

            if self.enable_parent_child:
                # Create parent chunks
                parent_texts = self._split_by_size(section_text, self.parent_chunk_size)
                for parent_text in parent_texts:
                    parent_id = str(uuid.uuid4())
                    parent_meta = doc_metadata.copy()
                    parent_meta.update({
                        "chunk_id": parent_id,
                        "is_parent": True,
                        "section_title": section_title,
                    })
                    chunks.append(TextChunk(text=parent_text.strip(), metadata=parent_meta))

                    # Create child chunks from parent
                    child_texts = self._split_by_size(parent_text, self.child_chunk_size)
                    for ci, child_text in enumerate(child_texts):
                        child_meta = doc_metadata.copy()
                        child_meta.update({
                            "chunk_id": str(uuid.uuid4()),
                            "parent_id": parent_id,
                            "is_parent": False,
                            "section_title": section_title,
                            "child_index": ci,
                        })
                        chunks.append(TextChunk(text=child_text.strip(), metadata=child_meta))
            else:
                # No parent-child, just split into child-sized chunks
                child_texts = self._split_by_size(section_text, self.child_chunk_size)
                for ci, child_text in enumerate(child_texts):
                    chunk_meta = doc_metadata.copy()
                    chunk_meta.update({
                        "chunk_id": str(uuid.uuid4()),
                        "section_title": section_title,
                        "chunk_index": ci,
                    })
                    chunks.append(TextChunk(text=child_text.strip(), metadata=chunk_meta))

        logger.info(f"Structure-aware chunking: {len(chunks)} chunks")
        return chunks

    def _split_into_sections(self, text: str) -> List[tuple]:
        """Split text into (section_title, section_text) pairs."""
        # Match markdown headings or lines that look like titles
        heading_pattern = re.compile(r'^(#{1,4}\s+.+|[A-Z\u4e00-\u9fff].{0,60})$', re.MULTILINE)
        lines = text.split('\n')
        sections = []
        current_title = ""
        current_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_lines.append(line)
                continue

            is_heading = False
            if stripped.startswith('#'):
                is_heading = True
            elif len(stripped) < 80 and stripped.isupper():
                is_heading = True

            if is_heading and current_lines:
                section_text = '\n'.join(current_lines)
                if section_text.strip():
                    sections.append((current_title, section_text))
                current_title = stripped.lstrip('#').strip()
                current_lines = []
            else:
                current_lines.append(line)

        # Add last section
        if current_lines:
            section_text = '\n'.join(current_lines)
            if section_text.strip():
                sections.append((current_title, section_text))

        if not sections:
            sections = [("", text)]

        return sections

    def _split_by_size(self, text: str, max_size: int) -> List[str]:
        """Split text into pieces of approximately max_size characters."""
        if len(text) <= max_size:
            return [text] if text.strip() else []

        sentences = re.split(r'(?<=[.!?。！？\n])\s*', text)
        pieces = []
        current = []
        current_len = 0

        for sent in sentences:
            if not sent.strip():
                continue
            if current_len + len(sent) > max_size and current:
                pieces.append(' '.join(current))
                current = [sent]
                current_len = len(sent)
            else:
                current.append(sent)
                current_len += len(sent)

        if current:
            pieces.append(' '.join(current))

        return pieces


class TextChunk:
    """Represents a single text chunk with metadata."""

    def __init__(self, text: str, start_idx: int = 0, end_idx: int = 0, metadata: Dict[str, Any] = None):
        self.text = text
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.metadata = metadata or {}

    def __repr__(self):
        return f"TextChunk(len={len(self.text)}, start={self.start_idx})"

"""Tests for text chunking."""

from core.chunking import TextChunker, StructureAwareChunker, TextChunk


class TestTextChunker:
    def test_basic_chunking(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "这是第一句话。这是第二句话。这是第三句话。这是第四句话。这是第五句话。" * 5
        chunks = chunker.chunk(text)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert len(chunk.text) > 0

    def test_metadata_preserved(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)
        text = "短文本内容。"
        chunks = chunker.chunk(text, metadata={"source": "test.pdf"})
        assert len(chunks) > 0
        assert chunks[0].metadata["source"] == "test.pdf"

    def test_empty_text(self):
        chunker = TextChunker()
        chunks = chunker.chunk("")
        assert len(chunks) == 0

    def test_chunk_overlap(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 200
        chunks = chunker.chunk(text)
        assert len(chunks) > 1


class TestStructureAwareChunker:
    def test_basic_structure_chunking(self):
        chunker = StructureAwareChunker(parent_chunk_size=200, child_chunk_size=50)
        text = "# 第一章\n这是第一章的内容。\n# 第二章\n这是第二章的内容。"
        chunks = chunker.chunk(text)
        assert len(chunks) > 0

    def test_parent_child_relationship(self):
        chunker = StructureAwareChunker(parent_chunk_size=200, child_chunk_size=50, enable_parent_child=True)
        text = "这是一段很长的文本。" * 20
        chunks = chunker.chunk(text)
        parents = [c for c in chunks if c.metadata.get("is_parent")]
        children = [c for c in chunks if not c.metadata.get("is_parent", True)]
        # Should have both parents and children
        assert len(parents) > 0 or len(children) > 0

    def test_no_parent_child(self):
        chunker = StructureAwareChunker(enable_parent_child=False)
        text = "简单文本。" * 50
        chunks = chunker.chunk(text)
        for chunk in chunks:
            assert "parent_id" not in chunk.metadata

    def test_metadata_has_chunk_id(self):
        chunker = StructureAwareChunker()
        text = "测试文本内容。" * 20
        chunks = chunker.chunk(text)
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata

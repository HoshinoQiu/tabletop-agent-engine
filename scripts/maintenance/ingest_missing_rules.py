"""
Ingest only rule files that are present in data/rules but missing in vector store.

Useful after dropping in new files without doing a full --clean reindex.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.react_agent import ReActAgent
from core.rag_engine import RAGEngine
import reindex


def main() -> None:
    rag = RAGEngine()
    agent = ReActAgent(rag)

    rules = sorted([p for p in Path("data/rules").iterdir() if p.is_file()])
    source_map = agent.source_game_map
    all_rule_names = [
        reindex.normalize_rule_name(p.stem) or p.stem
        for p in rules
    ]

    added_chunks = 0
    added_files = 0

    for path in rules:
        source = str(path).replace("/", "\\")
        if source in source_map:
            continue

        text = reindex.load_file(path)
        if not text or not reindex.is_text_readable(text):
            print(f"skip: {path.name}")
            continue

        detected_game_name = reindex.detect_game_name(path, text)
        identity = reindex.infer_game_identity(path, detected_game_name, all_rule_names)
        game_name = identity["game_name"]
        chunks = rag.ingest_document(
            text,
            metadata={
                "source": source,
                "document_name": path.name,
                "game_name": game_name,
                "base_game_name": identity["base_game_name"],
                "is_expansion": identity["is_expansion"],
                "expansion_name": identity["expansion_name"],
            },
        )
        print(f"added: {path.name} | game={game_name} | chunks={chunks}")
        added_files += 1
        added_chunks += chunks

    rag.vector_store.save()
    print(f"done: added_files={added_files}, added_chunks={added_chunks}")


if __name__ == "__main__":
    main()

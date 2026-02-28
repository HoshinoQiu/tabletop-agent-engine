"""
检查向量库里存储的内容
"""
import pickle
from pathlib import Path

def main():
    vector_store_path = Path("data/vector_store")

    # 读取 documents.pkl
    documents_path = vector_store_path / "documents.pkl"
    if not documents_path.exists():
        print("❌ 找不到 documents.pkl，请先初始化规则书")
        return

    # 读取 metadatas.pkl
    metadatas_path = vector_store_path / "metadatas.pkl"
    if not metadatas_path.exists():
        print("❌ 找不到 metadatas.pkl")
        return

    with open(documents_path, "rb") as f:
        documents = pickle.load(f)

    with open(metadatas_path, "rb") as f:
        metadatas = pickle.load(f)

    print("=" * 80)
    print(f"📦 向量库检查报告")
    print("=" * 80)
    print(f"\n总共有 {len(documents)} 个文本片段\n")

    # 统计来源
    sources = {}
    for metadata in metadatas:
        source = metadata.get('source', 'unknown')
        if source not in sources:
            sources[source] = 0
        sources[source] += 1

    print("📂 来源统计：")
    for source, count in sources.items():
        print(f"  - {source}: {count} 个片段")
    print()

    # 显示每个来源的前 3 个片段
    for source in sources.keys():
        print("=" * 80)
        print(f"📄 来源: {source}")
        print("=" * 80)

        count = 0
        for i, metadata in enumerate(metadatas):
            if metadata.get('source') == source:
                print(f"\n片段 #{count + 1} (索引 {i}):")
                print(f"内容预览: {documents[i][:150]}...")
                count += 1
                if count >= 3:
                    print(f"\n... (还有 {sources[source] - 3} 个片段)")
                    break

    print("\n" + "=" * 80)
    print("✅ 检查完成")
    print("=" * 80)

if __name__ == "__main__":
    main()

"""
清理向量库，删除所有已存储的规则
"""
import os
from pathlib import Path

def clean_vector_store():
    vector_store_path = Path("data/vector_store")

    if not vector_store_path.exists():
        print("❌ 向量库不存在，无需清理")
        return

    # 列出所有文件
    files = list(vector_store_path.glob("*"))
    if not files:
        print("❌ 向量库为空，无需清理")
        return

    print("=" * 80)
    print(f"🗑️  准备清理向量库")
    print("=" * 80)
    print(f"\n将删除以下文件：")
    for file in files:
        print(f"  - {file.name}")
    print()

    # 确认删除
    response = input("确定要删除这些文件吗？(输入 'yes' 确认): ")
    if response.lower() != 'yes':
        print("❌ 已取消删除")
        return

    # 删除文件
    for file in files:
        file.unlink()

    print("\n✅ 向量库已清理！")
    print("\n下一步：运行以下命令重新初始化规则书")
    print("  python init_rulebook.py --input data/rules/你的规则书.pdf")

if __name__ == "__main__":
    clean_vector_store()

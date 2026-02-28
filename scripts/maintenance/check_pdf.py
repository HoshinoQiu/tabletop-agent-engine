"""快速检查PDF能否提取文本"""
import sys
from pathlib import Path

def check_pdf(pdf_path):
    print(f"\n检查: {pdf_path.name}")
    print("=" * 60)

    # 方法1: PyMuPDF
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()

        print(f"[OK] PyMuPDF: 提取了 {len(text)} 字符")

        if len(text) > 500:
            print(f"[OK] 前200字符预览: {text[:200]}...")
            return True, text
        else:
            print(f"[WARN] 文本过少，可能是扫描版")
            doc.close()
    except ImportError:
        print("[ERROR] PyMuPDF 未安装")

    # 方法2: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t

            print(f"[OK] pdfplumber: 提取了 {len(text)} 字符")

            if len(text) > 500:
                print(f"[OK] 前200字符预览: {text[:200]}...")
                return True, text
            else:
                print(f"[WARN] 文本过少，可能是扫描版")
    except ImportError:
        print("[ERROR] pdfplumber 未安装")

    return False, None

if __name__ == "__main__":
    rules_dir = Path("data/rules")

    for pdf_file in sorted(rules_dir.glob("*.pdf")):
        can_extract, text = check_pdf(pdf_file)

        if can_extract:
            print(f"\n[SUCCESS] {pdf_file.name} 可以直接向量化！")
        else:
            print(f"\n[NEED OCR] {pdf_file.name} 需要OCR处理")

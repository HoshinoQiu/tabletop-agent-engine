# 🔧 问题修复指南

## 问题 1：faiss-cpu 版本错误 ✅ 已修复
**错误信息**：`ERROR: No matching distribution found for faiss-cpu==1.7.4`

**原因**：faiss-cpu 1.7.4 版本已经不存在了

**解决方案**：
我已经把 `requirements.txt` 中的 `faiss-cpu==1.7.4` 改为 `faiss-cpu==1.8.0`

---

## 问题 2：sentence-transformers 安装失败
**错误信息**：`No module named 'sentence_transformers'`

**原因**：因为 faiss-cpu 安装失败，导致后续依赖安装失败

**解决方案**：
```bash
# 重新安装所有依赖
pip install -r requirements.txt
```

**如果还是很慢，用清华镜像**：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 📦 已完成的修复

1. ✅ `requirements.txt`：faiss-cpu 版本从 1.7.4 改为 1.8.0
2. ✅ `config/settings.py`：默认模型改为中文模型 `shibing624/text2vec-base-chinese`

---

## 🚀 现在的操作步骤

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 测试环境（可选）
```bash
python test_setup.py
```

### 3. 初始化规则书（使用中文模型）
```bash
python init_rulebook.py --input QUICKSTART.md
```

### 4. 启动服务
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. 测试 API
打开浏览器访问：http://localhost:8000/docs

---

## 🤔 如果还是遇到问题

### 问题 A：sentence-transformers 下载很慢
**解决**：
```bash
# 只安装 sentence-transformers，用清华镜像
pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 B：模型下载失败（首次运行时）
**解决**：
模型会自动下载，但如果下载失败，可以手动设置代理或镜像：
```python
# 在代码中添加
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

或者在命令行设置：
```bash
set HF_ENDPOINT=https://hf-mirror.com
python init_rulebook.py --input QUICKSTART.md
```

### 问题 C：想换回英文模型
编辑 `config/settings.py`，把：
```python
EMBEDDING_MODEL: str = "shibing624/text2vec-base-chinese"
```
改回：
```python
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
```

---

## 📝 模型说明

### 当前默认：shibing624/text2vec-base-chinese（中文模型）
- ✅ 中文效果好
- ✅ 体积适中（120MB）
- ✅ 适合你的面试场景
- ⚠️ 只支持中文

### 英文备选：sentence-transformers/all-MiniLM-L6-v2
- ✅ 英文效果好
- ✅ 超小体积（80MB）
- ✅ 速度快
- ❌ 中文支持差

### 多语言备选：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- ✅ 中英文都支持
- ✅ 效果好
- ❌ 体积大（471MB）
- ❌ 加载慢

---

## 💡 面试话术

**如果问模型选择：**
> "我选择了 shibing624/text2vec-base-chinese 这个中文嵌入模型，因为：
> 1. 它专门为中文优化，检索准确率高
> 2. 体积适中（120MB），加载速度快
> 3. 适合本地部署，不需要调用外部 API
>
> 在测试中，它对中文规则的理解明显优于通用的多语言模型。"

**如果问为什么不用 GPT 等大模型：**
> "为了展示本地部署的能力，我选择使用本地嵌入模型。这种方式的优势是：
> 1. 数据隐私：不需要把规则书上传到外部服务
> 2. 成本低：没有 API 调用成本
> 3. 可控性：完全自主可控
>
> 当然，在生产环境中，如果需要更强的生成能力，可以结合大模型使用。"

---

## ✅ 验收清单

- [ ] 依赖安装成功
- [ ] test_setup.py 运行通过
- [ ] 规则书初始化成功
- [ ] 服务启动成功
- [ ] API 测试成功（http://localhost:8000/docs）
- [ ] 能说出为什么选择这个模型
- [ ] 能解释 RAG 流程
- [ ] 能解释 ReAct 流程

---

现在试试重新安装依赖！有问题随时告诉我 🚀

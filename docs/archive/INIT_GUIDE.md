# Tabletop-Agent-Engine

## 📋 项目初始化指南

### 1. 环境准备

```bash
# 进入项目目录
cd tabletop-agent-engine

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 初始化规则库

**方法 A：使用示例规则文件**

```bash
# 初始化示例规则库
python init_rulebook.py --input data/rules/example_rules.txt
```

**方法 B：使用自己的规则书**

1. 将你的 PDF 或 TXT 规则书放入 `data/rules/` 目录
2. 初始化规则库

```bash
python init_rulebook.py --input data/rules/your-game-rules.pdf
```

**方法 C：通过 API 上传规则**

```bash
# 启动 API 服务器
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 使用 curl 上传规则
curl -X POST http://localhost:8000/api/upload-rules \
  -F "file=@data/rules/your-game-rules.pdf"
```

### 3. 启动服务

```bash
# 启动 API 服务器
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

服务将在 http://localhost:8000 启动。

### 4. 测试 Agent

**方法 A：使用 curl**

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "我现在有三张手牌，能发动攻击吗？",
    "game_state": {
      "hand_cards": 3,
      "phase": "combat"
    }
  }'
```

**方法 B：使用 Python**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/query",
    json={
        "query": "我现在有三张手牌，能发动攻击吗？",
        "game_state": {
            "hand_cards": 3,
            "phase": "combat"
        }
    }
)

print(response.json())
```

**方法 C：使用浏览器**

访问 http://localhost:8000/docs 查看交互式 API 文档。

### 5. Docker 部署

```bash
# 构建镜像
docker build -t tabletop-agent-engine .

# 运行容器
docker run -p 8000:8000 tabletop-agent-engine

# 或者使用 docker-compose
docker-compose up
```

## 🎯 核心功能演示

### 示例 1：查询攻击规则

**Query:**
```
我现在有三张手牌，能发动攻击吗？
```

**Expected Output:**
```
Thought: 我需要先核实攻击规则
Action: retrieve_rules(keyword="攻击规则")
Observation: Found relevant rules:
[1] Score: 0.8523
Content: 攻击规则:
1. 玩家必须至少有一张手牌才能发动攻击。
2. 攻击需要消耗 1 点生命值。
3. 攻击可以造成 2 点伤害。
4. 防御方可以选择进行格挡，格挡可以减少 1 点伤害。

Final Response: 根据规则，攻击需要至少有一张手牌。你目前有三张手牌，满足条件，可以发动攻击。发动攻击会消耗你 1 点生命值，并对敌方造成 2 点伤害。
```

### 示例 2：查询手牌上限

**Query:**
```
我手牌已经有 7 张了，下一回合该怎么办？
```

**Expected Output:**
```
Thought: 我需要查询手牌上限规则
Action: retrieve_rules(keyword="手牌规则")
Observation: Found relevant rules:
[1] Score: 0.8234
Content: 手牌规则:
1. 每回合开始时，玩家可以从牌库抽 1 张手牌。
2. 手牌数量上限为 7 张。
3. 当手牌达到上限时，必须打出或弃置手牌。

Final Response: 根据手牌规则，你的手牌上限是 7 张。当手牌达到上限时，你必须打出或弃置手牌。建议你下一回合开始时，选择打出一张手牌或弃置多余的手牌，以保持手牌数量在 7 张以内。
```

## 🔧 配置说明

主要配置项在 `config/settings.py` 中：

```python
# RAG 配置
CHUNK_SIZE = 500           # 文本块大小
CHUNK_OVERLAP = 50         # 文本块重叠
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 嵌入模型
VECTOR_STORE_PATH = "data/vector_store"  # 向量存储路径

# Agent 配置
MAX_REACT_ITERATIONS = 10  # 最大 ReAct 迭代次数
TOP_K_RESULTS = 3          # 检索结果数量

# API 配置
API_HOST = "0.0.0.0"
API_PORT = 8000
DEBUG = True
```

## 📊 日志输出

Agent 的思考过程会以详细日志的形式输出：

```
2024-06-18 10:30:00 - __main__ - INFO - ============================================================
2024-06-18 10:30:00 - __main__ - INFO - Processing query: 我现在有三张手牌，能发动攻击吗？
2024-06-18 10:30:00 - __main__ - INFO - Game state: {'hand_cards': 3, 'phase': 'combat'}
2024-06-18 10:30:00 - __main__ - INFO - ============================================================
2024-06-18 10:30:00 - __main__ - INFO -
[Iteration 1]
----------------------------------------
2024-06-18 10:30:00 - __main__ - INFO - Current message:
Thought: 我需要先核实攻击规则
Action: retrieve_rules(keyword="攻击规则")
Observation: Found relevant rules...
Final Response: 根据规则，攻击需要至少有一张手牌...
```

## 🎓 面试展示要点

1. **架构清晰**：RAG + ReAct 模式分离，模块解耦
2. **可扩展性**：易于添加新的工具和规则
3. **可观测性**：详细的日志输出，展示思考过程
4. **生产就绪**：FastAPI 异步接口，Docker 支持
5. **性能优化**：本地向量存储，FAISS 索引

## 📚 技术栈

- **FastAPI**: 现代、快速的 Web 框架
- **FAISS**: Facebook 的向量相似度搜索库
- **SentenceTransformers**: 本地文本嵌入
- **Pydantic**: 数据验证
- **Loguru**: 优雅的日志记录

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

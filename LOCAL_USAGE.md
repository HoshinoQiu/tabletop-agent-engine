# 本地使用说明（规则书入库 + API 调用）

## 1. 新规则书放哪里

- 把文件放到 `data/rules/`
- 支持格式：`.pdf` `.docx` `.txt` `.md` `.markdown`
- 建议文件名直接用桌游名，例如：`卡内基.pdf`、`Carnegie.pdf`

## 2. 重新向量化（推荐全量重建）

在项目根目录执行：

```bash
venv\Scripts\python.exe reindex.py --clean
```

说明：
- `--clean` 会先清空旧向量库，再按 `data/rules/` 全量重建
- 向量库输出到 `data/vector_store/`

如果只想增量添加（不清空旧库）：

```bash
venv\Scripts\python.exe reindex.py
```

## 3. 启动服务

```bash
venv\Scripts\python.exe run.py
```

默认地址：
- `http://localhost:8000`

## 4. 常用 API

### 查看已收录桌游

`GET /api/games`

```bash
curl http://localhost:8000/api/games
```

### 普通模式（一次返回最终答案）

`POST /api/ask`

```bash
curl -X POST http://localhost:8000/api/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"卡内基怎么获胜？\"}"
```

### 流式模式（实时过程）

`POST /api/query/stream`（SSE）

```bash
curl -N -X POST http://localhost:8000/api/query/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Carnegie 的回合流程是什么？\",\"game_state\":{\"hand_cards\":0,\"phase\":\"playing\",\"other_info\":{}}}"
```

## 5. 桌游名匹配机制（已支持模糊）

当前匹配顺序：
1. 直接命中（问题里出现桌游名）
2. 模糊匹配（近似拼写、片段匹配、多词匹配）
3. LLM 兜底映射（跨中英文候选匹配）

如果仍命中不准，优先检查：
- 文件名是否规范（不要留 `rules_v1.2_web` 这类噪声后缀）
- `GET /api/games` 里 `display_name` 是否正确

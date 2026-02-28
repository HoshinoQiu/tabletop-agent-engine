import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pathlib import Path

from config.settings import settings
from core.rag_engine import RAGEngine
from core.session_manager import SessionManager
from core.document_manager import DocumentManager
from agents.react_agent import ReActAgent
from models.schemas import (
    PlayerQuery, AgentResponse, SimpleResponse, SimpleQuery,
    Citation, DocumentInfo, UploadResponse, GameStatus,
)


# Global instances
rag_engine = None
react_agent = None
session_manager = None
document_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine, react_agent, session_manager, document_manager
    import time

    logger.info("=" * 60)
    logger.info("Tabletop Agent Engine starting up")

    t0 = time.time()
    logger.info("Initializing RAG engine (loading embedding model)...")
    rag_engine = RAGEngine()
    logger.info(f"RAG engine ready ({time.time() - t0:.1f}s)")

    t1 = time.time()
    logger.info("Initializing agent...")
    react_agent = ReActAgent(rag_engine=rag_engine)
    logger.info(f"Agent ready ({time.time() - t1:.1f}s)")

    session_manager = SessionManager()
    document_manager = DocumentManager()

    logger.info(f"Startup complete in {time.time() - t0:.1f}s")
    logger.info("=" * 60)

    yield

    logger.info("Tabletop Agent Engine shutting down")


app = FastAPI(
    title="桌游规则问答引擎",
    description="基于 RAG 和 ReAct 模式的桌游规则智能问答系统",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


def _build_citations(raw_citations: list) -> list:
    """Convert raw citation dicts to Citation models."""
    citations = []
    for c in raw_citations:
        if isinstance(c, dict):
            citations.append(Citation(**c))
    return citations


def _get_session_history(session_id: str, query_text: str):
    """Get session and conversation history."""
    if not session_id:
        session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)
    history = session.get_history() if session else []
    return session_id, session, history


def _refresh_game_source_map():
    """Refresh game->source mapping after vector store changes."""
    global react_agent
    if not react_agent:
        return
    try:
        react_agent.game_source_map = react_agent._build_game_source_map()
        react_agent.game_match_cache.clear()
        logger.info(f"Refreshed game source map ({len(react_agent.game_source_map)} games)")
    except Exception as e:
        logger.warning(f"Failed to refresh game source map: {e}")


# ==================== Core Endpoints ====================

@app.get("/")
async def root():
    index_file = Path(__file__).parent.parent / "static" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"status": "ok", "service": "桌游规则问答引擎", "version": "2.0.0"}


@app.get("/api/status")
async def status():
    return {"status": "ok", "service": "桌游规则问答引擎", "version": "2.0.0"}


@app.get("/api/games")
async def list_games():
    if not react_agent:
        return {"count": 0, "games": []}
    games = react_agent.list_indexed_games()
    return {"count": len(games), "games": games}


@app.get("/health")
async def health_check():
    cache_stats = {}
    if rag_engine and rag_engine.cache:
        cache_stats = rag_engine.cache.stats()

    return {
        "status": "healthy",
        "rag_engine": "initialized" if rag_engine else "not initialized",
        "react_agent": "initialized" if react_agent else "not initialized",
        "tools": react_agent.tool_registry.list_tools() if react_agent else [],
        "sessions_active": session_manager.active_count() if session_manager else 0,
        "documents_count": len(document_manager.documents) if document_manager else 0,
        "cache": cache_stats,
        "config": {
            "chunk_size": settings.CHUNK_SIZE,
            "hybrid_search": settings.HYBRID_SEARCH_ENABLED,
            "reranker": settings.RERANKER_ENABLED,
            "cache_enabled": settings.CACHE_ENABLED,
            "faiss_index_type": settings.FAISS_INDEX_TYPE,
            "embedding_backend": settings.EMBEDDING_BACKEND,
            "embedding_model": settings.EMBEDDING_MODEL,
            "ingest_batch_size": settings.INGEST_BATCH_SIZE,
        },
    }


# ==================== Query Endpoints ====================

@app.post("/api/query", response_model=AgentResponse)
async def query_agent(query: PlayerQuery):
    game_state = query.game_state or GameStatus()
    session_id, session, history = _get_session_history(query.session_id, query.query)

    logger.info(f"收到问题: {query.query}")

    try:
        response = react_agent.query(
            query=query.query,
            game_state=game_state.model_dump(),
            conversation_history=history,
        )

        # Update session
        if session:
            session.add_turn("user", query.query)
            session.add_turn("assistant", response["final_response"])

        citations = _build_citations(response.get("citations", []))

        return AgentResponse(
            query=query.query,
            game_state=game_state,
            iterations=response["iterations"],
            thought_chain=response["thought_chain"],
            final_response=response["final_response"],
            success=True,
            citations=citations,
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return AgentResponse(
            query=query.query, game_state=game_state, iterations=0,
            thought_chain=[], final_response=f"Error: {str(e)}",
            success=False, session_id=session_id,
        )


@app.post("/api/query/simple", response_model=SimpleResponse)
async def query_agent_simple(query: PlayerQuery):
    logger.info(f"[简化模式] 收到问题: {query.query}")
    game_state = query.game_state.model_dump() if query.game_state else {
        "hand_cards": 0, "phase": "setup", "other_info": {}
    }
    session_id, session, history = _get_session_history(query.session_id, query.query)

    try:
        response = react_agent.query(
            query=query.query, game_state=game_state, conversation_history=history,
        )
        if session:
            session.add_turn("user", query.query)
            session.add_turn("assistant", response["final_response"])

        citations = _build_citations(response.get("citations", []))
        return SimpleResponse(
            question=query.query, answer=response["final_response"],
            status="success", used_rules_count=len(citations),
            citations=citations, session_id=session_id,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        return SimpleResponse(
            question=query.query, answer=f"抱歉，处理问题时出错了：{str(e)}",
            status="error", session_id=session_id,
        )


@app.post("/api/ask", response_model=SimpleResponse)
async def ask_question(query: SimpleQuery):
    logger.info(f"[超简洁模式] 收到问题: {query.query}")
    session_id, session, history = _get_session_history(query.session_id, query.query)

    try:
        response = react_agent.query(
            query=query.query,
            game_state={"hand_cards": 0, "phase": "setup", "other_info": {}},
            conversation_history=history,
        )
        if session:
            session.add_turn("user", query.query)
            session.add_turn("assistant", response["final_response"])

        citations = _build_citations(response.get("citations", []))
        return SimpleResponse(
            question=query.query, answer=response["final_response"],
            status="success", citations=citations, session_id=session_id,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        return SimpleResponse(
            question=query.query, answer=f"抱歉，处理问题时出错了：{str(e)}",
            status="error", session_id=session_id,
        )


# ==================== SSE Streaming ====================

@app.post("/api/query/stream")
async def query_stream(query: PlayerQuery):
    """SSE streaming endpoint for real-time responses."""
    game_state = query.game_state or GameStatus()
    session_id, session, history = _get_session_history(query.session_id, query.query)

    async def event_generator():
        yield f"data: {json.dumps({'event': 'session', 'session_id': session_id})}\n\n"

        final_text = ""
        async for event in react_agent.query_stream(
            query=query.query,
            game_state=game_state.model_dump(),
            conversation_history=history,
        ):
            event_type = event.get("event", "message")
            data = event.get("data", "")
            if event_type == "answer":
                final_text = data
            yield f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        if session and final_text:
            session.add_turn("user", query.query)
            session.add_turn("assistant", final_text)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ==================== Session Endpoints ====================

@app.post("/api/session/clear")
async def clear_session(session_id: str):
    if session_manager.clear_session(session_id):
        return {"status": "ok", "message": f"Session {session_id} cleared"}
    return {"status": "not_found", "message": f"Session {session_id} not found"}


# ==================== Document Management ====================

def _process_document_background(doc_id: str, file_path: str):
    """Background task to process an uploaded document."""
    import time
    start = time.time()
    try:
        logger.info(f"[{doc_id}] Starting document processing: {file_path}")
        from core.document_parser import PDFStructureParser
        parser = PDFStructureParser()
        text = parser.parse_to_text(file_path)

        # If parse_to_text returned nothing and it's NOT a PDF, try reading as text
        ext = Path(file_path).suffix.lower()
        if (not text or len(text) < 50) and ext in (".txt", ".md", ".markdown"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        # For PDFs, try pdfplumber as a second fallback
        if (not text or len(text) < 50) and ext == ".pdf":
            try:
                import pdfplumber
                pages_text = []
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            pages_text.append(page_text)
                text = "\n".join(pages_text)
                if text:
                    logger.info(f"[{doc_id}] pdfplumber extracted {len(text)} chars")
            except Exception as e:
                logger.warning(f"[{doc_id}] pdfplumber fallback failed: {e}")

        if text and len(text) > 50:
            logger.info(f"[{doc_id}] Extracted {len(text)} chars, starting embedding...")
            chunks_count = rag_engine.ingest_document(
                text,
                metadata={"source": file_path, "document_name": Path(file_path).name},
                batch_size=settings.INGEST_BATCH_SIZE,
            )
            rag_engine.vector_store.save()
            _refresh_game_source_map()
            elapsed = time.time() - start
            document_manager.update_status(doc_id, "ready", chunks_count)
            logger.info(f"[{doc_id}] Done: {chunks_count} chunks in {elapsed:.1f}s")
        else:
            message = (
                "PDF appears to be scanned images (OCR required)"
                if ext == ".pdf"
                else f"insufficient text extracted ({len(text or '')} chars)"
            )
            document_manager.update_status(doc_id, "error", error_message=message)
            if ext == ".pdf":
                logger.error(f"[{doc_id}] Failed: PDF appears to be scanned images (no extractable text). OCR is required.")
            else:
                logger.error(f"[{doc_id}] Failed: insufficient text extracted ({len(text or '')} chars)")
    except Exception as e:
        document_manager.update_status(doc_id, "error", error_message=str(e))
        logger.error(f"[{doc_id}] Processing failed: {e}")
        import traceback
        traceback.print_exc()


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a document for processing. Supports PDF, TXT, and Markdown files."""
    from fastapi import HTTPException
    import re as _re

    filename = file.filename or "unknown"

    # --- Validate file type ---
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}。允许的类型: {', '.join(settings.ALLOWED_FILE_TYPES)}"
        )

    # --- Read and validate file size ---
    file_content = await file.read()
    file_size = len(file_content)
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"文件太大: {file_size / 1024 / 1024:.1f}MB，上限为 {settings.MAX_UPLOAD_SIZE_MB}MB"
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="文件为空")

    # --- Sanitize filename (prevent path traversal) ---
    safe_name = _re.sub(r'[^\w\u4e00-\u9fff.\-]', '_', Path(filename).name)
    if not safe_name or safe_name.startswith('.'):
        safe_name = f"upload_{safe_name}"

    # Save file
    save_path = Path("data/rules") / safe_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(file_content)

    doc_id = document_manager.register_document(safe_name, file_size)
    background_tasks.add_task(_process_document_background, doc_id, str(save_path))

    return UploadResponse(
        doc_id=doc_id, filename=safe_name, status="processing",
        message="文档已上传，正在后台处理中",
    )


@app.get("/api/documents")
async def list_documents():
    return {"documents": document_manager.list_documents()}


@app.get("/api/documents/{doc_id}/status")
async def document_status(doc_id: str):
    doc = document_manager.get_document(doc_id)
    if doc:
        return doc
    return {"error": "Document not found"}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    doc = document_manager.get_document(doc_id)
    if not doc:
        return {"error": "Document not found"}

    # Remove vectors from store
    source_path = str(document_manager.get_file_path(doc_id))
    rag_engine.vector_store.remove_by_source(source_path)
    rag_engine.vector_store.save()
    _refresh_game_source_map()

    # Remove document metadata and file
    document_manager.delete_document(doc_id)
    return {"status": "ok", "message": f"Document {doc_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )

"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class GameStatus(BaseModel):
    """游戏状态信息"""
    hand_cards: int = Field(default=0, description="手牌数量")
    phase: str = Field(default="setup", description="当前游戏阶段")
    other_info: dict = Field(default_factory=dict, description="其他游戏状态信息")


class Citation(BaseModel):
    """引用来源信息"""
    document_name: str = Field(default="", description="文档名称")
    page_number: int = Field(default=0, description="页码")
    section_title: str = Field(default="", description="章节标题")
    chunk_text: str = Field(default="", description="引用文本片段（前150字）")
    relevance_score: float = Field(default=0.0, description="相关性分数")


class PlayerQuery(BaseModel):
    """玩家查询请求"""
    query: str = Field(..., description="你的问题")
    game_state: Optional[GameStatus] = Field(default=None, description="当前游戏状态（可选）")
    session_id: Optional[str] = Field(default=None, description="会话ID（用于多轮对话）")


class SimpleQuery(BaseModel):
    """超简化查询请求"""
    query: str = Field(..., description="你的问题")
    session_id: Optional[str] = Field(default=None, description="会话ID（用于多轮对话）")


class AgentResponse(BaseModel):
    """Agent 响应"""
    query: str = Field(..., description="原始问题")
    game_state: GameStatus = Field(..., description="游戏状态")
    iterations: int = Field(..., description="ReAct 循环次数")
    thought_chain: list = Field(..., description="完整的思考链条")
    final_response: str = Field(..., description="最终答案")
    success: bool = Field(..., description="是否成功回答问题")
    citations: List[Citation] = Field(default_factory=list, description="引用来源")
    session_id: Optional[str] = Field(default=None, description="会话ID")


class SimpleResponse(BaseModel):
    """简化的用户友好响应"""
    question: str = Field(..., description="你的问题")
    answer: str = Field(..., description="AI的回答")
    status: str = Field(default="success", description="请求状态")
    used_rules_count: int = Field(default=0, description="使用的规则片段数量")
    citations: List[Citation] = Field(default_factory=list, description="引用来源")
    session_id: Optional[str] = Field(default=None, description="会话ID")


class DocumentInfo(BaseModel):
    """文档信息"""
    doc_id: str = Field(..., description="文档ID")
    filename: str = Field(..., description="文件名")
    upload_time: str = Field(..., description="上传时间")
    chunk_count: int = Field(default=0, description="文本块数量")
    status: str = Field(default="processing", description="状态: processing/ready/error")
    file_size: int = Field(default=0, description="文件大小（字节）")


class UploadResponse(BaseModel):
    """文档上传响应"""
    doc_id: str = Field(..., description="文档ID")
    filename: str = Field(..., description="文件名")
    status: str = Field(default="processing", description="处理状态")
    message: str = Field(default="", description="消息")

"""
Simple RAG Agent: Retrieve relevant rules, then ask LLM to answer.
Auto-detects game names from indexed documents for precise filtering.
"""

import logging
import json
import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, AsyncGenerator
from collections import defaultdict
from pathlib import Path

from loguru import logger
from config.settings import settings
from agents.tools import ToolRegistry

try:
    from zhipuai import ZhipuAI
except ImportError:
    logger.warning("zhipuai not installed. Please run: pip install zhipuai")
    ZhipuAI = None

try:
    from openai import OpenAI
except ImportError:
    logger.warning("openai not installed. Please run: pip install openai")
    OpenAI = None


class ReActAgent:
    """Simple RAG agent: retrieve rules then generate answer with LLM."""

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.llm_provider = (settings.LLM_PROVIDER or "zhipuai").strip().lower()
        self.game_match_cache: Dict[str, Optional[str]] = {}
        self.source_game_map: Dict[str, str] = {}
        self.source_alias_map: Dict[str, List[str]] = {}

        # Initialize tool registry with real tools
        self.tool_registry = ToolRegistry()
        self._register_tools()

        # Initialize LLM client
        self.client = None
        self._init_llm_client()

        # Build game name -> source file mapping from vector store
        self.game_source_map = self._build_game_source_map()
        logger.info(f"Game source map: {self.game_source_map}")
        logger.info("RAG agent initialized")

    def _init_llm_client(self):
        if self.llm_provider == "openai":
            if not OpenAI:
                logger.warning("OpenAI SDK unavailable; set LLM_PROVIDER=zhipuai or install openai")
                return
            if not settings.OPENAI_API_KEY:
                logger.warning("OPENAI_API_KEY not configured")
                return
            try:
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
            return

        # Default/fallback provider: zhipuai
        self.llm_provider = "zhipuai"
        if not ZhipuAI:
            logger.warning("ZhipuAI SDK unavailable; set LLM_PROVIDER=openai or install zhipuai")
            return
        if not settings.ZHIPU_API_KEY:
            logger.warning("ZHIPU_API_KEY not configured")
            return
        try:
            self.client = ZhipuAI(api_key=settings.ZHIPU_API_KEY)
            logger.info("ZhipuAI client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize ZhipuAI: {e}")

    @staticmethod
    def _extract_message_content(message_content: Any) -> str:
        """Normalize SDK message content into plain text."""
        if isinstance(message_content, str):
            return message_content.strip()
        if isinstance(message_content, list):
            parts = []
            for item in message_content:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if text:
                        parts.append(str(text))
                else:
                    text = getattr(item, "text", "")
                    if text:
                        parts.append(str(text))
            return "\n".join(parts).strip()
        return str(message_content).strip()

    def _register_tools(self):
        """Register agent tools in the ToolRegistry."""
        self.tool_registry.register(
            name="retrieve_rules",
            func=self._tool_retrieve_rules,
            description="从规则库中检索与关键词相关的规则片段",
            parameters={"keyword": "检索关键词"},
        )
        self.tool_registry.register(
            name="list_games",
            func=self._tool_list_games,
            description="列出当前已索引的所有游戏名称",
            parameters={},
        )
        logger.info(f"Registered tools: {self.tool_registry.list_tools()}")

    def _tool_retrieve_rules(self, keyword: str) -> str:
        """Tool: retrieve relevant rules by keyword."""
        context, citations, has_results = self._retrieve_context(keyword)
        if not has_results:
            return "未找到与该关键词相关的规则。"
        result_parts = []
        for i, c in enumerate(citations, 1):
            score = c.get("relevance_score", 0)
            text = c.get("chunk_text", "")
            result_parts.append(f"[{i}] (相关度: {score:.3f}) {text}")
        return "\n".join(result_parts)

    def _tool_list_games(self, _input: str = "") -> str:
        """Tool: list indexed games."""
        games = self.list_indexed_games()
        if not games:
            return "当前没有已索引的游戏。"
        names = [g.get("display_name") or g.get("game_name", "") for g in games]
        return "已索引的游戏: " + ", ".join([n for n in names if n])

    @staticmethod
    def _normalize_game_text(text: str) -> str:
        t = (text or "").lower().strip()
        t = t.replace("_", " ").replace("-", " ")
        t = re.sub(r"\s+", " ", t)
        return t

    @classmethod
    def _compact_game_text(cls, text: str) -> str:
        t = cls._normalize_game_text(text)
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", t)

    @staticmethod
    def _humanize_game_name(name: str) -> str:
        n = (name or "").strip()
        if not n:
            return ""
        if re.search(r"[\u4e00-\u9fff]", n):
            return n
        return " ".join(part.capitalize() for part in n.split(" ") if part)

    @classmethod
    def _is_valid_game_name(cls, name: str) -> bool:
        n = cls._normalize_game_text(name)
        if not n or len(n) < 2:
            return False
        if re.fullmatch(r"[0-9.\s]+", n):
            return False
        if len(n) > 40:
            return False
        if re.search(r"[，。,.:;!?！？]", n):
            return False

        word_count = len([w for w in n.split(" ") if w])
        if re.search(r"[a-zA-Z]", n) and word_count > 5:
            return False

        generic = {
            "rules", "rulebook", "manual", "official", "english", "中文", "中文版",
            "game rules", "board game rules", "unknown",
            "introduction", "how to play", "how to play video", "table of contents",
            "contents", "setup", "scoring", "turn order",
        }
        if n in generic:
            return False
        if n.startswith("by "):
            return False
        alpha_or_cn = re.search(r"[a-zA-Z\u4e00-\u9fff]", n)
        return bool(alpha_or_cn)

    @classmethod
    def _extract_bracket_game_name(cls, text: str) -> str:
        if not text:
            return ""

        head = text[:1500]
        patterns = [
            r"[《](.+?)[》]",
            r"[【](.+?)[】]",
            r"\[([^\[\]\n]{1,30})\]",
        ]
        for p in patterns:
            for m in re.finditer(p, head):
                candidate = cls._normalize_game_text(m.group(1))
                candidate = re.sub(r"\b(rulebook|rules?|manual|official|english|中文版|中文)\b", " ", candidate, flags=re.IGNORECASE)
                candidate = cls._normalize_game_text(candidate)
                if cls._is_valid_game_name(candidate):
                    return candidate
        return ""

    @classmethod
    def _pick_primary_game_name(cls, aliases: List[str]) -> str:
        valid = [cls._normalize_game_text(a) for a in aliases if cls._is_valid_game_name(a)]
        if not valid:
            return ""
        valid = list(dict.fromkeys(valid))
        valid.sort(key=lambda x: (
            0 if re.search(r"[\u4e00-\u9fff]", x) else 1,
            len(x.split()),
            len(x),
        ))
        return valid[0]

    @classmethod
    def _extract_query_fragments(cls, query: str) -> List[str]:
        fragments = []
        normalized = cls._normalize_game_text(query)
        if normalized:
            fragments.append(normalized)

        # Chinese fragments
        for m in re.finditer(r"[\u4e00-\u9fff]{2,}", query or ""):
            f = cls._normalize_game_text(m.group(0))
            if f:
                fragments.append(f)

        # English fragments (single token + 2~4 token window)
        tokens = [t for t in re.findall(r"[a-z0-9]+", normalized) if t]
        for token in tokens:
            fragments.append(token)
        max_size = min(4, len(tokens))
        for size in range(2, max_size + 1):
            for i in range(0, len(tokens) - size + 1):
                fragments.append(" ".join(tokens[i:i + size]))

        deduped = []
        seen = set()
        for f in fragments:
            nf = cls._normalize_game_text(f)
            if nf and nf not in seen:
                seen.add(nf)
                deduped.append(nf)
        return deduped

    def _detect_game_name_fuzzy(self, user_query: str) -> Optional[str]:
        candidates = sorted(
            self.game_source_map.keys(),
            key=lambda x: len(self._compact_game_text(x)),
            reverse=True,
        )
        if not candidates:
            return None

        query_fragments = self._extract_query_fragments(user_query)
        if not query_fragments:
            return None

        best_key = None
        best_score = 0.0

        for game_key in candidates:
            norm_key = self._normalize_game_text(game_key)
            compact_key = self._compact_game_text(game_key)
            if not norm_key or len(compact_key) < 3:
                continue

            key_tokens = [t for t in norm_key.split(" ") if t]

            for frag in query_fragments:
                compact_frag = self._compact_game_text(frag)
                if not compact_frag:
                    continue

                # Deterministic partial match first.
                if len(compact_key) >= 2 and compact_key in compact_frag:
                    logger.info(f"Fuzzy exact-contained match '{user_query}' -> '{game_key}'")
                    return game_key

                # Token inclusion for multi-word English names (e.g. imperial struggle).
                frag_tokens = [t for t in frag.split(" ") if t]
                if len(key_tokens) >= 2 and all(t in frag_tokens for t in key_tokens):
                    logger.info(f"Fuzzy token match '{user_query}' -> '{game_key}'")
                    return game_key

                ratio_norm = SequenceMatcher(None, norm_key, frag).ratio()
                ratio_compact = SequenceMatcher(None, compact_key, compact_frag).ratio()
                score = max(ratio_norm, ratio_compact)

                # Short keys need higher confidence to reduce false positives.
                threshold = 0.78 if len(compact_key) >= 6 else 0.86
                if score >= threshold and score > best_score:
                    best_score = score
                    best_key = game_key

        if best_key:
            logger.info(
                f"Fuzzy ratio match '{user_query}' -> '{best_key}' "
                f"(score={best_score:.2f})"
            )
        return best_key

    @staticmethod
    def _guess_game_name_from_source(source_path: str) -> str:
        name = Path(source_path).name
        name = re.sub(r"\.\w+$", "", name)
        name = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", name)
        name = name.replace("_", " ").replace("-", " ")
        name = re.sub(r"\b(v?\d+(?:\.\d+)+|20\d{2}|19\d{2})\b", " ", name, flags=re.IGNORECASE)
        name = re.sub(r"\b(web|rev|version|edition|ed|en|eng|cn|zh|chs|cht)\b", " ", name, flags=re.IGNORECASE)
        name = re.sub(r"\b[a-z]\b", " ", name, flags=re.IGNORECASE)
        # Remove common non-name suffix/prefix words in rulebook files.
        name = re.sub(
            r"\b(rulebook|rules?|manual|official|english|中文版|中文|v\d+|ver\d+)\b",
            " ",
            name,
            flags=re.IGNORECASE,
        )
        name = re.sub(r"\s+", " ", name).strip()
        return name.lower()

    def _build_game_source_map(self) -> Dict[str, List[str]]:
        """Scan vector store to build game_name -> [source_files] mapping."""
        source_texts = defaultdict(str)
        source_meta_names = defaultdict(set)
        for doc, meta in zip(
            self.rag_engine.vector_store.documents,
            self.rag_engine.vector_store.metadatas,
        ):
            src = meta.get("source", "")
            if len(source_texts[src]) < 2000:
                source_texts[src] += " " + doc
            meta_game = self._normalize_game_text(meta.get("game_name", ""))
            if self._is_valid_game_name(meta_game):
                source_meta_names[src].add(meta_game)
            meta_base = self._normalize_game_text(meta.get("base_game_name", ""))
            if self._is_valid_game_name(meta_base):
                source_meta_names[src].add(meta_base)

            is_expansion = str(meta.get("is_expansion", "")).lower() in {"1", "true", "yes"}
            if is_expansion and self._is_valid_game_name(meta_base):
                source_meta_names[src].add(f"{meta_base} 扩展")

        game_map = {}
        self.source_game_map = {}
        self.source_alias_map = {}

        for src, text in source_texts.items():
            aliases = []

            # Highest confidence: explicit metadata name from indexing stage.
            for meta_name in source_meta_names.get(src, set()):
                if self._is_valid_game_name(meta_name):
                    aliases.append(meta_name)

            guessed = self._guess_game_name_from_source(src)
            guessed = self._normalize_game_text(guessed)
            if self._is_valid_game_name(guessed):
                aliases.append(guessed)

            cn_match = re.search(r'[《](.+?)[》]', text[:500])
            if cn_match:
                cn_name = cn_match.group(1).lower()
                cn_name = self._normalize_game_text(cn_name)
                if self._is_valid_game_name(cn_name):
                    aliases.append(cn_name)

            extracted = self._extract_bracket_game_name(text)
            if self._is_valid_game_name(extracted):
                aliases.append(extracted)

            if not aliases:
                fallback = self._normalize_game_text(Path(src).stem)
                if self._is_valid_game_name(fallback):
                    aliases.append(fallback)

            aliases = list(dict.fromkeys([a for a in aliases if self._is_valid_game_name(a)]))
            primary = self._pick_primary_game_name(aliases)
            self.source_game_map[src] = primary or "unknown"
            self.source_alias_map[src] = aliases

            for alias in aliases:
                game_map.setdefault(alias, [])
                if src not in game_map[alias]:
                    game_map[alias].append(src)

        return game_map

    def _detect_game_name(self, user_query: str) -> Optional[str]:
        lower = self._normalize_game_text(user_query)
        if not lower:
            return None

        cache_key = lower
        if cache_key in self.game_match_cache:
            return self.game_match_cache[cache_key]

        for game_key in sorted(
            self.game_source_map.keys(),
            key=lambda x: len(self._compact_game_text(x)),
            reverse=True,
        ):
            norm_key = self._normalize_game_text(game_key)
            if len(norm_key) >= 2 and norm_key in lower:
                self.game_match_cache[cache_key] = game_key
                return game_key

        fuzzy_key = self._detect_game_name_fuzzy(user_query)
        if fuzzy_key:
            self.game_match_cache[cache_key] = fuzzy_key
            return fuzzy_key

        mapped = self._detect_game_name_with_llm(user_query)
        self.game_match_cache[cache_key] = mapped
        return mapped

    def _detect_game_name_with_llm(self, user_query: str) -> Optional[str]:
        """Use LLM to map user mention to an indexed game key (cross-language)."""
        if not self.client or not self.game_source_map:
            return None

        candidates = sorted(self.game_source_map.keys())
        if not candidates:
            return None

        candidates = candidates[:200]
        candidates_text = "\n".join(f"- {c}" for c in candidates)

        prompt = (
            "你是桌游名称映射器。给定用户问题和候选游戏名列表，"
            "请选出最可能对应的一个候选。"
            "如果都不匹配，返回 NONE。\n"
            "仅输出 JSON：{\"match\":\"候选名或NONE\"}\n\n"
            f"用户问题：{user_query}\n"
            f"候选列表：\n{candidates_text}"
        )

        try:
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,
            )
            raw = self._extract_message_content(response.choices[0].message.content)
            match = self._parse_llm_game_match(raw, candidates)
            if match:
                logger.info(f"LLM mapped game mention '{user_query}' -> '{match}'")
            return match
        except Exception as e:
            logger.warning(f"Game-name LLM mapping failed: {e}")
            return None

    @staticmethod
    def _parse_llm_game_match(raw_text: str, candidates: List[str]) -> Optional[str]:
        if not raw_text:
            return None

        raw_text = raw_text.strip()
        lower_map = {c.lower(): c for c in candidates}
        if raw_text.lower() in {"none", "null", "无", "没有"}:
            return None

        payload = None
        try:
            payload = json.loads(raw_text)
        except Exception:
            m = re.search(r"\{.*\}", raw_text, flags=re.S)
            if m:
                try:
                    payload = json.loads(m.group(0))
                except Exception:
                    payload = None

        if isinstance(payload, dict):
            match_val = str(payload.get("match", "")).strip()
            if match_val.lower() in {"none", "null", "无", "没有"}:
                return None
            if match_val.lower() in lower_map:
                return lower_map[match_val.lower()]

        raw_lower = raw_text.lower()
        for c_low, c in lower_map.items():
            if c_low in raw_lower:
                return c
        return None

    def _get_game_sources(self, game_key: str) -> List[str]:
        return self.game_source_map.get(game_key, [])

    def _build_search_queries(self, user_query: str, game_key: str = "") -> List[str]:
        queries = [user_query]
        g = game_key if game_key else ""
        lower_query = user_query.lower()

        if g:
            queries.append(f"{g} {user_query}")
            queries.append(f"{g} rules setup turn order scoring winning conditions")
            queries.append(f"{g} 规则 玩法 回合 顺序 计分 胜利条件")

        if any(kw in user_query for kw in ["取胜", "胜利", "赢", "获胜"]):
            prefix = g or user_query
            queries.append(f"{prefix} WINNING THE GAME winner victory")
            queries.append(f"{prefix} end game scoring")

        if any(kw in user_query for kw in ["几个人", "人数", "多少人"]):
            prefix = g or user_query
            queries.append(f"{prefix} players number setup")

        if any(kw in user_query for kw in ["出牌", "打牌", "怎么玩"]):
            prefix = g or user_query
            queries.append(f"{prefix} play card turn PLAY")

        if any(kw in user_query for kw in ["规则", "玩法"]):
            prefix = g or user_query
            queries.append(f"{prefix} rules setup PLAY")

        if any(kw in user_query for kw in ["得分", "计分", "分数"]):
            prefix = g or user_query
            queries.append(f"{prefix} SCORING points")

        if any(kw in user_query for kw in ["几轮", "回合", "轮数"]):
            prefix = g or user_query
            queries.append(f"{prefix} rounds turns game round")

        # If user asks in Chinese, add English retrieval intent; if in English, add Chinese intent.
        if re.search(r"[\u4e00-\u9fff]", user_query):
            queries.append(f"{g or user_query} rules gameplay turn setup")
        if re.search(r"[a-zA-Z]", user_query):
            queries.append(f"{g or user_query} 规则 玩法 回合 设置")

        if "rule" in lower_query or "规则" in user_query:
            queries.append(f"{g or user_query} basic rules setup")
            queries.append(f"{g or user_query} 基础规则 开局")

        deduped = []
        seen = set()
        for q in queries:
            nq = self._normalize_game_text(q)
            if nq and nq not in seen:
                seen.add(nq)
                deduped.append(q)

        return deduped[:4]

    def list_indexed_games(self) -> List[Dict[str, Any]]:
        """Return indexed games for UI display."""
        grouped = {}
        for src, primary in self.source_game_map.items():
            if not self._is_valid_game_name(primary):
                continue
            key = self._normalize_game_text(primary)
            grouped.setdefault(key, {"sources": set(), "aliases": set()})
            grouped[key]["sources"].add(Path(src).name)
            for alias in self.source_alias_map.get(src, []):
                if self._is_valid_game_name(alias):
                    grouped[key]["aliases"].add(self._humanize_game_name(alias))

        items = []
        for key, val in grouped.items():
            sources = sorted(val["sources"])
            aliases = sorted({a for a in val["aliases"] if a})
            items.append({
                "game_name": key,
                "display_name": self._humanize_game_name(key),
                "aliases": aliases,
                "document_count": len(sources),
                "documents": sources,
            })

        items.sort(key=lambda x: (-x["document_count"], x["display_name"]))
        return items

    def _retrieve_context(self, user_query: str) -> tuple:
        """Retrieve relevant rule chunks using registered retrieve tool."""
        game_key = self._detect_game_name(user_query)
        game_sources = self._get_game_sources(game_key) if game_key else []
        search_queries = self._build_search_queries(user_query, game_key or "")

        logger.info(f"Detected game: {game_key}, sources: {game_sources}")
        logger.info(f"Search queries: {search_queries}")

        all_results = []
        seen_texts = set()

        for sq in search_queries:
            query_top_k = settings.TOP_K_RESULTS * (10 if game_sources else 2)
            results, _ = self.rag_engine.retrieve(
                sq, top_k=query_top_k, min_score=settings.MIN_SIMILARITY_SCORE
            )
            for r in results:
                text = r.get("document", "")
                text_key = text[:100]
                if text_key not in seen_texts and self._is_readable(text):
                    seen_texts.add(text_key)
                    all_results.append(r)

        if game_sources:
            filtered = [r for r in all_results
                        if r.get("metadata", {}).get("source", "") in game_sources]
            if filtered:
                all_results = filtered
                logger.info(f"Filtered to {len(all_results)} results from {game_sources}")
            else:
                # Keep answer grounded: if game is known but recall misses, use same-source fallback chunks.
                all_results = self._source_fallback_results(
                    game_sources, limit=settings.TOP_K_RESULTS * 3
                )
                if all_results:
                    logger.info(
                        f"No direct filtered hits; fallback to {len(all_results)} chunks "
                        f"from known sources {game_sources}"
                    )

        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_results = all_results[:settings.TOP_K_RESULTS]

        if not top_results:
            return "", [], False

        context_parts = []
        citations = []
        for i, r in enumerate(top_results, 1):
            context_parts.append(f"[规则片段{i}]\n{r['document']}")
            if "citation" in r:
                citations.append(r["citation"])

        return "\n\n".join(context_parts), citations, True

    def _source_fallback_results(self, game_sources: List[str], limit: int = 15) -> List[Dict[str, Any]]:
        fallback = []
        seen = set()
        for doc, meta in zip(
            self.rag_engine.vector_store.documents,
            self.rag_engine.vector_store.metadatas,
        ):
            source = str(meta.get("source", ""))
            if source not in game_sources:
                continue
            if not self._is_readable(doc):
                continue

            key = doc[:120]
            if key in seen:
                continue
            seen.add(key)

            fallback.append({
                "document": doc,
                "score": 0.0,
                "metadata": meta,
                "citation": {
                    "document_name": meta.get("source", meta.get("document_name", "unknown")),
                    "page_number": meta.get("page_number", 0),
                    "section_title": meta.get("section_title", ""),
                    "chunk_text": doc[:150],
                    "relevance_score": 0.0,
                },
            })
            if len(fallback) >= limit:
                break
        return fallback

    def _call_llm(self, user_query: str, context: str,
                  conversation_history: List[Dict] = None) -> str:
        """Call LLM with context to generate answer."""
        if not self.client:
            return "错误：LLM 客户端未初始化。"

        history_text = ""
        if conversation_history:
            for turn in conversation_history[-settings.MAX_HISTORY_TURNS * 2:]:
                role = "用户" if turn["role"] == "user" else "助手"
                history_text += f"{role}: {turn['content'][:200]}\n"

        # Include available tools in system prompt
        tools_desc = self.tool_registry.get_tools_prompt()

        if context:
            system_msg = f"""你是一个桌游规则专家助手。根据下面提供的规则内容回答用户的问题。

可用工具：
{tools_desc}

规则：
{context}

要求：
1. 用中文回答
2. 只根据提供的规则内容回答，不要编造
3. 如果规则内容是英文，翻译成中文后回答
4. 回答要简洁明了，直接回答问题
5. 如果提供的规则内容与问题无关，告知用户未找到相关规则"""
        else:
            system_msg = f"你是一个桌游规则专家助手。用中文回答。\n\n可用工具：\n{tools_desc}"

        messages = [{"role": "system", "content": system_msg}]
        if history_text:
            messages.append({"role": "system", "content": f"对话历史:\n{history_text}"})
        messages.append({"role": "user", "content": user_query})

        try:
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=2000,
            )
            content = response.choices[0].message.content
            answer = self._extract_message_content(content)
            return answer or "抱歉，模型没有返回可用内容。"
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"LLM 调用失败：{str(e)}"

    def query(self, query: str, game_state: Optional[Dict[str, Any]] = None,
              conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Process a user query: retrieve context via tools, then generate answer."""
        logger.info(f"Processing query: {query}")

        # Use retrieve_rules tool through the registry
        context, citations, has_results = self._retrieve_context(query)
        logger.info(f"Retrieved context: {len(context)} chars, {len(citations)} citations")

        if has_results:
            answer = self._call_llm(query, context, conversation_history)
        else:
            answer = self._call_llm(query, "", conversation_history)
            if not answer or "错误" in answer:
                answer = "抱歉，规则库中没有找到与您问题相关的规则。请确认游戏名称是否正确。"

        return {
            "query": query,
            "game_state": game_state,
            "iterations": 1,
            "thought_chain": [],
            "final_response": answer,
            "citations": citations,
        }

    async def query_stream(self, query: str, game_state: Optional[Dict[str, Any]] = None,
                           conversation_history: List[Dict] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming version — yields SSE events."""
        logger.info(f"[Stream] Processing query: {query}")

        yield {"event": "thought", "data": "正在检索相关规则..."}
        context, citations, has_results = self._retrieve_context(query)

        if has_results:
            yield {"event": "observation", "data": f"找到 {len(citations)} 条相关规则"}
        else:
            yield {"event": "observation", "data": "未找到直接相关的规则"}

        yield {"event": "thought", "data": "正在生成回答..."}
        if has_results:
            answer = self._call_llm(query, context, conversation_history)
        else:
            answer = "抱歉，规则库中没有找到与您问题相关的规则。请确认游戏名称是否正确。"

        yield {"event": "answer", "data": answer}

        for c in citations:
            yield {"event": "citation", "data": c}

        yield {"event": "done", "data": ""}

    @staticmethod
    def _is_readable(text: str) -> bool:
        if not text or len(text) < 10:
            return False
        readable = sum(1 for c in text[:200] if '\u4e00' <= c <= '\u9fff' or c.isascii())
        return readable / min(len(text), 200) > 0.5

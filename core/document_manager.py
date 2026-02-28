"""
Document Manager: Manages document lifecycle (upload, index, delete).
"""

import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from config.settings import settings


from loguru import logger


class DocumentManager:
    """Manages document metadata and lifecycle."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.rules_dir = self.data_dir / "rules"
        self.meta_path = self.data_dir / "documents_meta.json"
        self.rules_dir.mkdir(parents=True, exist_ok=True)
        self.documents: Dict[str, dict] = {}
        self._load_meta()

    def _load_meta(self):
        """Load document metadata from JSON file."""
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded metadata for {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Failed to load document metadata: {e}")
                self.documents = {}

    def _save_meta(self):
        """Save document metadata to JSON file."""
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save document metadata: {e}")

    def register_document(self, filename: str, file_size: int) -> str:
        """Register a new document and return its ID."""
        doc_id = str(uuid.uuid4())[:8]
        self.documents[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "upload_time": datetime.now().isoformat(),
            "chunk_count": 0,
            "status": "processing",
            "file_size": file_size,
            "error_message": "",
        }
        self._save_meta()
        logger.info(f"Registered document: {filename} (ID: {doc_id})")
        return doc_id

    def update_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: int = 0,
        error_message: str = "",
    ):
        """Update document processing status."""
        if doc_id in self.documents:
            self.documents[doc_id]["status"] = status
            if chunk_count:
                self.documents[doc_id]["chunk_count"] = chunk_count
            self.documents[doc_id]["error_message"] = error_message or ""
            self._save_meta()

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Get document metadata by ID."""
        return self.documents.get(doc_id)

    def list_documents(self) -> List[dict]:
        """List all documents."""
        return list(self.documents.values())

    def delete_document(self, doc_id: str) -> bool:
        """Delete document metadata and file."""
        if doc_id not in self.documents:
            return False

        doc = self.documents[doc_id]
        filename = doc["filename"]

        # Remove file
        file_path = self.rules_dir / filename
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")

        # Remove metadata
        del self.documents[doc_id]
        self._save_meta()
        logger.info(f"Deleted document: {filename} (ID: {doc_id})")
        return True

    def get_file_path(self, doc_id: str) -> Optional[Path]:
        """Get the file path for a document."""
        doc = self.documents.get(doc_id)
        if doc:
            return self.rules_dir / doc["filename"]
        return None

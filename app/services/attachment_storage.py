from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.schemas import AttachmentRef, new_id


DEFAULT_ATTACHMENT_DIR = Path("storage/attachments")
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024


class AttachmentStorage:
    """Stores uploaded chat attachments for local multimodal validation."""

    def __init__(self, root_dir: Path | str = DEFAULT_ATTACHMENT_DIR):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, file: UploadFile) -> AttachmentRef:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="附件为空，无法上传。")
        if len(content) > MAX_ATTACHMENT_BYTES:
            raise HTTPException(status_code=413, detail="附件超过 20MB 上限。")

        attachment_id = new_id("att")
        original_name = Path(file.filename or "attachment").name
        suffix = Path(original_name).suffix.lower()
        digest = hashlib.sha256(content).hexdigest()
        stored_name = f"{attachment_id}{suffix}"
        target = self.root_dir / stored_name
        target.write_bytes(content)

        file_type = self._infer_file_type(original_name, file.content_type)
        return AttachmentRef(
            attachment_id=attachment_id,
            file_type=file_type,
            file_ref=stored_name,
            original_name=original_name,
            mime_type=file.content_type,
            file_size=len(content),
            sha256=digest,
            storage_path=str(target),
            download_url=f"/attachments/{attachment_id}/download",
        )

    def resolve(self, attachment: AttachmentRef) -> Path | None:
        candidates = [
            Path(attachment.storage_path) if attachment.storage_path else None,
            self.root_dir / attachment.file_ref if attachment.file_ref else None,
        ]
        for candidate in candidates:
            if candidate and candidate.exists() and candidate.is_file():
                return candidate
        return None

    def resolve_by_id(self, attachment_id: str) -> Path | None:
        for path in self.root_dir.glob(f"{attachment_id}*"):
            if path.is_file():
                return path
        return None

    @staticmethod
    def _infer_file_type(filename: str, mime_type: str | None) -> str:
        name = filename.lower()
        mime = (mime_type or "").lower()
        if mime.startswith("image/"):
            return "image"
        if mime == "application/pdf" or name.endswith(".pdf"):
            return "pdf"
        if name.endswith((".xls", ".xlsx")):
            return "excel"
        if name.endswith((".doc", ".docx")):
            return "word"
        return "other"

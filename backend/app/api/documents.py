"""Documents API: 上传、解析、切分、向量化、入库。"""
from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from loguru import logger

from app.api.deps import get_current_user
from app.core.kb_registry import ensure_kb_for_read, ensure_kb_for_write
from app.rag.chunker import TextChunker
from app.rag.parser import DocumentParser
from app.rag.retriever import HybridRetriever
from app.schemas.document import DocumentUploadResponse

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

UPLOAD_DIR = Path("./data/uploads")


def _ensure_upload_dir() -> Path:
    """延迟到每次请求都确保上传目录存在, 兼容测试时 chdir 到临时目录."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


@router.post("/upload", response_model=DocumentUploadResponse, summary="上传文档并自动入库")
async def upload_document(
    file: UploadFile = File(...),
    kb_id: str = Query("default", description="目标知识库 ID, 必须属于当前用户 (私有 kb 自动创建并归属)"),
    kb_id_form: str | None = Form(default=None, alias="kb_id"),
    user: str = Depends(get_current_user),
):
    # 兼容: 上传时 kb_id 既可以在 query string 也可以在 multipart/form 中;
    # form 优先 (前端两个接口都用得上).
    resolved_kb_id = kb_id_form if kb_id_form is not None else kb_id
    kb_id = resolved_kb_id
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件名")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in DocumentParser.SUPPORTED:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {suffix}")

    # 0) ACL: 若 kb 不存在则自动创建并归属当前用户; 若存在则必须是当前用户的 kb.
    #    公共 kb (settings.public_kb_ids, 默认 "default") 不允许写入.
    ensure_kb_for_write(kb_id, user)

    # 1) 保存到本地
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    upload_dir = _ensure_upload_dir()
    dest = upload_dir / f"{doc_id}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) 解析 + 切分 + 入库 (CPU 密集: 解析+切分+embedding+向量/BM25 写盘)
    try:
        text = await asyncio.to_thread(DocumentParser().parse, str(dest))
        if not text.strip():
            raise HTTPException(status_code=400, detail="文档内容为空或解析失败")
        chunks = await asyncio.to_thread(TextChunker().split, text, doc_id=doc_id)
        if not chunks:
            raise HTTPException(status_code=400, detail="切片为空")
        retriever = HybridRetriever(kb_id=kb_id)
        ids = [c.metadata["chunk_id"] for c in chunks]
        docs = [c.text for c in chunks]
        metas = [c.metadata | {"filename": file.filename, "user": user} for c in chunks]
        await retriever.add_documents_async(ids=ids, documents=docs, metadatas=metas)
        logger.info(f"Indexed {file.filename} -> {len(chunks)} chunks (doc_id={doc_id})")
        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            chunks=len(chunks),
            status="indexed",
            message=f"成功入库 {len(chunks)} 个 chunk",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"入库失败: {e}")


@router.get("/list", summary="列出当前知识库 chunk 数")
async def list_documents(kb_id: str = "default", user: str = Depends(get_current_user)):
    # ACL: 公共 kb 直接放行; 私有 kb 必须 owner == 当前用户
    ensure_kb_for_read(kb_id, user)
    retriever = HybridRetriever(kb_id=kb_id)
    return {
        "kb_id": kb_id,
        "bm25_count": retriever.bm25_store.count(),
        "vector_count": retriever.vector_store.count(),
    }

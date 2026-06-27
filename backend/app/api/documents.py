"""Documents API: 上传、解析、切分、向量化、入库。"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from loguru import logger

from app.api.deps import get_current_user
from app.rag.chunker import TextChunker
from app.rag.parser import DocumentParser
from app.rag.retriever import HybridRetriever
from app.schemas.document import DocumentUploadResponse

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=DocumentUploadResponse, summary="上传文档并自动入库")
async def upload_document(
    file: UploadFile = File(...),
    kb_id: str = "default",
    user: str = Depends(get_current_user),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件名")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in DocumentParser.SUPPORTED:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {suffix}")

    # 1) 保存到本地
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    dest = UPLOAD_DIR / f"{doc_id}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) 解析 + 切分 + 入库
    try:
        text = DocumentParser().parse(str(dest))
        if not text.strip():
            raise HTTPException(status_code=400, detail="文档内容为空或解析失败")
        chunks = TextChunker().split(text, doc_id=doc_id)
        if not chunks:
            raise HTTPException(status_code=400, detail="切片为空")
        retriever = HybridRetriever(kb_id=kb_id)
        ids = [c.metadata["chunk_id"] for c in chunks]
        docs = [c.text for c in chunks]
        metas = [c.metadata | {"filename": file.filename, "user": user} for c in chunks]
        retriever.add_documents(ids=ids, documents=docs, metadatas=metas)
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
    retriever = HybridRetriever(kb_id=kb_id)
    return {
        "kb_id": kb_id,
        "bm25_count": retriever.bm25_store.count(),
        "vector_count": retriever.vector_store.count(),
    }

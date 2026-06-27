def test_imports():
    # 只测不需要 email_validator 的核心模块
    from app.rag.chunker import TextChunker
    from app.rag.retriever import rrf_fuse
    from app.rag.bm25_store import BM25Store
    from app.rag.reranker import CrossEncoderReranker
    from app.agents.llm_router import LLMRouter
    from app.mcp.server import build_mcp_server
    assert TextChunker is not None
    assert rrf_fuse is not None
    assert BM25Store is not None
    assert CrossEncoderReranker is not None
    assert LLMRouter is not None


def test_app_title():
    # app.main 会触发 EmailStr 导入，需要 email-validator
    # 这个用例单独跑（用 pip install pydantic[email]）
    from app.main import app
    assert app.title.startswith("UniKB")

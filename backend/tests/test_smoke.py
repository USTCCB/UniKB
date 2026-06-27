def test_imports():
    from app.main import app
    from app.rag.chunker import TextChunker
    from app.rag.retriever import rrf_fuse
    from app.agents.graph import build_agent_graph
    from app.agents.llm_router import LLMRouter
    from app.mcp.server import build_mcp_server
    assert app.title.startswith("UniKB")
    assert TextChunker is not None
    assert rrf_fuse is not None
    assert LLMRouter is not None

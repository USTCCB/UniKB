from app.rag.chunker import TextChunker

def test_split_short():
    ch = TextChunker(chunk_size=50, chunk_overlap=10)
    out = ch.split("你好世界。这是一个测试。", doc_id="d1")
    assert out, "should produce chunks"
    assert all(c.metadata["doc_id"] == "d1" for c in out)
    assert all("chunk_id" in c.metadata for c in out)

def test_split_long():
    text = "句子。" * 200
    ch = TextChunker(chunk_size=100, chunk_overlap=20)
    out = ch.split(text, doc_id="d2")
    assert len(out) > 1
    for c in out:
        assert len(c.text) <= 200

def test_split_empty():
    ch = TextChunker()
    assert ch.split("", doc_id="x") == []

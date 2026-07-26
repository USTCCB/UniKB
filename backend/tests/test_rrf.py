from app.rag.retriever import rrf_fuse

def test_rrf_basic():
    a = [{"id": "1", "document": "A"}, {"id": "2", "document": "B"}]
    b = [{"id": "2", "document": "B"}, {"id": "3", "document": "C"}]
    fused = rrf_fuse([a, b], k=60)
    ids = [f["id"] for f in fused]
    assert ids[0] == "2"
    assert set(ids) == {"1", "2", "3"}

def test_rrf_empty():
    assert rrf_fuse([[], []]) == []

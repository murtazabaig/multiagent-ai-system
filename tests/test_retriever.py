def test_faiss_retriever():
    from src.retriever.faiss_retriever import FAISSRetriever
    docs = [
        "France is a country in Europe. Its capital is Paris.",
        "Paris is the largest city in France and its capital.",
        "London is the capital of the UK."
    ]
    retriever = FAISSRetriever(docs)
    result = retriever.retrieve("capital of France", top_k=2)
    assert "documents" in result
    assert len(result["documents"]) == 2

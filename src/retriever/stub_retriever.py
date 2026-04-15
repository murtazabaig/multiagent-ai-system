class StubRetriever:
    def retrieve(self, subtask: str) -> dict:
        """Stub retriever that returns mock documents."""
        return {
            "documents": ["Doc1", "Doc2"],
            "scores": [0.9, 0.8]
        }

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class FAISSRetriever:
    def __init__(self, documents: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.embeddings = self.model.encode(documents, show_progress_bar=False)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings, dtype=np.float32))

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        query_emb = self.model.encode([query])
        D, I = self.index.search(np.array(query_emb, dtype=np.float32), top_k)
        results = [self.documents[i] for i in I[0]]
        scores = [float(d) for d in D[0]]
        return {"documents": results, "scores": scores}

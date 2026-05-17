from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

class FAISSRetriever:
    def __init__(self, documents: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", index_path: str = "data/faiss_index"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.index_path = index_path
        
        # Ensure directories exist
        os.makedirs(index_path, exist_ok=True)
        self.faiss_file = os.path.join(index_path, "index.faiss")
        self.docs_file = os.path.join(index_path, "docs.pkl")

        if os.path.exists(self.faiss_file) and os.path.exists(self.docs_file):
            print("Loading FAISS index from disk...")
            self.index = faiss.read_index(self.faiss_file)
            with open(self.docs_file, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"Loaded {len(self.documents)} documents from index.")
        else:
            print(f"Building new FAISS index for {len(documents)} documents. This may take a while...")
            # We show a progress bar for the encoding process
            self.embeddings = self.model.encode(documents, show_progress_bar=True)
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(np.array(self.embeddings, dtype=np.float32))
            
            # Save to disk
            faiss.write_index(self.index, self.faiss_file)
            with open(self.docs_file, 'wb') as f:
                pickle.dump(self.documents, f)
            print("Saved FAISS index to disk.")

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        query_emb = self.model.encode([query])
        D, I = self.index.search(np.array(query_emb, dtype=np.float32), top_k)
        results = [self.documents[i] for i in I[0]]
        scores = [float(d) for d in D[0]]
        return {"documents": results, "scores": scores}

    def add_documents(self, new_docs: List[str]):
        """Dynamically add new documents to the existing FAISS index."""
        print(f"Encoding {len(new_docs)} new documents...")
        new_embeddings = self.model.encode(new_docs, show_progress_bar=False)
        self.index.add(np.array(new_embeddings, dtype=np.float32))
        self.documents.extend(new_docs)
        
        # Save updated index to disk
        faiss.write_index(self.index, self.faiss_file)
        with open(self.docs_file, 'wb') as f:
            pickle.dump(self.documents, f)
        print("Updated FAISS index saved to disk.")

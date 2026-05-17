from src.planner.mvp_planner import validate_planner_output
from src.planner.llm_planner import LLMPlanner
from src.retriever.faiss_retriever import FAISSRetriever
from src.executor.llm_executor import LLMExecutor
from src.data.dataset_loader import load_dataset
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import set_seed
import yaml
from pathlib import Path
import json

class Pipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        setup_logging()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        set_seed(self.config.get('seed', 42))
        
        self.planner = LLMPlanner()
        
        # Load processed dataset for retriever
        dataset_path = self.config.get('dataset_path', 'data/processed/squad_contexts.json')
        if Path(dataset_path).exists():
            print(f"Loading dataset from {dataset_path}...")
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            if not self.documents:
                self.documents = ["Sample document 1", "Sample document 2"]
        else:
            self.documents = ["Sample document 1", "Sample document 2"]
            
        self.retriever = FAISSRetriever(self.documents, self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'))
        self.executor = LLMExecutor()
        self.top_k = self.config.get('top_k', 5)

    def run(self, query: str):
        plan = self.planner.plan(query)
        assert validate_planner_output(plan), "Planner output schema invalid!"
        results = []
        retrieved_context = ""
        reasoning_output = ""
        
        for subtask in plan["subtasks"]:
            if subtask["type"] == "retrieve":
                docs = self.retriever.retrieve(subtask["description"], top_k=self.top_k)
                if "documents" in docs and docs["documents"]:
                    retrieved_context = " ".join(docs["documents"])
                results.append(docs)
            elif subtask["type"] == "reason":
                reasoning = self.executor.reason(subtask["description"], retrieved_context)
                reasoning_output = reasoning
                results.append({"reasoning": reasoning})
            elif subtask["type"] == "execute":
                combined_context = f"Knowledge: {retrieved_context}\nReasoning: {reasoning_output}"
                exec_result = self.executor.execute(query, combined_context)
                results.append(exec_result)
        return {"plan": plan, "results": results}

    def add_documents(self, docs: list):
        """Pass new documents down to the retriever to update the vector index."""
        self.retriever.add_documents(docs)

if __name__ == "__main__":
    pipeline = Pipeline()
    query = "What is the capital of France? Explain why."
    result = pipeline.run(query)
    print(result)

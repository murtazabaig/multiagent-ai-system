from src.planner.mvp_planner import MVPPlanner, validate_planner_output
from src.retriever.faiss_retriever import FAISSRetriever
from src.executor.stub_executor import StubExecutor
from src.data.dataset_loader import load_dataset
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import set_seed
import yaml
from pathlib import Path

class Pipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        setup_logging()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        set_seed(self.config.get('seed', 42))
        self.planner = MVPPlanner()
        # Load processed dataset for retriever
        dataset_path = self.config.get('dataset_path', 'data/processed/sample.json')
        if Path(dataset_path).exists():
            dataset = load_dataset(dataset_path)
            # Assume context_docs is a list of documents
            self.documents = []
            for item in dataset:
                self.documents.extend(item.get('context_docs', []))
            if not self.documents:
                self.documents = ["Sample document 1", "Sample document 2"]
        else:
            self.documents = ["Sample document 1", "Sample document 2"]
        self.retriever = FAISSRetriever(self.documents, self.config.get('embedding_model'))
        self.executor = StubExecutor()
        self.top_k = self.config.get('top_k', 5)

    def run(self, query: str):
        plan = self.planner.plan(query)
        assert validate_planner_output(plan), "Planner output schema invalid!"
        results = []
        for subtask in plan["subtasks"]:
            if subtask["type"] == "retrieve":
                docs = self.retriever.retrieve(subtask["description"], top_k=self.top_k)
                results.append(docs)
            elif subtask["type"] == "reason":
                # Placeholder for reasoning
                results.append({"reasoning": "Reasoned over context"})
            elif subtask["type"] == "execute":
                exec_result = self.executor.execute(subtask["description"], "context")
                results.append(exec_result)
        return {"plan": plan, "results": results}

if __name__ == "__main__":
    pipeline = Pipeline()
    query = "What is the capital of France? Explain why."
    result = pipeline.run(query)
    print(result)

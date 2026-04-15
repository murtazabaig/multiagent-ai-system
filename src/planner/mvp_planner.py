from typing import Dict, Any, List
import re

class MVPPlanner:
    """
    A simple rule/template-based planner for decomposing queries into sub-tasks.
    """
    def plan(self, query: str) -> Dict[str, Any]:
        # Example: If the query contains 'find', 'retrieve', 'explain', etc., create subtasks accordingly
        subtasks = []
        if re.search(r"find|retrieve|search", query, re.IGNORECASE):
            subtasks.append({"id": 1, "description": "Retrieve relevant documents", "type": "retrieve"})
        if re.search(r"explain|why|how|reason", query, re.IGNORECASE):
            subtasks.append({"id": 2, "description": "Reason over retrieved context", "type": "reason"})
        subtasks.append({"id": len(subtasks)+1, "description": "Generate final answer", "type": "execute"})
        return {
            "goal": query,
            "subtasks": subtasks,
            "required_tools": [],
            "notes": "MVP planner output"
        }

# Schema validation utility
from pydantic import BaseModel, ValidationError

class Subtask(BaseModel):
    id: int
    description: str
    type: str

class PlannerOutput(BaseModel):
    goal: str
    subtasks: List[Subtask]
    required_tools: List[str]
    notes: str

def validate_planner_output(output: dict) -> bool:
    try:
        PlannerOutput(**output)
        return True
    except ValidationError as e:
        print(f"Planner output validation error: {e}")
        return False

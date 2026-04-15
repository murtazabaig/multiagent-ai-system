class StubPlanner:
    def plan(self, query: str) -> dict:
        """Stub planner that returns a fixed plan."""
        return {
            "goal": query,
            "subtasks": [
                {"id": 1, "description": "Retrieve context", "type": "retrieve"},
                {"id": 2, "description": "Reason over context", "type": "reason"},
                {"id": 3, "description": "Execute answer", "type": "execute"}
            ],
            "required_tools": [],
            "notes": "Stub output"
        }

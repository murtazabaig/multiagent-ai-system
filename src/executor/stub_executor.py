class StubExecutor:
    def execute(self, action: str, context: str) -> dict:
        """Stub executor that returns a mock result."""
        return {"result": f"Executed {action} with context: {context}"}

# API Contracts

## Planner
Input: {"query": "..."}
Output: {"goal": "...", "subtasks": [...], "required_tools": [], "notes": ""}

## Retriever
Input: {"subtask": "..."}
Output: {"documents": [...], "scores": [...]} 

## Executor
Input: {"action": "...", "context": "..."}
Output: {"result": "..."}

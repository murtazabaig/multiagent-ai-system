def test_mvp_planner():
    from src.planner.mvp_planner import MVPPlanner, validate_planner_output
    planner = MVPPlanner()
    query = "Find the capital of France and explain why."
    plan = planner.plan(query)
    assert validate_planner_output(plan)
    assert plan["goal"] == query
    assert any(s["type"] == "retrieve" for s in plan["subtasks"])
    assert any(s["type"] == "reason" for s in plan["subtasks"])
    assert any(s["type"] == "execute" for s in plan["subtasks"])

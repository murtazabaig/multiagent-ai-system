def test_stub_pipeline():
    from src.core.pipeline import Pipeline
    pipeline = Pipeline()
    result = pipeline.run("What is the capital of France?")
    assert "plan" in result
    assert "results" in result

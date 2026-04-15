"""
Main entrypoint for Multi-Agent AI System (Planner + Retriever + Executor)
Run this file to test the end-to-end pipeline.
"""
import argparse
from src.core.pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent AI System Runner")
    parser.add_argument('--query', type=str, default="What is the capital of France? Explain why.", help='Input query for the system')
    parser.add_argument('--config', type=str, default="configs/config.yaml", help='Path to config file')
    args = parser.parse_args()

    pipeline = Pipeline(config_path=args.config)
    result = pipeline.run(args.query)
    print("\n=== Pipeline Output ===")
    print(result)

if __name__ == "__main__":
    main()

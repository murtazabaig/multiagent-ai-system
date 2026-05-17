"""
Main entrypoint for Multi-Agent AI System (Planner + Retriever + Executor)
Run this file to test the end-to-end pipeline, evaluate, or launch the UI.
"""
import argparse
import sys
import os
import subprocess

# Fix windows console unicode errors
sys.stdout.reconfigure(encoding='utf-8')

from src.core.pipeline import Pipeline

def print_banner():
    print("\n" + "="*50)
    print("🚀 MULTI-AGENT AI SYSTEM - MAIN MENU")
    print("="*50)

def launch_dashboard():
    print("\n[+] Launching Premium Streamlit Dashboard...")
    print("Press Ctrl+C in this terminal to stop the server.")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])

def run_evaluation():
    print("\n[+] Running Evaluation Suite (ROUGE / BLEU)...")
    subprocess.run([sys.executable, "evaluate.py"])

def run_cli_query():
    query = input("\n[?] Enter your query: ")
    if not query.strip():
        print("Query cannot be empty.")
        return
        
    print("\n[+] Initializing Pipeline (Loading Models & Index)...")
    pipeline = Pipeline()
    
    print("\n[+] Running Agents...")
    result = pipeline.run(query)
    
    print("\n" + "="*50)
    print("🧠 EXECUTIVE SYNTHESIS")
    print("="*50)
    final = result.get("results", [])[-1].get("result", "No result generated.")
    print(final)
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent AI System Runner")
    parser.add_argument('--ui', action='store_true', help='Launch the Streamlit Dashboard directly')
    parser.add_argument('--eval', action='store_true', help='Run the evaluation suite directly')
    args = parser.parse_args()

    # If flags are provided, run directly without menu
    if args.ui:
        launch_dashboard()
        return
    if args.eval:
        run_evaluation()
        return

    # Interactive Menu
    while True:
        print_banner()
        print("1. Launch Streamlit Dashboard (Premium UI)")
        print("2. Run Full System Evaluation (Classification/Retrieval/Generation)")
        print("3. Ask a Custom Question (CLI Mode)")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            launch_dashboard()
        elif choice == '2':
            run_evaluation()
        elif choice == '3':
            run_cli_query()
        elif choice == '4':
            print("Exiting... Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()

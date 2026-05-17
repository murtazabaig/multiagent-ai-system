# Autonomous Multi-Agent Reasoning System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0052CC?style=for-the-badge&logo=facebook&logoColor=white)

A localized, Multi-Agent Artificial Intelligence System utilizing a **Retrieval-Augmented Generation (RAG)** architecture. Developed for the Artificial Neural Networks & Deep Learning Lab, this project divides cognitive tasks across three distinct agents to autonomously ingest, search, and synthesize information from complex documents.

Developed by **Murtaza Baig**.

## 🧠 System Architecture

The system avoids the hallucination risks of monolithic LLMs by structuring a robust pipeline of specialized agents:

1. **Planner Agent:** Decomposes complex user queries into logical sub-tasks.
2. **Retriever Agent:** Utilizes `all-MiniLM-L6-v2` embeddings and a **FAISS** (Facebook AI Similarity Search) index to perform hyper-fast semantic search across ingested documents.
3. **Executor Agent:** Powered by `google/flan-t5-base`, an instruction-tuned LLM that synthesizes direct, fact-grounded answers based strictly on the retrieved context.

## 🚀 Key Features

* **Dynamic Knowledge Ingestion:** Upload custom PDFs directly via the Streamlit dashboard for real-time text chunking, embedding, and vectorization.
* **Automated Evaluation Suite:** A built-in evaluation script (`evaluate.py`) that tests the system against the SQuAD benchmark dataset, calculating quantitative metrics:
  * **Classification:** Token-Level Accuracy (Exact Match), Precision, Recall, F1 Score.
  * **Retrieval:** Recall@K, Mean Reciprocal Rank (MRR).
  * **Generation:** BLEU Score, ROUGE-1 / ROUGE-L F1.
* **Premium Glassmorphic UI:** A sleek, modern Streamlit dashboard for real-time interaction and telemetry monitoring.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/murtazabaig/multiagent-ai-system.git
   cd multiagent-ai-system
   ```

2. **Install dependencies:**
   Ensure you have Python 3.10+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python main.py
   ```
   * Select **Option 1** to launch the Streamlit Web Interface.
   * Select **Option 2** to run the System Evaluation benchmarks.
   * Select **Option 3** to run a headless CLI query.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Copyright (c) 2026 Murtaza Baig.

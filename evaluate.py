import json
import time
from tqdm import tqdm
from src.core.pipeline import Pipeline
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Fix windows console unicode errors
import sys
sys.stdout.reconfigure(encoding='utf-8')

def load_qa_pairs(file_path="train-v2.0.json", limit=15):
    """Loads a specific number of Question-Answer pairs from the raw SQuAD dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)['data']
    
    qa_pairs = []
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if not qa['is_impossible'] and len(qa['answers']) > 0:
                    question = qa['question']
                    answer = qa['answers'][0]['text']
                    qa_pairs.append({"question": question, "answer": answer})
                    if len(qa_pairs) >= limit:
                        return qa_pairs
    return qa_pairs

def calculate_classification_metrics(prediction: str, ground_truth: str):
    """Token-level precision, recall, f1, and exact match (accuracy) for QA."""
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    
    if not pred_tokens or not gt_tokens:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    common_tokens = set(pred_tokens) & set(gt_tokens)
    tp = len(common_tokens)
    
    precision = tp / len(pred_tokens)
    recall = tp / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Accuracy is traditionally Exact Match (EM) or if the GT is perfectly contained
    accuracy = 1.0 if ground_truth.lower() in prediction.lower() else 0.0
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def calculate_retrieval_metrics(retrieved_docs: list, ground_truth: str):
    """Calculates Recall@K and MRR based on if the answer is in the retrieved docs."""
    mrr = 0.0
    recall_at_k = 0.0
    
    for rank, doc in enumerate(retrieved_docs, start=1):
        if ground_truth.lower() in doc.lower():
            if mrr == 0.0:
                mrr = 1.0 / rank
            recall_at_k = 1.0  # Found at least once in top K
            
    return {"mrr": mrr, "recall_at_k": recall_at_k}

def main():
    print("\n" + "="*50)
    print("📊 MULTI-AGENT SYSTEM FULL EVALUATION PLAN")
    print("="*50)
    
    print("[1/3] Loading SQuAD dataset...")
    qa_pairs = load_qa_pairs(limit=15)
    print(f"Successfully loaded {len(qa_pairs)} ground-truth QA pairs.")
    
    print("\n[2/3] Initializing Multi-Agent Pipeline...")
    pipeline = Pipeline()
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    
    # Metric accumulators
    metrics = {
        "rouge1": 0, "rougeL": 0, "bleu": 0,
        "accuracy": 0, "precision": 0, "recall": 0, "f1": 0,
        "mrr": 0, "recall_at_k": 0
    }
    results_log = []
    
    print("\n[3/3] Commencing Evaluation Loop (Comparing AI vs Human)...")
    
    for i, item in enumerate(tqdm(qa_pairs)):
        question = item['question']
        ground_truth = item['answer']
        prediction = ""
        retrieved_docs = []
        
        try:
            result = pipeline.run(question)
            prediction = result["results"][-1].get("result", "")
            # Extract retrieved documents from the pipeline plan
            for step in result.get("results", []):
                if "documents" in step:
                    retrieved_docs = step["documents"]
                    break
        except Exception as e:
            print(f"Error on question {i}: {e}")
            
        # 1. Generation Metrics
        rouge_scores = scorer.score(ground_truth, prediction)
        r1 = rouge_scores['rouge1'].fmeasure
        rl = rouge_scores['rougeL'].fmeasure
        bleu = sentence_bleu([ground_truth.split()], prediction.split(), smoothing_function=smoothie)
        
        # 2. Classification Metrics
        cls_metrics = calculate_classification_metrics(prediction, ground_truth)
        
        # 3. Retrieval Metrics
        ret_metrics = calculate_retrieval_metrics(retrieved_docs, ground_truth)
        
        # Accumulate
        metrics["rouge1"] += r1
        metrics["rougeL"] += rl
        metrics["bleu"] += bleu
        metrics["accuracy"] += cls_metrics["accuracy"]
        metrics["precision"] += cls_metrics["precision"]
        metrics["recall"] += cls_metrics["recall"]
        metrics["f1"] += cls_metrics["f1"]
        metrics["mrr"] += ret_metrics["mrr"]
        metrics["recall_at_k"] += ret_metrics["recall_at_k"]
        
        results_log.append({
            "question": question,
            "ground_truth": ground_truth,
            "ai_prediction": prediction,
            "generation": {"rouge1": round(r1, 4), "rougeL": round(rl, 4), "bleu": round(bleu, 4)},
            "classification": {k: round(v, 4) for k,v in cls_metrics.items()},
            "retrieval": {k: round(v, 4) for k,v in ret_metrics.items()}
        })
        
    # Calculate Averages
    num_q = len(qa_pairs)
    avg = {k: v / num_q for k, v in metrics.items()}
    
    print("\n\n" + "="*50)
    print("🎯 FINAL EVALUATION PLAN METRICS")
    print("="*50)
    print("1. CLASSIFICATION METRICS (Token-Level)")
    print(f"   Accuracy (Exact Match) : {avg['accuracy']:.4f}")
    print(f"   Precision              : {avg['precision']:.4f}")
    print(f"   Recall                 : {avg['recall']:.4f}")
    print(f"   F1 Score               : {avg['f1']:.4f}")
    print("\n2. RETRIEVAL METRICS (FAISS Vector Space)")
    print(f"   Recall@K               : {avg['recall_at_k']:.4f}")
    print(f"   Mean Reciprocal Rank   : {avg['mrr']:.4f}")
    print("\n3. GENERATION METRICS (LLM Synthesis)")
    print(f"   BLEU Score             : {avg['bleu']:.4f}")
    print(f"   ROUGE-1 F1             : {avg['rouge1']:.4f}")
    print(f"   ROUGE-L F1             : {avg['rougeL']:.4f}")
    print("="*50)
    
    with open("evaluation_report.json", "w", encoding='utf-8') as f:
        json.dump({"final_metrics": avg, "detailed_logs": results_log}, f, indent=4)
        
    print("\n✅ Saved detailed evaluation logs to 'evaluation_report.json'")

if __name__ == "__main__":
    main()

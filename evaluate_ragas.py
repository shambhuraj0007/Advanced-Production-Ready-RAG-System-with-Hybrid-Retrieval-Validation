"""
Automated RAG Evaluation using RAGAS.
Measures the RAG Triad: Faithfulness, Answer Relevance, Context Precision, and Context Recall.
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.rag_system import SimpleRAG

def load_eval_dataset(dataset_path: str = "tests/eval_dataset.json"):
    """Load benchmark question-answer dataset."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation(dataset_path: str = "tests/eval_dataset.json", use_hybrid: bool = True, use_reranking: bool = True):
    """
    Run the RAG system over the test dataset and compute RAGAS metrics.
    """
    eval_data = load_eval_dataset(dataset_path)
    print(f" Loaded {len(eval_data)} test questions from {dataset_path}")
    
    # 1. Initialize RAG system
    print(f"\n Initializing RAG System (Hybrid={use_hybrid}, Re-ranking={use_reranking})...")
    rag = SimpleRAG(
        use_hybrid_search=use_hybrid,
        use_reranking=use_reranking,
        use_memory=False  # Keep stateless for deterministic evaluation
    )
    
    # 2. Ingest documents if ChromaDB doesn't have them
    doc_dir = Path("documents")
    if doc_dir.exists():
        sample_files = [str(p) for p in doc_dir.glob("*.txt")]
        if sample_files:
            print(f" Ingesting document files: {[Path(f).name for f in sample_files]}")
            rag.load_from_files(sample_files)
    
    # 3. Generate answers and gather contexts
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    print("\n Running inference on benchmark dataset...")
    for idx, item in enumerate(eval_data, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f" [{idx}/{len(eval_data)}] Question: {q}")
        
        res = rag.query(q)
        answer = res.get("result", "")
        source_docs = res.get("source_documents", [])
        retrieved_contexts = [doc.page_content for doc in source_docs]
        
        questions.append(q)
        answers.append(answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(gt)
    
    # 4. Prepare HuggingFace Dataset for RAGAS
    from datasets import Dataset
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    hf_dataset = Dataset.from_dict(data_dict)
    
    # 5. Run RAGAS Evaluation
    print("\n Computing RAGAS metrics (Faithfulness, Answer Relevance, Context Precision, Context Recall)...")
    
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        )
        
        # Configure evaluator with rag's LLM and embeddings if available
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        
        eval_kwargs = {
            "dataset": hf_dataset,
            "metrics": metrics
        }
        
        # Try to pass provider LLM/embeddings to avoid OpenAI default requirement
        try:
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            eval_kwargs["llm"] = LangchainLLMWrapper(rag.llm)
            eval_kwargs["embeddings"] = LangchainEmbeddingsWrapper(rag.embeddings)
        except Exception:
            pass  # Fallback to standard ragas defaults
        
        results = evaluate(**eval_kwargs)
        df_results = results.to_pandas()
        
        # 6. Print and Save Benchmark Report
        print("\n" + "="*70)
        print(" RAGAS BENCHMARK RESULTS")
        print("="*70)
        
        score_summary = {}
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if m in df_results.columns:
                val = float(df_results[m].mean())
                score_summary[m] = round(val, 4)
                print(f" {m.replace('_', ' ').title():<25}: {val:.4f}")
        print("="*70)
        
        # Save to JSON
        output_file = "benchmark_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "configuration": {
                    "provider": rag.provider,
                    "hybrid_search": use_hybrid,
                    "reranking": use_reranking
                },
                "scores": score_summary,
                "samples_count": len(eval_data)
            }, f, indent=2)
            
        print(f"\n Detailed results saved to {output_file}")
        return score_summary
        
    except Exception as e:
        print(f"\n Note: RAGAS evaluation execution encountered: {e}")
        print("Saving dataset & responses for manual inspection.")
        with open("benchmark_samples.json", "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2)
        return None

if __name__ == "__main__":
    run_evaluation()

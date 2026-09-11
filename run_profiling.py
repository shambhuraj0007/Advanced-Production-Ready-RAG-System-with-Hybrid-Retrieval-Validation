#!/usr/bin/env python3
"""
Run profiling on the RAG system.
Usage: python run_profiling.py
"""

import os
import sys

# Enable profiling BEFORE importing
os.environ["RAG_ENABLE_PROFILING"] = "true"

# Now import
try:
    from src.rag_system import SimpleRAG
except ImportError:
    print("Error: Could not import SimpleRAG")
    print("Make sure you're in the project root and dependencies are installed.")
    sys.exit(1)

def main():
    print("="*70)
    print("RAG SYSTEM PERFORMANCE PROFILING")
    print("="*70)
    print()
    
    # Initialize
    print("Initializing RAG system...")
    rag = SimpleRAG()
    print("✓ Initialized")
    
    # Load from ChromaDB
    print("\nLoading from ChromaDB...")
    if os.path.exists("./chroma_db/chroma.sqlite3"):
        success = rag.load_from_chromadb("./chroma_db")
        if success:
            print("✓ Loaded")
        else:
            print("⚠ Failed to load. Please load documents first.")
            return
    else:
        print("⚠ ChromaDB not found. Please load documents first.")
        return
    
    # Run test query
    print("\n" + "="*70)
    print("Running test query with profiling...")
    print("="*70)
    
    test_query = "What is the main topic?"
    print(f"\nQuery: {test_query}\n")
    
    result = rag.query(test_query, enable_validation=True)
    
    # Display detailed timings
    timings = result.get("_timings", {})
    
    if timings:
        print("\n" + "="*70)
        print("DETAILED TIMING BREAKDOWN")
        print("="*70)
        print(f"\n{'Operation':<40} {'Time (s)':<15} {'% of Total':<15}")
        print("-"*70)
        
        total = timings.get("total", 0)
        sorted_ops = sorted(
            [(k, v) for k, v in timings.items() if k != "total"],
            key=lambda x: x[1],
            reverse=True
        )
        
        for op, duration in sorted_ops:
            pct = (duration / total * 100) if total > 0 else 0
            print(f"{op:<40} {duration:<15.4f} {pct:<15.1f}%")
        
        print("-"*70)
        print(f"{'TOTAL':<40} {total:<15.4f} {'100.0%':<15}")
        
        # Bottleneck analysis
        print("\n" + "="*70)
        print("BOTTLENECK ANALYSIS")
        print("="*70)
        if sorted_ops:
            print("\nTop 3 slowest operations:")
            for i, (op, dur) in enumerate(sorted_ops[:3], 1):
                pct = (dur / total * 100) if total > 0 else 0
                print(f"  {i}. {op}: {dur:.2f}s ({pct:.1f}% of total)")
        
        # Optimization suggestions
        print("\n" + "="*70)
        print("OPTIMIZATION SUGGESTIONS")
        print("="*70)
        suggestions = []
        
        if "answer_validation" in timings and timings["answer_validation"] > 1:
            suggestions.append("• Disable validation: query(..., enable_validation=False) - saves ~{:.1f}s".format(
                timings["answer_validation"]
            ))
        
        if "query_routing" in timings and timings["query_routing"] > 1:
            suggestions.append("• Disable query routing: use_query_routing=False - saves ~{:.1f}s".format(
                timings["query_routing"]
            ))
        
        if "query_rewriting" in timings and timings["query_rewriting"] > 1:
            suggestions.append("• Disable query rewriting: use_query_rewriting=False - saves ~{:.1f}s".format(
                timings["query_rewriting"]
            ))
        
        if "document_retrieval" in timings and timings["document_retrieval"] > 1.5:
            suggestions.append("• Consider reducing retrieval_k or disabling reranking")
        
        if suggestions:
            for suggestion in suggestions:
                print(suggestion)
        else:
            print("• System is already well optimized!")
    else:
        print("\n⚠ No timing data available")
        print("   Profiling should be enabled automatically when RAG_ENABLE_PROFILING=true")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


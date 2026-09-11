# Performance Profiling Guide

## 🎯 How to Enable Profiling

### Method 1: Environment Variable (Recommended)
```bash
export RAG_ENABLE_PROFILING=true
python your_script.py
```

### Method 2: In Code
```python
import os
os.environ["RAG_ENABLE_PROFILING"] = "true"

from src.rag_system import SimpleRAG
rag = SimpleRAG(...)
result = rag.query("Your question")
```

### Method 3: Use the Profiling Script
```bash
python run_profiling.py
```

## 📊 What Gets Profiled

The profiling system tracks time for these operations:

1. **query_routing** - Query classification and routing strategy
2. **query_rewriting** - Query rewriting/expansion for better retrieval
3. **document_retrieval** - Vector + BM25 hybrid search and reranking
4. **llm_generation** - LLM call to generate the answer
5. **answer_validation** - Validating answer against sources (optional)
6. **citation_formatting** - Formatting source citations
7. **tool_execution** - External tool execution (calculator, web search)
8. **total** - Total query time

## 📈 Example Output

When profiling is enabled, you'll see output like:

```
⏱️  Query Performance:
   Total: 8.45s
   llm_generation: 3.21s (38.0%)
   document_retrieval: 2.15s (25.4%)
   answer_validation: 1.89s (22.4%)
   query_routing: 0.95s (11.2%)
   query_rewriting: 0.15s (1.8%)
   citation_formatting: 0.10s (1.2%)
```

## 🔍 Accessing Detailed Timings

```python
result = rag.query("Your question")
timings = result.get("_timings", {})

# Print all timings
for operation, duration in timings.items():
    print(f"{operation}: {duration:.2f}s")

# Get specific timing
llm_time = timings.get("llm_generation", 0)
print(f"LLM took {llm_time:.2f}s")
```

## 📋 Typical Time Ranges

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| query_routing | 0.5-2s | LLM call for classification |
| query_rewriting | 0.5-2s | LLM call for rewriting |
| document_retrieval | 0.5-3s | Vector search + BM25 + reranking |
| llm_generation | 2-5s | Main answer generation |
| answer_validation | 1-3s | Optional validation LLM call |
| citation_formatting | 0.05-0.2s | Fast, just formatting |
| tool_execution | 0.5-5s | Depends on tool (web search slower) |
| **Total** | **5-15s** | Varies by features enabled |

## 🎯 Interpreting Results

### High LLM Generation Time (>5s)
- **Cause**: Large context or complex query
- **Fix**: Reduce context size, use faster model, or simplify query

### High Document Retrieval Time (>3s)
- **Cause**: Large document set, expensive reranking
- **Fix**: 
  - Reduce `retrieval_k`
  - Disable reranking: `use_reranking=False`
  - Use lighter reranker model

### High Validation Time (>2s)
- **Cause**: Additional LLM call after answer
- **Fix**: Disable validation: `query(..., enable_validation=False)`

### High Query Routing Time (>2s)
- **Cause**: LLM call for classification
- **Fix**: Disable routing: `use_query_routing=False`

### High Query Rewriting Time (>2s)
- **Cause**: LLM call for query expansion
- **Fix**: Disable rewriting: `use_query_rewriting=False`

## 🚀 Quick Optimization Based on Profiling

```python
# If validation is slow
result = rag.query("question", enable_validation=False)

# If routing is slow
rag = SimpleRAG(..., use_query_routing=False)

# If rewriting is slow
rag = SimpleRAG(..., use_query_rewriting=False)

# If retrieval is slow
rag = SimpleRAG(..., use_reranking=False, rerank_top_k=3)
```

## 📝 Example: Full Profiling Session

```python
import os
os.environ["RAG_ENABLE_PROFILING"] = "true"

from src.rag_system import SimpleRAG

rag = SimpleRAG(
    persist_directory="./chroma_db",
    use_query_routing=True,
    use_query_rewriting=True,
    use_answer_validation=True,
    use_reranking=True
)

# Load documents or ChromaDB
rag.load_from_chromadb("./chroma_db")

# Run query
result = rag.query("What is the main topic?")

# Access timings
timings = result.get("_timings", {})
print("\nDetailed Breakdown:")
for op, time in sorted(timings.items(), key=lambda x: x[1], reverse=True):
    print(f"  {op}: {time:.2f}s")
```

## 💡 Tips

1. **Profile first, optimize second** - Know where time is spent before optimizing
2. **Profile multiple queries** - Some operations vary by query type
3. **Compare with/without features** - See actual impact of each feature
4. **Check network latency** - LLM calls include network time
5. **Monitor over time** - Performance can vary with API load

## 🔧 Troubleshooting

### No timing data in result
- Check `RAG_ENABLE_PROFILING` is set to "true"
- Ensure it's set BEFORE importing SimpleRAG
- Check that `_timings` key exists in result

### Timings seem incorrect
- Some operations may overlap (timing is sequential)
- Network latency affects LLM calls
- First query may be slower (cold start)

### Want more detail
- Check individual component code for internal timing
- Use Python's `cProfile` for deeper analysis
- Monitor API response times separately


# Advanced RAG System with LangChain

A comprehensive Retrieval-Augmented Generation (RAG) system built with Python and LangChain 1.2.3. This system features advanced retrieval techniques, conversational memory, query optimization, answer validation, and external tool integration.

## ✨ Features

### Core RAG Capabilities
- **📄 Document Loading**: Support for PDF, TXT, and MD files
- **🔍 Hybrid Search**: Combines vector similarity search with BM25 keyword search for comprehensive retrieval
- **🎯 Re-ranking**: Uses cross-encoder models to improve document relevance
- **💾 Vector Storage**: ChromaDB for efficient document embedding storage
- **📎 Enhanced Citations**: Detailed source attribution with file names, page numbers, and paths

### Advanced Features
- **💭 Conversational Memory**: Maintains context across multiple queries using conversation buffer memory
- **🔄 Query Rewriting**: Automatically improves queries for better retrieval using LLM-based rewriting
- **✅ Answer Validation**: Validates answers against source documents with confidence scores and warnings
- **🧭 Query Routing**: Intelligently classifies queries and adapts retrieval strategy (factual, analytical, multi-hop)
- **🛠️ External Tools**: Integration with calculator and web search (Tavily/DuckDuckGo) for enhanced capabilities
- **🎨 Web UI**: Streamlit-based user interface for easy interaction

### Technical Highlights
- **LCEL Architecture**: Built with LangChain Expression Language for composable chains
- **Modular Design**: Clean separation of concerns with dedicated modules for each component
- **Provider Flexibility**: Supports Google Gemini, Groq, and OpenAI providers
- **Router-Aware Tool Selection**: Intelligent tool selection based on query classification

## 📁 Project Structure

```
ragV0/
├── src/
│   ├── __init__.py
│   ├── rag_system.py          # Main RAG system implementation
│   ├── retriever.py           # Hybrid retrieval (vector + BM25)
│   ├── memory_manager.py      # Conversational memory management
│   ├── query_router.py        # Query classification and routing
│   ├── query_rewriter.py      # Query rewriting and expansion
│   ├── validator.py           # Answer validation
│   ├── citation_formatter.py  # Citation formatting
│   ├── tool_manager.py        # External tools management
│   └── tools.py               # Tool implementations (calculator, web search)
├── documents/                 # Place your documents here
├── chroma_db/                 # Vector store persistence (auto-generated)
├── app.py                     # Streamlit web UI
├── example.py                 # Command-line example usage
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# For Google Gemini (default)
GEMINI_API_KEY=your_gemini_api_key_here

# For Groq
GROQ_API_KEY=your_groq_api_key_here

# Optional: For web search (Tavily)
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: For OpenAI
# OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Web UI

```bash
streamlit run app.py
```

The UI will open in your browser at `http://localhost:8501`

### 4. Or Use the Command-Line Example

```bash
python example.py
```

## 📖 Usage

### Web UI Usage

1. **Initialize System**: Click "🔄 Initialize RAG System" in the sidebar
2. **Configure Features**: Toggle features and adjust settings in the sidebar
3. **Load Documents**: 
   - Upload files in the "Documents" tab, or
   - Load from a directory
4. **Start Chatting**: Ask questions in the "Chat" tab

### Programmatic Usage

```python
from src.rag_system import SimpleRAG

# Initialize the RAG system (all settings auto-load from .env by default)
rag = SimpleRAG()

# Or optionally override specific settings in code:
# rag = SimpleRAG(provider="groq", llm_model="llama-3.3-70b-versatile")

# Load documents
rag.load_from_files([
    "documents/document1.pdf",
    "documents/document2.txt"
])

# Query the system
result = rag.query("What is the main topic?")
print(result['result'])

# Access additional information
print(f"Citations: {result.get('citations', [])}")
print(f"Validation: {result.get('validation', {})}")
print(f"Routing: {result.get('routing', {})}")
```

## ⚙️ Configuration

### Initialization Parameters

#### Core Settings
- `embedding_model`: Embedding model name (default: "models/text-embedding-004" for Gemini, "all-MiniLM-L6-v2" for Groq)
- `llm_model`: LLM model name (default: "gemini-1.5-flash" for Gemini, "llama-3.3-70b-versatile" for Groq)
- `provider`: Provider to use - "gemini", "groq", or "openai" (default: "gemini")
- `persist_directory`: Directory to persist vector store (None for in-memory)
- `api_key`: API key (optional, uses environment variable if not provided)
- `base_url`: Base URL for API (optional, uses default based on provider)

#### Feature Flags
- `use_reranking`: Enable re-ranking for better document selection (default: True)
- `rerank_top_k`: Number of top documents to keep after re-ranking (default: 3)
- `reranker_model`: Model name for the cross-encoder reranker (default: "cross-encoder/ms-marco-MiniLM-L-6-v2")
- `use_memory`: Enable conversational memory (default: True)
- `memory_type`: Type of memory ("buffer" for conversation buffer memory)
- `use_hybrid_search`: Enable hybrid search (vector + BM25) (default: True)
- `hybrid_search_alpha`: Weight for vector search (0.0=only BM25, 1.0=only vector, 0.5=balanced)
- `use_query_rewriting`: Enable query rewriting for better retrieval (default: True)
- `use_answer_validation`: Enable answer validation against sources (default: True)
- `use_query_routing`: Enable query routing for adaptive retrieval (default: True)
- `use_tools`: Enable external tools (default: True)
- `enable_calculator`: Enable calculator tool (default: True)
- `enable_web_search`: Enable web search tool (default: True)
- `web_search_provider`: Web search provider ("tavily" or "duckduckgo", default: "tavily")
- `web_search_api_key`: API key for web search (optional, uses TAVILY_API_KEY from env)

### Example Configurations

#### Minimal Configuration
```python
rag = SimpleRAG(
    provider="gemini",
    persist_directory="./chroma_db"
)
```

#### Full-Featured Configuration
```python
rag = SimpleRAG(
    provider="gemini",
    embedding_model="models/text-embedding-004",
    llm_model="gemini-1.5-flash",
    persist_directory="./chroma_db",
    use_reranking=True,
    rerank_top_k=5,
    use_memory=True,
    use_hybrid_search=True,
    hybrid_search_alpha=0.6,
    use_query_rewriting=True,
    use_answer_validation=True,
    use_query_routing=True,
    use_tools=True,
    enable_calculator=True,
    enable_web_search=True,
    web_search_provider="tavily"
)
```

#### OpenAI Configuration
```python
rag = SimpleRAG(
    embedding_model="text-embedding-3-large",
    llm_model="gpt-4",
    provider="openai",
    persist_directory="./chroma_db"
)
```

## 🔍 Feature Details

### Hybrid Search
Combines semantic vector search with keyword-based BM25 search:
- **Vector Search**: Captures semantic meaning and synonyms
- **BM25 Search**: Captures exact keyword matches and term frequency
- **Alpha Parameter**: Controls the balance (0.5 = equal weight)

### Re-ranking
Uses cross-encoder models to re-rank retrieved documents:
- More accurate than cosine similarity alone
- Considers query-document interaction
- Configurable top-k results

### Query Routing
Intelligently classifies queries and adapts retrieval:
- **Query Types**: Factual, Analytical, Multi-hop, Conversational
- **Complexity Levels**: Simple, Medium, Complex
- **Adaptive K**: Adjusts number of documents retrieved based on query complexity

### Query Rewriting
Improves queries for better retrieval:
- Expands abbreviations
- Adds context from conversation history
- Rephrases for clarity

### Answer Validation
Validates generated answers against source documents:
- **Confidence Score**: Measures answer reliability
- **Warnings**: Flags unsupported claims
- **Supporting Evidence**: Lists relevant source excerpts

### External Tools
- **Calculator**: Handles mathematical computations
- **Web Search**: Retrieves current information from the web (Tavily or DuckDuckGo)
- **Router-Aware Selection**: Tools are selected based on query classification

## 📊 Query Response Structure

The `query()` method returns a dictionary with the following structure:

```python
{
    'result': str,                    # The generated answer
    'source_documents': List[Document], # Retrieved source documents
    'citations': List[Dict],          # Formatted citations with metadata
    'validation': Dict,               # Validation results (if enabled)
    'routing': Dict,                  # Query routing information (if enabled)
    'rewritten_query': str,           # Rewritten query (if enabled)
    'tools_used': List[str],          # List of tools used (if any)
    'tool_results': Dict              # Tool execution results (if any)
}
```

## 🛠️ Dependencies

- `langchain==1.2.3`: Core LangChain framework
- `langchain-community==0.4.1`: Community integrations
- `langchain-google-genai>=2.0.0`: Google Gemini integrations
- `langchain-groq>=0.2.0`: Groq integrations
- `langchain-openai>=0.2.0`: OpenAI integrations
- `chromadb==1.4.0`: Vector database
- `python-dotenv==1.2.1`: Environment variable management
- `pypdf==6.6.0`: PDF document loading
- `sentence-transformers>=2.2.0`: Re-ranking models
- `rank-bm25>=0.2.2`: BM25 keyword search
- `tavily-python>=0.3.0`: Tavily web search API
- `duckduckgo-search>=3.9.0`: DuckDuckGo web search
- `streamlit>=1.28.0`: Web UI framework

## 🏗️ Architecture

### Modular Components

1. **SimpleRAG** (`rag_system.py`): Main orchestrator class
2. **HybridRetriever** (`retriever.py`): Handles vector + BM25 retrieval and re-ranking
3. **MemoryManager** (`memory_manager.py`): Manages conversational memory
4. **QueryRouter** (`query_router.py`): Classifies queries and routes retrieval
5. **QueryRewriter** (`query_rewriter.py`): Rewrites queries for better retrieval
6. **AnswerValidator** (`validator.py`): Validates answers against sources
7. **CitationFormatter** (`citation_formatter.py`): Formats source citations
8. **ToolManager** (`tool_manager.py`): Manages external tool selection and execution
9. **Tools** (`tools.py`): Individual tool implementations

### LCEL Chain Structure

The system uses LangChain Expression Language (LCEL) for composable chains:

```python
qa_chain = (
    RunnableLambda(retrieve_and_format)  # Retrieval and context building
    | prompt                              # Prompt template
    | llm                                 # Language model
    | StrOutputParser()                   # Output parsing
)
```

## 📝 Notes

- **Google Gemini is the default provider** - Make sure you have a valid `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in your `.env` file
- **Groq is supported for fast inference** - Set `provider="groq"` and set `GROQ_API_KEY` in `.env`
- You can also use OpenAI by setting `provider="openai"` and providing `OPENAI_API_KEY`
- The system uses ChromaDB for simplicity, but the architecture allows easy replacement with other vector databases
- Documents are automatically chunked with overlap (200 chars) for better context preservation
- Vector store is persisted to disk when `persist_directory` is specified

## 🔒 Security

- **Never commit API keys** - The `.env` file is in `.gitignore`
- Use `.env.example` as a template with placeholder values

## 🚧 Future Enhancements

Potential areas for extension:

- **Multi-modal Support**: Add support for images, tables, and charts
- **Advanced Memory**: Summary-based memory for long conversations
- **Caching Layer**: Query result and embedding caching
- **Streaming Responses**: Real-time token streaming
- **Evaluation Framework**: RAGAS metrics and benchmarking
- **API Server**: REST API with FastAPI
- **Additional Tools**: Code execution, database queries, etc.
- **Fine-tuning**: Custom embedding models for domain-specific use cases

## 📄 License

This is a starter project for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

"""
Simple RAG System using LangChain
Enhanced RAG system with modular components:
- Query routing and classification
- Query rewriting and expansion
- Hybrid search (vector + BM25)
- Re-ranking
- Answer validation
- Conversational memory
- Enhanced source citations
"""

from typing import List, Optional, Dict, Any, Set
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader, PDFPlumberLoader
from langchain_core.documents import Document
import os
import hashlib
import time
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import modular components
try:
    from .query_router import QueryRouter
    from .query_rewriter import QueryRewriter
    from .validator import AnswerValidator
    from .citation_formatter import CitationFormatter
    from .retriever import HybridRetriever
    from .memory_manager import MemoryManager
    from .tool_manager import ToolManager
except ImportError:
    # Fallback to absolute imports if relative imports fail
    from src.query_router import QueryRouter
    from src.query_rewriter import QueryRewriter
    from src.validator import AnswerValidator
    from src.citation_formatter import CitationFormatter
    from src.retriever import HybridRetriever
    from src.memory_manager import MemoryManager
    from src.tool_manager import ToolManager

load_dotenv()


class SimpleRAG:
    """
    A simple RAG (Retrieval-Augmented Generation) system with modular components.
    
    This implementation uses:
    - ChromaDB for vector storage (can be replaced with Milvus later)
    - Google Gemini, Groq, or OpenAI for embeddings and LLMs
    - Modular components for query routing, rewriting, validation, etc.
    """
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        provider: Optional[str] = None,
        persist_directory: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_reranking: Optional[bool] = None,
        rerank_top_k: Optional[int] = None,
        reranker_model: Optional[str] = None,
        use_memory: Optional[bool] = None,
        memory_type: Optional[str] = None,
        use_hybrid_search: Optional[bool] = None,
        hybrid_search_alpha: Optional[float] = None,
        use_query_rewriting: Optional[bool] = None,
        use_answer_validation: Optional[bool] = None,
        use_query_routing: Optional[bool] = None,
        use_tools: Optional[bool] = None,
        enable_calculator: Optional[bool] = None,
        enable_web_search: Optional[bool] = None,
        web_search_provider: Optional[str] = None,
        web_search_api_key: Optional[str] = None
    ):
        """
        Initialize the RAG system.
        All settings can be specified via parameters or configured in .env variables.
        """
        # Helper to get boolean from parameter or environment variable
        def _get_bool(param_val: Optional[bool], env_var: str, default: bool) -> bool:
            if param_val is not None:
                return param_val
            env_val = os.getenv(env_var)
            if env_val is not None:
                return env_val.strip().lower() in ("true", "1", "yes")
            return default

        self.provider = (provider or os.getenv("RAG_PROVIDER", "groq")).lower()
        default_persist = "/tmp/chroma_db" if os.getenv("VERCEL") == "1" else "./chroma_db"
        self.persist_directory = persist_directory or os.getenv("RAG_PERSIST_DIRECTORY", default_persist)

        # Set default models based on provider if not specified
        env_embedding = os.getenv("RAG_EMBEDDING_MODEL")
        env_llm = os.getenv("RAG_LLM_MODEL")
        if self.provider == "gemini":
            self.embedding_model = embedding_model or env_embedding or "models/gemini-embedding-001"
            self.llm_model = llm_model or env_llm or "gemini-1.5-flash"
        elif self.provider == "groq":
            has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
            default_emb = "models/gemini-embedding-001" if has_gemini else "all-MiniLM-L6-v2"
            self.embedding_model = embedding_model or env_embedding or default_emb
            self.llm_model = llm_model or env_llm or "openai/gpt-oss-120b"
        elif self.provider == "openai":
            self.embedding_model = embedding_model or env_embedding or "text-embedding-3-small"
            self.llm_model = llm_model or env_llm or "gpt-4o-mini"
        else:
            raise ValueError(f"Unsupported provider: '{self.provider}'. Supported providers are: 'gemini', 'groq', 'openai'.")

        # Configuration flags from parameters or .env
        use_reranking = _get_bool(use_reranking, "RAG_USE_RERANKING", True)
        rerank_top_k = rerank_top_k if rerank_top_k is not None else int(os.getenv("RAG_RERANK_TOP_K", "3"))
        reranker_model = reranker_model or os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        use_memory = _get_bool(use_memory, "RAG_USE_MEMORY", True)
        memory_type = memory_type or os.getenv("RAG_MEMORY_TYPE", "buffer")
        use_hybrid_search = _get_bool(use_hybrid_search, "RAG_USE_HYBRID_SEARCH", True)
        hybrid_search_alpha = hybrid_search_alpha if hybrid_search_alpha is not None else float(os.getenv("RAG_HYBRID_ALPHA", "0.5"))
        use_query_rewriting = _get_bool(use_query_rewriting, "RAG_USE_QUERY_REWRITING", False)
        use_answer_validation = _get_bool(use_answer_validation, "RAG_USE_ANSWER_VALIDATION", False)
        use_query_routing = _get_bool(use_query_routing, "RAG_USE_QUERY_ROUTING", True)
        use_tools = _get_bool(use_tools, "RAG_USE_TOOLS", True)
        enable_calculator = _get_bool(enable_calculator, "RAG_ENABLE_CALCULATOR", True)
        enable_web_search = _get_bool(enable_web_search, "RAG_ENABLE_WEB_SEARCH", bool(os.getenv("TAVILY_API_KEY")))
        web_search_provider = web_search_provider or os.getenv("RAG_WEB_SEARCH_PROVIDER", "tavily")
        web_search_api_key = web_search_api_key or os.getenv("TAVILY_API_KEY")
        
        # Get API key from parameter or environment
        if api_key is None:
            if self.provider == "gemini":
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) not found in environment variables. Please set it in your .env file.")
            elif self.provider == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
            elif self.provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
        
        # Initialize embeddings
        if self.provider == "gemini":
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.embedding_model,
                google_api_key=api_key
            )
        elif self.provider == "groq":
            gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            if ("text-embedding" in self.embedding_model.lower() or "gemini" in self.embedding_model.lower()) and gemini_key:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model=self.embedding_model,
                    google_api_key=gemini_key
                )
            elif "openai" in self.embedding_model.lower() and openai_key:
                self.embeddings = OpenAIEmbeddings(
                    model=self.embedding_model,
                    openai_api_key=openai_key
                )
            else:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model
                )
        elif self.provider == "openai":
            openai_kwargs = {
                "model": self.embedding_model,
                "openai_api_key": api_key
            }
            if base_url:
                openai_kwargs["base_url"] = base_url
            self.embeddings = OpenAIEmbeddings(**openai_kwargs)
        
        # Initialize LLM
        # Initialize LLM with automatic multi-provider fallback
        if self.provider == "groq":
            primary_llm = ChatGroq(
                model_name=self.llm_model,
                temperature=0,
                groq_api_key=api_key,
            )
            # Automatic fallback to Gemini if Groq errors or hits rate limit
            gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if gemini_key:
                try:
                    fallback_llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        temperature=0,
                        google_api_key=gemini_key,
                    )
                    self.llm = primary_llm.with_fallbacks([fallback_llm])
                    print("[OK] Configured Groq as primary LLM with Gemini-1.5-Flash as automatic fallback.")
                except Exception as e:
                    print(f"Warning: Could not configure Gemini fallback: {e}")
                    self.llm = primary_llm
            else:
                self.llm = primary_llm

        elif self.provider == "gemini":
            primary_llm = ChatGoogleGenerativeAI(
                model=self.llm_model,
                temperature=0,
                google_api_key=api_key,
            )
            # Automatic fallback to Groq if Gemini hits rate limits (HTTP 429)
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                try:
                    fallback_llm = ChatGroq(
                        model_name="openai/gpt-oss-120b",
                        temperature=0,
                        groq_api_key=groq_key,
                    )
                    self.llm = primary_llm.with_fallbacks([fallback_llm])
                    print("[OK] Configured Gemini as primary LLM with Groq (openai/gpt-oss-120b) as automatic fallback.")
                except Exception as e:
                    print(f"Warning: Could not configure Groq fallback: {e}")
                    self.llm = primary_llm
            else:
                self.llm = primary_llm

        elif self.provider == "openai":
            openai_kwargs = {
                "model": self.llm_model,
                "temperature": 0,
                "openai_api_key": api_key
            }
            if base_url:
                openai_kwargs["base_url"] = base_url
            self.llm = ChatOpenAI(**openai_kwargs)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize modular components
        self.memory_manager = MemoryManager(use_memory=use_memory, memory_type=memory_type)
        self.query_router = QueryRouter(self.llm, use_routing=use_query_routing)
        self.query_rewriter = QueryRewriter(self.llm, use_rewriting=use_query_rewriting)
        self.validator = AnswerValidator(self.llm, use_validation=use_answer_validation)
        self.citation_formatter = CitationFormatter()
        
        # Initialize tool manager
        self.tool_manager = ToolManager(
            llm=self.llm,
            use_tools=use_tools,
            enable_calculator=enable_calculator,
            enable_web_search=enable_web_search,
            web_search_provider=web_search_provider,
            web_search_api_key=web_search_api_key or os.getenv("TAVILY_API_KEY")
        )
        
        # Vector store and retriever (will be initialized when documents are loaded)
        self.vectorstore = None
        self.hybrid_retriever = None
        self.qa_chain = None
        
        # Performance profiling
        self.enable_profiling = os.getenv("RAG_ENABLE_PROFILING", "false").lower() == "true"
        self._last_query_timings = {}
        
        # Temporary storage for chain results (since StrOutputParser converts to string)
        self._last_retrieved_docs = []
        self._last_routing_info = None
        self._last_tool_results = None
        
        # Configuration flags
        self.use_reranking = use_reranking
        self.rerank_top_k = rerank_top_k
        self.reranker_model = reranker_model
        self.use_hybrid_search = use_hybrid_search
        self.hybrid_search_alpha = hybrid_search_alpha
        self.use_query_rewriting = use_query_rewriting
        self.use_answer_validation = use_answer_validation
        self.use_query_routing = use_query_routing
        self.use_tools = use_tools
        self.web_search_provider = web_search_provider

        # Try to load from existing ChromaDB if persist_directory is set
        if self.persist_directory and os.path.exists(self.persist_directory):
            try:
                self.load_from_chromadb()
                print(f"Loaded existing vector store from {self.persist_directory}")
            except Exception as e:
                print(f"Could not load from existing ChromaDB: {e}. Will create new one when documents are added.")
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load documents from file paths.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found, skipping...")
                continue
                
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                loader = PDFPlumberLoader(file_path)
            elif file_ext in ['.txt', '.md']:
                loader = TextLoader(file_path)
            else:
                print(f"Warning: Unsupported file type {file_ext}, skipping {file_path}")
                continue
            
            docs = loader.load()
            
            # Enhance metadata for better source citation
            for doc in docs:
                if not doc.metadata.get("source"):
                    doc.metadata["source"] = file_path
                doc.metadata["file_name"] = os.path.basename(file_path)
                doc.metadata["file_path"] = file_path
                
                # Try to extract page number if available
                if "page" not in doc.metadata and file_ext == '.pdf':
                    page_num = doc.metadata.get("page_number")
                    if page_num is not None:
                        doc.metadata["page"] = str(int(page_num)) if isinstance(page_num, (int, float)) else str(page_num)
                    else:
                        doc.metadata["page"] = "N/A"
                
                # Ensure all metadata fields are set for proper citation
                if not doc.metadata.get("file_name"):
                    doc.metadata["file_name"] = os.path.basename(file_path)
                if not doc.metadata.get("file_path"):
                    doc.metadata["file_path"] = file_path
                if not doc.metadata.get("source"):
                    doc.metadata["source"] = file_path
            
            documents.extend(docs)
        
        return documents
    
    def load_from_chromadb(self, persist_directory: Optional[str] = None) -> bool:
        """
        Load vector store from existing ChromaDB.
        
        Args:
            persist_directory: Directory containing the ChromaDB (uses self.persist_directory if None)
            
        Returns:
            True if successfully loaded, False otherwise
        """
        directory = persist_directory or self.persist_directory
        if not directory:
            print("No persist_directory specified. Cannot load from ChromaDB.")
            return False
        
        if not os.path.exists(directory):
            print(f"ChromaDB directory {directory} does not exist.")
            return False
        
        # Check if ChromaDB files exist
        chroma_sqlite = os.path.join(directory, "chroma.sqlite3")
        if not os.path.exists(chroma_sqlite):
            print(f"ChromaDB not found at {directory}. Database may be empty or corrupted.")
            return False
        
        try:
            # Load existing ChromaDB
            self.vectorstore = Chroma(
                persist_directory=directory,
                embedding_function=self.embeddings
            )
            
            # Initialize hybrid retriever
            base_retriever = self.vectorstore.as_retriever()
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=base_retriever,
                use_hybrid_search=self.use_hybrid_search,
                hybrid_search_alpha=self.hybrid_search_alpha,
                use_reranking=self.use_reranking,
                rerank_top_k=self.rerank_top_k,
                reranker_model=self.reranker_model
            )
            
            # Rebuild BM25 index from existing documents if hybrid search is enabled
            if self.use_hybrid_search:
                try:
                    # Get all documents from the vector store to rebuild BM25 index
                    all_docs = self.vectorstore.get()
                    if all_docs and 'documents' in all_docs and all_docs['documents']:
                        # Reconstruct Document objects from stored data
                        from langchain_core.documents import Document
                        documents = []
                        for i, content in enumerate(all_docs['documents']):
                            metadata = all_docs.get('metadatas', [{}])[i] if 'metadatas' in all_docs else {}
                            doc = Document(page_content=content, metadata=metadata)
                            documents.append(doc)
                        self.hybrid_retriever.build_bm25_index(documents)
                        print(f"Rebuilt BM25 index with {len(documents)} documents.")
                except Exception as e:
                    print(f"Warning: Could not rebuild BM25 index: {e}. Hybrid search may be limited.")
            
            # Create QA chain
            self._create_qa_chain(base_retriever)
            
            print(f"Successfully loaded vector store from {directory}")
            return True
            
        except Exception as e:
            print(f"Error loading ChromaDB from {directory}: {e}")
            self.vectorstore = None
            return False
    
    def get_existing_documents(self) -> Set[str]:
        """
        Get set of file paths that already exist in ChromaDB.
        
        Returns:
            Set of file paths (from metadata) that are already indexed
        """
        if not self.vectorstore:
            return set()
        
        try:
            # Get all documents from ChromaDB
            all_docs = self.vectorstore.get()
            if not all_docs or 'metadatas' not in all_docs:
                return set()
            
            # Extract unique file paths from metadata
            existing_files = set()
            for metadata in all_docs['metadatas']:
                if metadata:
                    # Check for file_path first, then source, then file_name
                    file_path = metadata.get('file_path') or metadata.get('source') or metadata.get('file_name')
                    if file_path:
                        existing_files.add(file_path)
            
            return existing_files
        except Exception as e:
            print(f"Warning: Could not check existing documents: {e}")
            return set()
    
    def check_document_exists(self, file_path: str) -> bool:
        """
        Check if a document with the given file path already exists in ChromaDB.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if document exists, False otherwise
        """
        if not self.vectorstore:
            return False
        
        existing_files = self.get_existing_documents()
        return file_path in existing_files
    
    def add_documents(self, documents: List[Document], skip_existing: bool = True) -> Dict[str, Any]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            skip_existing: If True, skip documents that already exist in ChromaDB
            
        Returns:
            Dictionary with statistics: {'added': count, 'skipped': count, 'total': count}
        """
        if not documents:
            print("No documents to add.")
            return {'added': 0, 'skipped': 0, 'total': 0}
        
        # Check for existing documents if skip_existing is enabled
        existing_files = set()
        if skip_existing and self.vectorstore:
            existing_files = self.get_existing_documents()
            print(f"Found {len(existing_files)} existing files in ChromaDB.")
        
        # Filter out documents that already exist
        new_documents = []
        skipped_count = 0
        
        for doc in documents:
            file_path = doc.metadata.get('file_path') or doc.metadata.get('source') or doc.metadata.get('file_name')
            if file_path and file_path in existing_files:
                skipped_count += 1
                print(f"Skipping existing document: {file_path}")
            else:
                new_documents.append(doc)
        
        if not new_documents:
            print(f"All {len(documents)} documents already exist in ChromaDB. Nothing to add.")
            return {'added': 0, 'skipped': skipped_count, 'total': len(documents)}
        
        print(f"Adding {len(new_documents)} new documents (skipped {skipped_count} existing).")
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(new_documents)
        print(f"Split {len(new_documents)} documents into {len(texts)} chunks.")
        
        # Create or update vector store
        if self.vectorstore is None:
            if self.persist_directory:
                # Check if ChromaDB already exists
                if os.path.exists(self.persist_directory) and os.path.exists(
                    os.path.join(self.persist_directory, "chroma.sqlite3")
                ):
                    # Load existing and add new documents
                    self.vectorstore = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embeddings
                    )
                    self.vectorstore.add_documents(texts)
                else:
                    # Create new ChromaDB
                    self.vectorstore = Chroma.from_documents(
                        documents=texts,
                        embedding=self.embeddings,
                        persist_directory=self.persist_directory
                    )
            else:
                self.vectorstore = Chroma.from_documents(
                    documents=texts,
                    embedding=self.embeddings
                )
        else:
            # Add to existing vector store
            self.vectorstore.add_documents(texts)
        
        # Initialize hybrid retriever
        base_retriever = self.vectorstore.as_retriever()
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=base_retriever,
            use_hybrid_search=self.use_hybrid_search,
            hybrid_search_alpha=self.hybrid_search_alpha,
            use_reranking=self.use_reranking,
            rerank_top_k=self.rerank_top_k,
            reranker_model=self.reranker_model
        )
        
        # Build BM25 index if hybrid search is enabled
        # Get all documents from vectorstore to rebuild complete BM25 index
        if self.use_hybrid_search:
            try:
                # Get all documents from ChromaDB to rebuild complete BM25 index
                all_docs_data = self.vectorstore.get()
                if all_docs_data and 'documents' in all_docs_data and all_docs_data['documents']:
                    # Reconstruct Document objects from stored data
                    all_documents = []
                    for i, content in enumerate(all_docs_data['documents']):
                        metadata = all_docs_data.get('metadatas', [{}])[i] if 'metadatas' in all_docs_data else {}
                        doc = Document(page_content=content, metadata=metadata)
                        all_documents.append(doc)
                    self.hybrid_retriever.build_bm25_index(all_documents)
                    print(f"Rebuilt BM25 index with {len(all_documents)} total document chunks.")
                else:
                    # Fallback: just use the new texts
                    self.hybrid_retriever.build_bm25_index(texts)
            except Exception as e:
                print(f"Warning: Could not rebuild complete BM25 index: {e}. Using new documents only.")
                self.hybrid_retriever.build_bm25_index(texts)
        
        # Create QA chain using LCEL
        self._create_qa_chain(base_retriever)
        
        print(f"Successfully added {len(new_documents)} documents to vector store.")
        
        return {
            'added': len(new_documents),
            'skipped': skipped_count,
            'total': len(documents),
            'chunks_added': len(texts)
        }
    
    def _create_qa_chain(self, retriever) -> None:
        """Create the QA chain using LCEL."""
        # Create prompt template with optional chat history
        doc_system_prompt = (
            "You are a knowledgeable assistant answering questions based on the user's provided document context.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Answer the question thoroughly and accurately using ONLY the information present in the Context below.\n"
            "2. If the provided Context does NOT contain enough information to answer the question, or if the question is about a topic not mentioned in the context, "
            "you MUST reply with EXACTLY: NOT_FOUND_IN_DOCS\n"
            "Do NOT speculate, guess, or use external knowledge if the answer is missing from the Context.\n\n"
            "Context:\n{context}"
        )
        if self.memory_manager.use_memory:
            prompt = ChatPromptTemplate.from_messages([
                ("system", doc_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", doc_system_prompt),
                ("human", "{input}")
            ])
        
        # Format documents function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Function to retrieve and format documents
        # OPTIMIZATION: Accept pre-computed routing and rewritten query to avoid duplicates
        def retrieve_and_format(input_dict):
            question = input_dict["input"]
            chat_history = input_dict.get("chat_history", [])
            
            # Use pre-computed values if available (passed from query method)
            retrieval_query = input_dict.get("_retrieval_query", question)
            routing_info = input_dict.get("_routing_info", None)
            
            # Only compute if not provided (fallback for direct chain invocation)
            if retrieval_query == question and self.use_query_rewriting:
                start_time = time.time()
                retrieval_query = self.query_rewriter.rewrite_query(question, chat_history)
                if self.enable_profiling:
                    self._last_query_timings["query_rewriting"] = time.time() - start_time
            
            if routing_info is None and self.use_query_routing:
                start_time = time.time()
                routing_info = self.query_router.route_query(retrieval_query, chat_history)
                if self.enable_profiling:
                    self._last_query_timings["query_routing"] = time.time() - start_time
            
            # Determine retrieval_k
            if routing_info:
                retrieval_k = routing_info["retrieval_k"]
            else:
                retrieval_k = self.rerank_top_k * 2  # Default: get more for reranking
            
            # Retrieve documents
            start_time = time.time()
            docs = self.hybrid_retriever.retrieve(retrieval_query, retrieval_k, apply_reranking=self.use_reranking)
            if self.enable_profiling:
                self._last_query_timings["document_retrieval"] = time.time() - start_time
            
            # Build context from retrieved documents
            context_parts = [format_docs(docs)]
            
            # Execute tools if needed and add results to context
            tool_results_text = ""
            tool_execution_results = None
            if self.use_tools:
                # Use pre-computed routing info instead of calling again
                query_classification = routing_info.get("classification") if routing_info else None
                
                if self.tool_manager.needs_tools(question, query_classification):
                    start_time = time.time()
                    tool_execution_results = self.tool_manager.execute_tools(
                        question,
                        query_classification=query_classification
                    )
                    if self.enable_profiling:
                        self._last_query_timings["tool_execution"] = time.time() - start_time
                    tool_results_text = self.tool_manager.format_tool_results(tool_execution_results)
                    if tool_results_text:
                        context_parts.append(f"\n\nEXTERNAL TOOLS RESULTS:{tool_results_text}")
            
            # Store retrieved docs, routing info, and tool results in instance for later retrieval
            # (since StrOutputParser will convert result to string)
            self._last_retrieved_docs = docs
            self._last_routing_info = routing_info
            self._last_tool_results = tool_execution_results
            
            result = {
                "context": "\n".join(context_parts),
                "input": question
            }
            
            # Add chat history if memory is enabled (always provide it, even if empty)
            if self.memory_manager.use_memory:
                result["chat_history"] = chat_history if chat_history else []
            
            return result
        
        # Create the retrieval chain using LCEL
        self.qa_chain = (
            RunnableLambda(retrieve_and_format)
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str, enable_validation: Optional[bool] = None) -> dict:
        """
        Query the RAG system with a question.
        
        OPTIMIZED VERSION: Reduces duplicate operations and LLM calls.
        
        Args:
            question: The question to ask
            enable_validation: Override validation setting (None = use default)
            
        Returns:
            Dictionary with 'result', 'source_documents', 'citations', 'validation', and 'routing'
        """
        if self.qa_chain is None:
            # Fall back directly to Tavily web search / General Answer if no documents are loaded
            fallback_res = self._generate_general_fallback_answer(question)
            answer = fallback_res["result"]
            if self.memory_manager.use_memory:
                self.memory_manager.add_message(question, answer)
            return {
                "result": answer,
                "source_documents": [],
                "citations": fallback_res["citations"],
                "is_general_answer": True
            }
        
        total_start = time.time()
        self._last_query_timings = {}
        
        # Get chat history if memory is enabled
        chat_history = self.memory_manager.get_chat_history() if self.memory_manager.use_memory else None
        
        # OPTIMIZATION: Pre-compute routing and rewriting ONCE, pass to chain
        routing_info = None
        if self.use_query_routing:
            start_time = time.time()
            routing_info = self.query_router.route_query(question, chat_history)
            if self.enable_profiling:
                self._last_query_timings["query_routing"] = time.time() - start_time
        
        # Rewrite query if enabled
        rewritten_query = None
        retrieval_query = question
        if self.use_query_rewriting:
            start_time = time.time()
            rewritten_query = self.query_rewriter.rewrite_query(question, chat_history)
            if self.enable_profiling:
                self._last_query_timings["query_rewriting"] = time.time() - start_time
            retrieval_query = rewritten_query
        
        # Prepare input for chain with pre-computed values
        if self.memory_manager.use_memory:
            input_dict = {
                "input": question,
                "chat_history": chat_history if chat_history else [],
                "_retrieval_query": retrieval_query,  # Pass pre-computed
                "_routing_info": routing_info  # Pass pre-computed
            }
        else:
            input_dict = {
                "input": question,
                "_retrieval_query": retrieval_query,
                "_routing_info": routing_info
            }
        
        # Get the answer from the chain (includes retrieval)
        start_time = time.time()
        chain_result = self.qa_chain.invoke(input_dict)
        if self.enable_profiling:
            self._last_query_timings["llm_generation"] = time.time() - start_time
        
        # Chain result is a string (from StrOutputParser)
        answer = chain_result if isinstance(chain_result, str) else str(chain_result)
        
        # OPTIMIZATION: Reuse retrieved documents stored in retrieve_and_format
        source_documents = self._last_retrieved_docs if self._last_retrieved_docs else []
        
        if not source_documents:
            if routing_info:
                retrieval_k = routing_info["retrieval_k"]
            else:
                retrieval_k = self.rerank_top_k * 2
            start_time = time.time()
            source_documents = self.hybrid_retriever.retrieve(
                retrieval_query,
                retrieval_k,
                apply_reranking=self.use_reranking
            )
            if self.enable_profiling:
                self._last_query_timings["document_retrieval_fallback"] = time.time() - start_time
        
        # Check if the answer was not found in the documents
        answer_clean = answer.strip()
        is_unknown = (
            "NOT_FOUND_IN_DOCS" in answer_clean or
            answer_clean == "NOT_FOUND_IN_DOCS" or
            not source_documents or
            ("i do not know" in answer_clean.lower() and len(answer_clean) < 150) or
            ("i don't know" in answer_clean.lower() and len(answer_clean) < 150) or
            ("context does not" in answer_clean.lower() and len(answer_clean) < 180) or
            ("documents do not" in answer_clean.lower() and len(answer_clean) < 180)
        )
        
        is_general_answer = False
        if is_unknown:
            # Fall back to Tavily Web Search and generate a General Answer
            fallback_res = self._generate_general_fallback_answer(question)
            answer = fallback_res["result"]
            source_documents = []  # Discard local document sources since answer came from web
            citations = fallback_res["citations"]
            is_general_answer = True
        else:
            # Format document citations
            start_time = time.time()
            citations = self.citation_formatter.format_citations(source_documents)
            if self.enable_profiling:
                self._last_query_timings["citation_formatting"] = time.time() - start_time
        
        # Reuse routing info from chain (stored in self._last_routing_info)
        if not routing_info:
            routing_info = self._last_routing_info
        
        # Save to memory if enabled
        if self.memory_manager.use_memory:
            self.memory_manager.add_message(question, answer)
        
        # OPTIMIZATION: Make validation optional and async-friendly
        validation = None
        should_validate = enable_validation if enable_validation is not None else self.use_answer_validation
        if should_validate and source_documents:
            start_time = time.time()
            validation = self.validator.validate_answer(answer, source_documents)
            if self.enable_profiling:
                self._last_query_timings["answer_validation"] = time.time() - start_time
        
        # Get tool execution results from chain (stored in self._last_tool_results)
        tool_execution_results = self._last_tool_results
        
        # Build result dictionary
        result = {
            "result": answer,
            "source_documents": source_documents,
            "citations": citations,
            "is_general_answer": is_general_answer
        }
        
        # Add optional fields
        if validation:
            result["validation"] = validation
        if rewritten_query and rewritten_query != question:
            result["rewritten_query"] = rewritten_query
        if routing_info:
            result["routing"] = routing_info["classification"]
        if tool_execution_results and tool_execution_results.get("tools_used"):
            result["tools_used"] = tool_execution_results["tools_used"]
            result["tool_results"] = tool_execution_results["results"]
        
        # Add performance metrics if profiling is enabled
        if self.enable_profiling:
            total_time = time.time() - total_start
            self._last_query_timings["total"] = total_time
            result["_timings"] = self._last_query_timings.copy()
            print(f"\n⏱️  Query Performance:")
            print(f"   Total: {total_time:.2f}s")
            for op, duration in sorted(self._last_query_timings.items(), key=lambda x: x[1], reverse=True):
                if op != "total":
                    pct = (duration / total_time * 100) if total_time > 0 else 0
                    print(f"   {op}: {duration:.2f}s ({pct:.1f}%)")
        
        return result

    def _generate_general_fallback_answer(self, question: str) -> Dict[str, Any]:
        """
        Generate a general answer using Tavily web search (or general LLM knowledge).
        Citations are attributed to 'General Answer'.
        """
        web_results = []
        provider_name = "Tavily"
        
        # Check if web search tool is available from tool manager
        web_tool = None
        if self.tool_manager and hasattr(self.tool_manager, "tools"):
            web_tool = self.tool_manager.tools.get("web_search")
        
        if web_tool is None:
            try:
                try:
                    from .tools import WebSearchTool
                except ImportError:
                    from src.tools import WebSearchTool
                web_tool = WebSearchTool(
                    provider=getattr(self, "web_search_provider", None) or "tavily",
                    api_key=os.getenv("TAVILY_API_KEY")
                )
            except Exception as e:
                print(f"Could not initialize WebSearchTool: {e}")
                web_tool = None
                
        if web_tool:
            try:
                search_res = web_tool.execute(question, max_results=4)
                if search_res.get("success") and search_res.get("results"):
                    web_results = search_res["results"]
                    provider_name = getattr(web_tool, "provider", "Tavily").capitalize()
            except Exception as e:
                print(f"Web search fallback notice: {e}")

        if web_results:
            web_context_lines = []
            for i, r in enumerate(web_results, 1):
                title = r.get("title", f"Web Source {i}")
                url = r.get("url", "")
                snippet = r.get("content") or r.get("snippet", "")
                web_context_lines.append(f"[{i}] {title}\nURL: {url}\nContent: {snippet}")
            web_context = "\n\n".join(web_context_lines)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a helpful assistant. The requested question was NOT found in the user's uploaded documents.\n"
                    "Provide a thorough, clear, and accurate general answer based on the following web search results.\n\n"
                    f"Web Search Results:\n{web_context}"
                )),
                ("human", "{question}")
            ])
            answer = (prompt | self.llm | StrOutputParser()).invoke({"question": question})
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a helpful assistant. The requested question was NOT found in the user's uploaded documents.\n"
                    "Provide a comprehensive, accurate general answer based on your general knowledge.\n"
                    "Be direct, structured, and informative."
                )),
                ("human", "{question}")
            ])
            answer = (prompt | self.llm | StrOutputParser()).invoke({"question": question})
            
        citations = self.citation_formatter.format_web_citations(web_results, provider=provider_name)
        return {
            "result": answer,
            "citations": citations,
            "web_results": web_results
        }
    
    def load_from_files(self, file_paths: List[str]) -> None:
        """
        Convenience method to load documents from files and add them to the vector store.
        
        Args:
            file_paths: List of file paths to load
        """
        documents = self.load_documents(file_paths)
        self.add_documents(documents)

    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory_manager.clear()
    
    def get_memory_summary(self) -> Optional[str]:
        """
        Get a summary of the conversation history.
        
        Returns:
            Summary string if memory is enabled, None otherwise
        """
        return self.memory_manager.get_memory_summary()

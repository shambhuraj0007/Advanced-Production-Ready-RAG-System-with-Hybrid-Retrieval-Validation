"""
Retriever Module for RAG System
Handles document retrieval including hybrid search, BM25, and re-ranking.
"""

from typing import List, Optional
from langchain_core.documents import Document
import re
import numpy as np

# Try to import BM25 for keyword search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# Try to import sentence-transformers for re-ranking
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False


class HybridRetriever:
    """
    Handles hybrid retrieval combining vector search and BM25 keyword search.
    """
    
    def __init__(
        self,
        vector_retriever,
        use_hybrid_search: bool = True,
        hybrid_search_alpha: float = 0.5,
        use_reranking: bool = True,
        rerank_top_k: int = 3,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_retriever: Vector store retriever
            use_hybrid_search: Whether to use hybrid search
            hybrid_search_alpha: Weight for vector search (0.0=only BM25, 1.0=only vector)
            use_reranking: Whether to use re-ranking
            rerank_top_k: Number of top documents after re-ranking
            reranker_model: Model name for cross-encoder reranker
        """
        self.vector_retriever = vector_retriever
        self.use_hybrid_search = use_hybrid_search and BM25_AVAILABLE
        self.hybrid_search_alpha = hybrid_search_alpha
        self.use_reranking = use_reranking and RERANKER_AVAILABLE
        self.rerank_top_k = rerank_top_k
        
        # BM25 index (built when documents are added)
        self.bm25_index = None
        self.document_chunks = []
        
        # Initialize re-ranker if enabled
        if self.use_reranking:
            try:
                self.reranker = CrossEncoder(reranker_model)
            except Exception as e:
                print(f"Warning: Failed to initialize re-ranker: {e}. Disabling re-ranking.")
                self.use_reranking = False
                self.reranker = None
        else:
            self.reranker = None
    
    def build_bm25_index(self, documents: List[Document]) -> None:
        """
        Build BM25 index for keyword search.
        
        Args:
            documents: List of Document objects to index
        """
        if not self.use_hybrid_search or not BM25_AVAILABLE:
            return
        
        # Store document chunks
        self.document_chunks = documents
        
        # Tokenize documents for BM25
        tokenized_docs = []
        for doc in documents:
            # Simple tokenization: split by whitespace and lowercase
            tokens = re.findall(r'\w+', doc.page_content.lower())
            tokenized_docs.append(tokens)
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)
        print(f"BM25 index built with {len(documents)} documents.")
    
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Re-rank documents using a cross-encoder model.
        
        Args:
            query: The search query
            documents: List of documents to re-rank
            
        Returns:
            Re-ranked list of documents (top k)
        """
        if not self.use_reranking or self.reranker is None or not documents:
            return documents[:self.rerank_top_k] if documents else []
        
        # Prepare pairs for cross-encoder: (query, document)
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get scores from cross-encoder
        scores = self.reranker.predict(pairs)
        
        # Sort documents by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k documents with confidence score attached
        import math
        reranked_docs = []
        for doc, score in scored_docs[:self.rerank_top_k]:
            # Convert cross-encoder logit to confidence (0.0 to 1.0) using sigmoid
            try:
                conf = 1.0 / (1.0 + math.exp(-float(score)))
            except Exception:
                conf = max(0.0, min(1.0, float(score)))
            doc.metadata["confidence_score"] = round(conf, 3)
            doc.metadata["relevance_score"] = round(float(score), 3)
            reranked_docs.append(doc)
        
        return reranked_docs
    
    def hybrid_retrieve(self, query: str, k: int) -> List[Document]:
        """
        Hybrid retrieval combining vector search and BM25 keyword search.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        if not self.use_hybrid_search or self.bm25_index is None:
            # Fall back to vector search only
            return self.vector_retriever.invoke(query)
        
        # Vector search
        vector_docs = self.vector_retriever.invoke(query)
        
        # BM25 keyword search
        query_tokens = re.findall(r'\w+', query.lower())
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top k BM25 results
        # Ensure we work with numpy arrays properly
        if isinstance(bm25_scores, np.ndarray):
            top_bm25_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: float(bm25_scores[i]),
                reverse=True
            )[:k * 2]  # Get more candidates for merging
            # Filter with explicit scalar conversion
            bm25_docs = [
                self.document_chunks[i] 
                for i in top_bm25_indices 
                if float(bm25_scores[i]) > 0
            ]
        else:
            top_bm25_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:k * 2]  # Get more candidates for merging
            bm25_docs = [self.document_chunks[i] for i in top_bm25_indices if bm25_scores[i] > 0]
        
        # Combine and deduplicate results
        # Create a map of document content to document for deduplication
        doc_map = {}
        vector_scores = {}
        bm25_scores_map = {}
        
        # Add vector search results
        for i, doc in enumerate(vector_docs):
            content_key = doc.page_content[:100]  # Use first 100 chars as key
            doc_map[content_key] = doc
            # Normalize vector score (assume higher is better, use position as proxy)
            vector_scores[content_key] = (len(vector_docs) - i) / len(vector_docs) if vector_docs else 0
        
        # Add BM25 results
        # Calculate max_bm25 once outside the loop to avoid numpy boolean ambiguity
        if len(bm25_scores) > 0:
            if isinstance(bm25_scores, np.ndarray):
                max_bm25 = float(np.max(bm25_scores))
            else:
                max_bm25 = float(max(bm25_scores))
        else:
            max_bm25 = 1.0
        
        for i, doc in enumerate(bm25_docs):
            content_key = doc.page_content[:100]
            if content_key not in doc_map:
                doc_map[content_key] = doc
            # Normalize BM25 score
            score_idx = top_bm25_indices[i]
            if isinstance(bm25_scores, np.ndarray):
                score_value = float(bm25_scores[score_idx])
            else:
                score_value = float(bm25_scores[score_idx])
            bm25_scores_map[content_key] = score_value / max_bm25 if max_bm25 > 0 else 0
        
        # Combine scores with alpha weighting
        combined_scores = {}
        all_keys = set(vector_scores.keys()) | set(bm25_scores_map.keys())
        
        for key in all_keys:
            vector_score = vector_scores.get(key, 0)
            bm25_score = bm25_scores_map.get(key, 0)
            # Alpha: weight for vector search, (1-alpha) for BM25
            combined_scores[key] = (
                self.hybrid_search_alpha * vector_score +
                (1 - self.hybrid_search_alpha) * bm25_score
            )
        
        # Sort by combined score and return top k
        sorted_keys = sorted(combined_scores.keys(), key=lambda k: combined_scores[k], reverse=True)
        hybrid_docs = []
        for key in sorted_keys[:k]:
            doc = doc_map[key]
            conf = float(combined_scores[key])
            doc.metadata["confidence_score"] = round(max(0.0, min(1.0, conf)), 3)
            doc.metadata["relevance_score"] = round(conf, 3)
            hybrid_docs.append(doc)
        
        return hybrid_docs
    
    def retrieve(self, query: str, k: int, apply_reranking: bool = True) -> List[Document]:
        """
        Main retrieval method that applies hybrid search and re-ranking.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            apply_reranking: Whether to apply re-ranking
            
        Returns:
            List of retrieved and re-ranked documents
        """
        # Retrieve documents
        if self.use_hybrid_search:
            docs = self.hybrid_retrieve(query, k * 2)  # Get more for reranking
        else:
            docs = self.vector_retriever.invoke(query)
        
        # Apply re-ranking if enabled
        if apply_reranking and self.use_reranking:
            docs = self.rerank_documents(query, docs)
        
        return docs[:k]  # Return top k


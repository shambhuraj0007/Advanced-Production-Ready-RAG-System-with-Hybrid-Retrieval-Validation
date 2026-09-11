"""
Query Rewriter for RAG System
Handles query rewriting and expansion for better retrieval.
"""

from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
import re


class QueryRewriter:
    """
    Rewrites and expands queries to improve retrieval quality.
    """
    
    def __init__(self, llm, use_rewriting: bool = True):
        """
        Initialize the query rewriter.
        
        Args:
            llm: Language model for query rewriting
            use_rewriting: Whether to enable query rewriting
        """
        self.llm = llm
        self.use_rewriting = use_rewriting
    
    def rewrite_query(self, query: str, chat_history: Optional[List] = None) -> str:
        """
        Rewrite query to improve retrieval quality.
        
        Args:
            query: Original query
            chat_history: Optional conversation history for context
            
        Returns:
            Rewritten query
        """
        if not self.use_rewriting:
            return query
        
        # Create a prompt for query rewriting
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query rewriting assistant. Your task is to rewrite the user's question 
to make it more effective for document retrieval. 

Guidelines:
1. Keep the core intent and meaning
2. Expand abbreviations and acronyms if helpful
3. Add relevant context if the question is ambiguous
4. Make it more specific if too vague
5. Keep it concise and focused

Return ONLY the rewritten query, nothing else."""),
            ("human", "Original query: {query}")
        ])
        
        # Add chat history context if available
        if chat_history and len(chat_history) > 0:
            # Get last few messages for context
            recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
            context = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in recent_history
            ])
            rewrite_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a query rewriting assistant. Your task is to rewrite the user's question 
to make it more effective for document retrieval. 

Recent conversation context:
{context}

Guidelines:
1. Keep the core intent and meaning
2. Expand abbreviations and acronyms if helpful
3. Add relevant context if the question is ambiguous
4. Make it more specific if too vague
5. Keep it concise and focused

Return ONLY the rewritten query, nothing else."""),
                ("human", "Original query: {query}")
            ])
            rewritten = (rewrite_prompt | self.llm | StrOutputParser()).invoke({
                "query": query,
                "context": context
            })
        else:
            rewritten = (rewrite_prompt | self.llm | StrOutputParser()).invoke({"query": query})
        
        # Clean up the response (remove quotes, extra whitespace)
        rewritten = rewritten.strip().strip('"').strip("'")
        
        # Fallback to original if rewriting failed or produced empty result
        if not rewritten or len(rewritten) < 3:
            return query
        
        return rewritten
    
    def expand_query(self, query: str) -> List[str]:
        """
        Generate multiple query variations for better retrieval.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations
        """
        if not self.use_rewriting:
            return [query]
        
        expand_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query expansion assistant. Generate 2-3 alternative phrasings 
of the user's query that might help retrieve relevant documents.

Guidelines:
1. Use synonyms and related terms
2. Vary the phrasing while keeping the same intent
3. Include technical terms and common alternatives
4. Keep each variation concise

Return the variations as a numbered list, one per line."""),
            ("human", "Query: {query}")
        ])
        
        try:
            expanded = (expand_prompt | self.llm | StrOutputParser()).invoke({"query": query})
            
            # Parse the numbered list
            variations = []
            for line in expanded.split('\n'):
                line = line.strip()
                # Remove numbering (1., 2., etc.)
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                if line and len(line) > 3:
                    variations.append(line)
            
            # Include original query and variations
            all_queries = [query] + variations[:3]  # Limit to 3 variations
            return all_queries
        except Exception as e:
            print(f"Warning: Query expansion failed: {e}. Using original query.")
            return [query]


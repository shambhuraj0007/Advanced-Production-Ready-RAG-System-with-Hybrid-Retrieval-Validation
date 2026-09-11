"""
Query Router for RAG System
Handles query classification and routing to appropriate retrieval strategies.
"""

from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import json
import re


class QueryRouter:
    """
    Routes queries to appropriate retrieval strategies based on query type and complexity.
    """
    
    def __init__(self, llm, use_routing: bool = True):
        """
        Initialize the query router.
        
        Args:
            llm: Language model for query classification
            use_routing: Whether to enable query routing
        """
        self.llm = llm
        self.use_routing = use_routing
        
        # Query type categories
        self.query_types = {
            "factual": "Simple factual questions requiring direct answers",
            "analytical": "Questions requiring analysis, comparison, or reasoning",
            "creative": "Open-ended or creative questions",
            "multi_hop": "Questions requiring information from multiple sources",
            "definition": "Questions asking for definitions or explanations"
        }
    
    def classify_query(self, query: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
        """
        Classify the query to determine its type and complexity.
        
        Args:
            query: The query to classify
            chat_history: Optional conversation history
            
        Returns:
            Dictionary with classification results
        """
        if not self.use_routing:
            return {
                "query_type": "factual",
                "complexity": "simple",
                "requires_multi_hop": False,
                "recommended_k": 3,
                "use_hybrid": True
            }
        
        # Use a format string approach to avoid template variable conflicts
        system_prompt = """You are a query classification assistant. Analyze the user's query and classify it.

Query Types:
- factual: Simple factual questions (e.g., "What is X?", "When did Y happen?")
- analytical: Questions requiring analysis or comparison (e.g., "Compare X and Y", "Why does X happen?")
- creative: Open-ended or creative questions (e.g., "How would you design X?")
- multi_hop: Questions requiring information from multiple sources (e.g., "What is X and how does it relate to Y?")
- definition: Questions asking for definitions (e.g., "Define X", "What does X mean?")

Complexity Levels:
- simple: Straightforward questions with clear answers
- medium: Questions requiring some reasoning or context
- complex: Questions requiring deep analysis or multiple steps

Respond ONLY with valid JSON, no other text. Use this exact format:
{{"query_type": "factual", "complexity": "simple", "requires_multi_hop": false, "recommended_k": 3, "use_hybrid": true}}"""

        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Query: {query}")
        ])
        
        try:
            result = (classification_prompt | self.llm | StrOutputParser()).invoke({"query": query})
            
            # Clean the result - remove markdown code blocks if present
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]  # Remove ```json
            elif result.startswith("```"):
                result = result[3:]  # Remove ```
            if result.endswith("```"):
                result = result[:-3]  # Remove closing ```
            result = result.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                classification = json.loads(json_str)
            else:
                # Try to parse the entire result as JSON
                classification = json.loads(result)
            
            # Validate and set defaults
            query_type = classification.get("query_type", "factual")
            if query_type not in self.query_types:
                query_type = "factual"
            
            complexity = classification.get("complexity", "simple")
            if complexity not in ["simple", "medium", "complex"]:
                complexity = "simple"
            
            recommended_k = int(classification.get("recommended_k", 3))
            recommended_k = max(3, min(10, recommended_k))  # Clamp between 3 and 10
            
            return {
                "query_type": query_type,
                "complexity": complexity,
                "requires_multi_hop": classification.get("requires_multi_hop", False),
                "recommended_k": recommended_k,
                "use_hybrid": classification.get("use_hybrid", True)
            }
        except Exception as e:
            print(f"Warning: Query classification failed: {e}. Using defaults.")
            return {
                "query_type": "factual",
                "complexity": "simple",
                "requires_multi_hop": False,
                "recommended_k": 3,
                "use_hybrid": True
            }
    
    def route_query(self, query: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
        """
        Route query to appropriate retrieval strategy.
        
        Args:
            query: The query to route
            chat_history: Optional conversation history
            
        Returns:
            Dictionary with routing decisions
        """
        classification = self.classify_query(query, chat_history)
        
        # Determine retrieval strategy based on classification
        routing = {
            "classification": classification,
            "retrieval_k": classification["recommended_k"],
            "use_hybrid_search": classification["use_hybrid"],
            "use_reranking": True,  # Always use reranking for better results
            "rerank_top_k": min(classification["recommended_k"], 5),  # Limit rerank top k
            "strategy": self._determine_strategy(classification)
        }
        
        return routing
    
    def _determine_strategy(self, classification: Dict[str, Any]) -> str:
        """
        Determine the best retrieval strategy based on classification.
        
        Args:
            classification: Query classification results
            
        Returns:
            Strategy name
        """
        query_type = classification["query_type"]
        complexity = classification["complexity"]
        requires_multi_hop = classification["requires_multi_hop"]
        
        if requires_multi_hop or complexity == "complex":
            return "multi_hop"  # May need multiple retrieval passes
        elif query_type == "analytical":
            return "analytical"  # May need more documents
        elif query_type == "factual" or query_type == "definition":
            return "factual"  # Standard retrieval
        else:
            return "standard"  # Default strategy


"""
Tools Module for RAG System
Provides external tools that can be used by the RAG system.
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import re
import json
import os


class BaseTool(ABC):
    """
    Base class for all tools.
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, input_data: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool.
        
        Args:
            input_data: Input data for the tool
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with tool results
        """
        pass
    
    def should_use(self, query: str) -> bool:
        """
        Determine if this tool should be used for the given query.
        
        Args:
            query: The user query
            
        Returns:
            True if tool should be used, False otherwise
        """
        # Simple keyword-based detection (can be overridden)
        keywords = getattr(self, 'keywords', [])
        query_lower = query.lower()
        return any(keyword.lower() in query_lower for keyword in keywords)


class CalculatorTool(BaseTool):
    """
    Calculator tool for mathematical operations.
    """
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs mathematical calculations and arithmetic operations"
        )
        self.keywords = ["calculate", "compute", "math", "arithmetic"]
    
    def should_use(self, query: str) -> bool:
        """Check if query genuinely contains a mathematical operation, avoiding false triggers on hyphens/slashes."""
        query_lower = query.lower()
        # Explicit math keywords with numbers
        if any(kw in query_lower for kw in ["calculate", "compute", "arithmetic"]):
            return True
        # Explicit arithmetic patterns like 10 + 5 or 20 * 3
        if re.search(r'\b\d+\s*[\+\*\/\=]\s*\d+\b', query) or re.search(r'\b\d+\s*-\s*\d+\b', query):
            return True
        return False
    
    def execute(self, input_data: str, **kwargs) -> Dict[str, Any]:
        """
        Execute calculation.
        
        Args:
            input_data: Mathematical expression or question
            
        Returns:
            Dictionary with calculation result
        """
        try:
            # Extract mathematical expressions from the query
            # Simple approach: look for expressions like "5 + 3" or "calculate 10 * 2"
            expression = self._extract_expression(input_data)
            
            if not expression:
                return {
                    "success": False,
                    "error": "No valid mathematical expression found",
                    "result": None
                }
            
            # Safety: Only allow basic math operations and numbers
            # Remove any potentially dangerous operations
            safe_expression = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            
            # Evaluate the expression
            result = eval(safe_expression)
            
            return {
                "success": True,
                "expression": expression,
                "result": result,
                "formatted_result": f"{expression} = {result}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
    
    def _extract_expression(self, query: str) -> Optional[str]:
        """
        Extract mathematical expression from query.
        
        Args:
            query: User query
            
        Returns:
            Mathematical expression or None
        """
        # Try to find expressions like "5 + 3" or "10 * 2"
        # Pattern: numbers with operators
        pattern = r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, query)
        
        if match:
            return match.group(0)
        
        # Try to extract numbers and operations separately
        # Look for phrases like "calculate X plus Y" or "X times Y"
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        if len(numbers) >= 2:
            # Try to find operators in text
            if 'plus' in query.lower() or '+' in query:
                return f"{numbers[0]} + {numbers[1]}"
            elif 'minus' in query.lower() or '-' in query:
                return f"{numbers[0]} - {numbers[1]}"
            elif 'times' in query.lower() or '*' in query or 'multiply' in query.lower():
                return f"{numbers[0]} * {numbers[1]}"
            elif 'divided by' in query.lower() or '/' in query or 'divide' in query.lower():
                return f"{numbers[0]} / {numbers[1]}"
        
        return None


class WebSearchTool(BaseTool):
    """
    Web search tool for getting real-time information.
    """
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "tavily"):
        """
        Initialize web search tool.
        
        Args:
            api_key: API key for search provider
            provider: Search provider ("tavily" or "duckduckgo")
        """
        super().__init__(
            name="web_search",
            description="Searches the web for current information and real-time data"
        )
        self.provider = provider.lower()
        self.api_key = api_key
        self.keywords = ["search", "current", "latest", "recent", "news", "today", "web", "internet", "online"]
        
        # Initialize search client if available
        self.search_client = None
        self._init_search_client()
    
    def _init_search_client(self):
        """Initialize the search client based on provider."""
        if self.provider == "tavily":
            try:
                from tavily import TavilyClient
                self.search_client = TavilyClient(api_key=self.api_key or os.getenv("TAVILY_API_KEY"))
            except ImportError:
                print("Warning: tavily-python not installed. Web search will use DuckDuckGo fallback.")
                self.provider = "duckduckgo"
        
        if self.provider == "duckduckgo":
            try:
                from duckduckgo_search import DDGS
                self.search_client = DDGS()
            except ImportError:
                print("Warning: duckduckgo-search not installed. Web search will be disabled.")
                self.search_client = None
    
    def execute(self, input_data: str, max_results: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Execute web search.
        
        Args:
            input_data: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        if not self.search_client:
            return {
                "success": False,
                "error": "Web search not available. Install tavily-python or duckduckgo-search.",
                "results": []
            }
        
        try:
            if self.provider == "tavily":
                results = self._search_tavily(input_data, max_results)
            else:  # duckduckgo
                results = self._search_duckduckgo(input_data, max_results)
            
            return {
                "success": True,
                "query": input_data,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    def _search_tavily(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Tavily API."""
        response = self.search_client.search(query, max_results=max_results)
        
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "snippet": item.get("content", "")[:200]
            })
        
        return results
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo."""
        results = []
        try:
            for result in self.search_client.text(query, max_results=max_results):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "content": result.get("body", ""),
                    "snippet": result.get("body", "")[:200]
                })
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
        
        return results


class ToolRegistry:
    """
    Registry for managing available tools.
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
    
    def register_tool(self, tool: BaseTool):
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self.tools.values())
    
    def select_tools(self, query: str) -> List[BaseTool]:
        """
        Select tools that should be used for a given query.
        
        Args:
            query: User query
            
        Returns:
            List of tools to use
        """
        selected = []
        for tool in self.tools.values():
            if tool.should_use(query):
                selected.append(tool)
        return selected
    
    def execute_tool(self, tool_name: str, input_data: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool
            input_data: Input data for the tool
            **kwargs: Additional keyword arguments
            
        Returns:
            Tool execution result
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "result": None
            }
        
        return tool.execute(input_data, **kwargs)


"""
Tool Manager for RAG System
Manages tool selection and execution based on queries.
"""

from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re

from .tools import ToolRegistry, BaseTool, CalculatorTool, WebSearchTool


class ToolManager:
    """
    Manages tool selection and execution for the RAG system.
    """
    
    def __init__(
        self,
        llm,
        use_tools: bool = True,
        enable_calculator: bool = True,
        enable_web_search: bool = True,
        web_search_provider: str = "tavily",
        web_search_api_key: Optional[str] = None,
            tool_selection_method: str = "router_aware"  # "keyword", "llm", "hybrid", or "router_aware" (recommended)
    ):
        """
        Initialize the tool manager.
        
        Args:
            llm: Language model for tool selection
            use_tools: Whether to enable tools
            enable_calculator: Whether to enable calculator tool
            enable_web_search: Whether to enable web search tool
            web_search_provider: Web search provider ("tavily" or "duckduckgo")
            web_search_api_key: API key for web search
        """
        self.llm = llm
        self.use_tools = use_tools
        self.tool_selection_method = tool_selection_method  # "keyword", "llm", or "hybrid"
        self.registry = ToolRegistry()
        
        # Register available tools
        if enable_calculator:
            self.registry.register_tool(CalculatorTool())
        
        if enable_web_search:
            self.registry.register_tool(WebSearchTool(
                api_key=web_search_api_key,
                provider=web_search_provider
            ))
    
    def needs_tools(self, query: str, query_classification: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if a query needs tools.
        
        Args:
            query: User query
            query_classification: Optional query classification from router
            
        Returns:
            True if tools are needed, False otherwise
        """
        if not self.use_tools:
            return False
        
        # Use classification hints if available
        if query_classification:
            # Check if query type suggests tool needs
            query_type = query_classification.get("query_type", "")
            complexity = query_classification.get("complexity", "")
            
            # Queries asking for current/recent info often need web search
            if "current" in query.lower() or "latest" in query.lower() or "recent" in query.lower():
                return True
            
            # Queries with math operations need calculator
            if any(op in query for op in ["calculate", "compute", "+", "-", "*", "/"]):
                return True
        
        # Check with selection method
        selected_tools = self.select_tools(query, query_classification=query_classification)
        return len(selected_tools) > 0
    
    def select_tools(
        self,
        query: str,
        query_classification: Optional[Dict[str, Any]] = None,
        use_llm_selection: Optional[bool] = None
    ) -> List[BaseTool]:
        """
        Select tools for a given query using the configured selection method.
        
        Args:
            query: User query
            query_classification: Optional query classification from router
            use_llm_selection: Override selection method (None = use default from config)
            
        Returns:
            List of selected tools
        """
        if not self.use_tools:
            return []
        
        # Determine selection method
        if use_llm_selection is None:
            method = self.tool_selection_method
        else:
            method = "llm" if use_llm_selection else "keyword"
        
        # Use appropriate selection method
        if method == "llm":
            return self._llm_select_tools(query)
        elif method == "hybrid":
            return self._hybrid_select_tools(query, query_classification)
        elif method == "router_aware":
            return self._router_aware_select_tools(query, query_classification)
        else:  # "keyword"
            return self.registry.select_tools(query)
    
    def _hybrid_select_tools(self, query: str, query_classification: Optional[Dict[str, Any]] = None) -> List[BaseTool]:
        """
        Hybrid tool selection: combines keyword detection with LLM-based selection.
        
        Strategy:
        1. Quick keyword check for obvious cases (fast path)
        2. LLM-based selection for ambiguous cases
        3. Use classification hints if available
        
        Args:
            query: User query
            query_classification: Optional query classification from router
            
        Returns:
            List of selected tools
        """
        # Step 1: Quick keyword-based detection for obvious cases
        keyword_tools = self.registry.select_tools(query)
        
        # If keywords clearly indicate tools, use them
        # (e.g., "calculate" → calculator, "current news" → web_search)
        if keyword_tools:
            # For unambiguous cases, trust keywords
            strong_indicators = {
                "calculator": ["calculate", "compute", "math", "+", "-", "*", "/", "plus", "minus", "times", "divided"],
                "web_search": ["current", "latest", "recent", "news", "today", "search the web", "online"]
            }
            
            query_lower = query.lower()
            unambiguous = False
            for tool_name, indicators in strong_indicators.items():
                if any(indicator in query_lower for indicator in indicators):
                    # Check if this tool was selected
                    if any(t.name == tool_name for t in keyword_tools):
                        unambiguous = True
                        break
            
            if unambiguous:
                return keyword_tools
        
        # Step 2: Use LLM for ambiguous or complex cases
        # Also use LLM if classification suggests tool needs but keywords didn't catch it
        if query_classification:
            query_type = query_classification.get("query_type", "")
            # Analytical or multi-hop queries might need tools even without keywords
            if query_type in ["analytical", "multi_hop"] and not keyword_tools:
                return self._llm_select_tools(query)
        
        # If keywords found something, try LLM to verify and add missing tools
        if keyword_tools and len(keyword_tools) > 0:
            llm_tools = self._llm_select_tools(query)
            # Combine results, preferring LLM selection (it's smarter)
            if llm_tools:
                return llm_tools
            return keyword_tools
        
        # Fallback to LLM if no clear keyword matches
        return self._llm_select_tools(query)
    
    def _router_aware_select_tools(self, query: str, query_classification: Optional[Dict[str, Any]] = None) -> List[BaseTool]:
        """
        Router-aware tool selection: uses query classification to intelligently select tools.
        
        This is the recommended method as it combines:
        1. Query routing classification (query type, complexity)
        2. Keyword detection for obvious cases
        3. LLM selection for ambiguous cases
        
        Args:
            query: User query
            query_classification: Query classification from QueryRouter
            
        Returns:
            List of selected tools
        """
        # If we have classification, use it to guide tool selection
        if query_classification:
            query_type = query_classification.get("query_type", "")
            complexity = query_classification.get("complexity", "")
            requires_multi_hop = query_classification.get("requires_multi_hop", False)
            
            # Rule-based tool selection based on classification
            selected_tools = []
            
            # Check for calculator needs
            calc_tool = self.registry.get_tool("calculator")
            if calc_tool:
                # Math-related queries or queries with numbers and operators
                if any(op in query.lower() for op in ["calculate", "compute", "math", "+", "-", "*", "/", "plus", "times", "divided"]):
                    selected_tools.append(calc_tool)
                # Complex analytical queries might need calculations
                elif query_type == "analytical" and any(char.isdigit() for char in query):
                    selected_tools.append(calc_tool)
            
            # Check for web search needs
            web_tool = self.registry.get_tool("web_search")
            if web_tool:
                # Queries asking for current/recent information
                current_keywords = ["current", "latest", "recent", "news", "today", "now", "up-to-date"]
                if any(kw in query.lower() for kw in current_keywords):
                    selected_tools.append(web_tool)
                # Multi-hop queries might need external information
                elif requires_multi_hop and query_type in ["analytical", "multi_hop"]:
                    selected_tools.append(web_tool)
                # Complex queries without clear document context
                elif complexity == "complex" and query_type == "analytical":
                    # Use LLM to decide if web search is needed
                    llm_tools = self._llm_select_tools(query)
                    if web_tool in llm_tools:
                        selected_tools.append(web_tool)
            
            # If classification-based selection found tools, return them
            if selected_tools:
                return selected_tools
        
        # Fallback to hybrid selection if no classification or no tools found
        return self._hybrid_select_tools(query, query_classification)
    
    def _llm_select_tools(self, query: str) -> List[BaseTool]:
        """
        Use LLM to select appropriate tools.
        
        Args:
            query: User query
            
        Returns:
            List of selected tools
        """
        available_tools = self.registry.get_all_tools()
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in available_tools
        ])
        
        selection_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a tool selection assistant. Determine which tools should be used for the user's query.

Available tools:
{tool_descriptions}

Respond with a JSON list of tool names that should be used, or an empty list [] if no tools are needed.
Example: ["calculator"] or ["web_search"] or ["calculator", "web_search"] or []"""),
            ("human", "Query: {query}")
        ])
        
        try:
            result = (selection_prompt | self.llm | StrOutputParser()).invoke({"query": query})
            
            # Parse JSON from response
            json_match = re.search(r'\[.*?\]', result)
            if json_match:
                tool_names = json.loads(json_match.group())
            else:
                tool_names = json.loads(result)
            
            # Get tools by name
            selected = []
            for name in tool_names:
                tool = self.registry.get_tool(name)
                if tool:
                    selected.append(tool)
            
            return selected
        except Exception as e:
            print(f"Warning: LLM tool selection failed: {e}. Using keyword-based selection.")
            return self.registry.select_tools(query)
    
    def execute_tools(
        self,
        query: str,
        selected_tools: Optional[List[BaseTool]] = None,
        query_classification: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute selected tools for a query.
        
        Args:
            query: User query
            selected_tools: List of tools to execute (if None, will select automatically)
            query_classification: Optional query classification for better tool selection
            
        Returns:
            Dictionary with tool execution results
        """
        if not self.use_tools:
            return {
                "tools_used": [],
                "results": {}
            }
        
        if selected_tools is None:
            selected_tools = self.select_tools(query, query_classification=query_classification)
        
        results = {}
        tools_used = []
        
        for tool in selected_tools:
            try:
                result = tool.execute(query)
                results[tool.name] = result
                if result.get("success"):
                    tools_used.append(tool.name)
            except Exception as e:
                results[tool.name] = {
                    "success": False,
                    "error": str(e),
                    "result": None
                }
        
        return {
            "tools_used": tools_used,
            "results": results
        }
    
    def format_tool_results(self, tool_results: Dict[str, Any]) -> str:
        """
        Format tool results for inclusion in context.
        
        Args:
            tool_results: Dictionary with tool execution results
            
        Returns:
            Formatted string with tool results
        """
        formatted_parts = []
        
        for tool_name, result in tool_results.get("results", {}).items():
            if result.get("success"):
                formatted_parts.append(f"\n{tool_name.upper()} RESULTS:")
                
                if tool_name == "calculator":
                    formatted_parts.append(result.get("formatted_result", str(result.get("result", ""))))
                
                elif tool_name == "web_search":
                    search_results = result.get("results", [])
                    for i, item in enumerate(search_results[:3], 1):  # Top 3 results
                        formatted_parts.append(f"\n{i}. {item.get('title', '')}")
                        formatted_parts.append(f"   {item.get('snippet', '')}")
                        formatted_parts.append(f"   Source: {item.get('url', '')}")
                
                else:
                    formatted_parts.append(str(result.get("result", "")))
        
        return "\n".join(formatted_parts) if formatted_parts else ""


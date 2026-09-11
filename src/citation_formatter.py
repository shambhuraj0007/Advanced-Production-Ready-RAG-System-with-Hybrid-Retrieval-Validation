"""
Citation Formatter for RAG System
Formats source citations with enhanced metadata.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
import os


class CitationFormatter:
    """
    Formats source documents with enhanced citation information.
    """
    
    def format_citations(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Format source documents with enhanced citation information.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of citation dictionaries with metadata
        """
        citations = []
        for i, doc in enumerate(documents, 1):
            # Get metadata (handle case where metadata might be None or empty)
            metadata = doc.metadata if doc.metadata else {}
            
            # Try multiple metadata keys to find file name
            # Priority: file_name > source (extract basename) > file_path (extract basename)
            file_name = None
            if metadata.get("file_name"):
                file_name = metadata["file_name"]
            elif metadata.get("source"):
                file_name = os.path.basename(metadata["source"])
            elif metadata.get("file_path"):
                file_name = os.path.basename(metadata["file_path"])
            
            # If still None or empty, try to extract from any string value in metadata
            if not file_name or file_name == "Unknown":
                for key in ["source", "file_path", "file_name"]:
                    value = metadata.get(key)
                    if value and isinstance(value, str) and value not in ["Unknown", "N/A", ""]:
                        file_name = os.path.basename(value)
                        break
            
            # Final fallback
            if not file_name or file_name == "Unknown":
                file_name = "Unknown"
            else:
                # If file_name is still a full path, extract just the filename
                if os.path.sep in file_name:
                    file_name = os.path.basename(file_name)
            
            # Get file path
            file_path = (
                metadata.get("file_path") or
                metadata.get("source") or
                "Unknown"
            )
            
            # Get page number - handle various formats
            page = metadata.get("page") or metadata.get("page_number")
            
            # Convert page to string and handle numeric values
            if page is None:
                page = "N/A"
            elif isinstance(page, (int, float)):
                page = str(int(page))
            elif isinstance(page, str):
                # Clean up string page numbers
                page = page.strip()
                if page in ["N/A", "Unknown", "", "None"]:
                    page = "N/A"
            else:
                page = str(page) if page else "N/A"
            
            citation = {
                "index": i,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "file_name": file_name,
                "file_path": file_path,
                "page": page,
                "source": metadata.get("source", file_path),
            }
            
            # Add any additional metadata
            if "section" in doc.metadata:
                citation["section"] = doc.metadata["section"]
            if "title" in doc.metadata:
                citation["title"] = doc.metadata["title"]
            
            # Format citation string
            citation["citation_string"] = self._create_citation_string(citation)
            
            citations.append(citation)
        
        return citations
    
    def _create_citation_string(self, citation: Dict[str, Any]) -> str:
        """
        Create a formatted citation string from citation metadata.
        
        Args:
            citation: Citation dictionary
            
        Returns:
            Formatted citation string
        """
        parts = []
        
        # Add file name
        if citation.get("file_name") and citation["file_name"] != "Unknown":
            parts.append(citation["file_name"])
        
        # Add page number if available
        if citation.get("page") and citation["page"] != "N/A" and citation["page"] != "Unknown":
            parts.append(f"page {citation['page']}")
        
        # Add section if available
        if citation.get("section"):
            parts.append(f"section: {citation['section']}")
        
        if parts:
            return " (" + ", ".join(parts) + ")"
        else:
            return f" (Source: {citation.get('source', 'Unknown')})"

    def format_web_citations(self, web_results: List[Dict[str, Any]], provider: str = "Tavily") -> List[Dict[str, Any]]:
        """
        Format web search results as citations for a General Answer.
        """
        citations = []
        if not web_results:
            return [{
                "index": 1,
                "type": "general",
                "file_name": "General Answer",
                "source": "General Knowledge",
                "url": "",
                "content": "Answer generated from general knowledge.",
                "citation_string": " (Source: General Answer)"
            }]
            
        for i, item in enumerate(web_results, 1):
            title = item.get("title") or f"Web Source {i}"
            url = item.get("url", "")
            content = item.get("content") or item.get("snippet", "")
            citations.append({
                "index": i,
                "type": "web",
                "file_name": "General Answer",
                "source": f"{provider}: {title}",
                "url": url,
                "content": content[:200] + "..." if len(content) > 200 else content,
                "citation_string": f" (Source: General Answer - {title})" if not url else f" (Source: General Answer - {url})"
            })
        return citations


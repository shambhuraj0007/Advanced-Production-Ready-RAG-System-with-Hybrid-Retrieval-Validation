"""
Answer Validator for RAG System
Validates answers against source documents to detect hallucinations.
"""

from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import json
import re


class AnswerValidator:
    """
    Validates answers against source documents.
    """
    
    def __init__(self, llm, use_validation: bool = True):
        """
        Initialize the answer validator.
        
        Args:
            llm: Language model for answer validation
            use_validation: Whether to enable answer validation
        """
        self.llm = llm
        self.use_validation = use_validation
    
    def validate_answer(self, answer: str, source_documents: List[Document]) -> Dict[str, Any]:
        """
        Validate if the answer is supported by the source documents.
        
        Args:
            answer: Generated answer
            source_documents: Retrieved source documents
            
        Returns:
            Dictionary with validation results including confidence score
        """
        if not self.use_validation or not source_documents:
            return {
                "is_valid": True,
                "confidence": 0.5,
                "supporting_evidence": [],
                "warnings": []
            }
        
        # Combine source documents
        source_text = "\n\n".join([doc.page_content for doc in source_documents])
        
        # Create validation prompt
        validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an answer validation assistant. Your task is to evaluate whether 
the given answer is well-supported by the provided source documents.

Analyze:
1. Are the key claims in the answer present in the sources?
2. Are there any unsupported statements or hallucinations?
3. How confident can we be in this answer based on the sources?

Respond in JSON format with:
- "is_valid": true/false
- "confidence": 0.0 to 1.0 (1.0 = fully supported, 0.0 = not supported)
- "supporting_evidence": list of key phrases from sources that support the answer
- "warnings": list of any unsupported claims or concerns"""),
            ("human", """Answer to validate:
{answer}

Source documents:
{sources}

Provide your analysis in JSON format.""")
        ])
        
        try:
            validation_result = (validation_prompt | self.llm | StrOutputParser()).invoke({
                "answer": answer,
                "sources": source_text[:3000]  # Limit source text length
            })
            
            # Try to parse JSON from the response
            # Extract JSON from response (might have extra text)
            json_match = re.search(r'\{.*\}', validation_result, re.DOTALL)
            if json_match:
                validation_data = json.loads(json_match.group())
            else:
                # Fallback: try to parse the whole response
                validation_data = json.loads(validation_result)
            
            # Ensure required fields
            result = {
                "is_valid": validation_data.get("is_valid", True),
                "confidence": float(validation_data.get("confidence", 0.5)),
                "supporting_evidence": validation_data.get("supporting_evidence", []),
                "warnings": validation_data.get("warnings", [])
            }
            
            return result
        except Exception as e:
            print(f"Warning: Answer validation failed: {e}. Using simple validation.")
            # Fallback: simple keyword-based validation
            return self._simple_validation(answer, source_documents)
    
    def _simple_validation(self, answer: str, source_documents: List[Document]) -> Dict[str, Any]:
        """
        Simple keyword-based validation fallback.
        
        Args:
            answer: Generated answer
            source_documents: Retrieved source documents
            
        Returns:
            Dictionary with validation results
        """
        source_text = " ".join([doc.page_content.lower() for doc in source_documents])
        answer_lower = answer.lower()
        
        # Extract key terms from answer (simple approach)
        answer_terms = set(re.findall(r'\b\w{4,}\b', answer_lower))  # Words with 4+ chars
        
        # Check how many terms appear in sources
        matching_terms = [term for term in answer_terms if term in source_text]
        
        if len(answer_terms) == 0:
            confidence = 0.5
        else:
            confidence = len(matching_terms) / len(answer_terms)
        
        return {
            "is_valid": confidence > 0.3,  # At least 30% of terms match
            "confidence": confidence,
            "supporting_evidence": matching_terms[:5],  # Top 5 matching terms
            "warnings": [] if confidence > 0.5 else ["Low confidence: some terms not found in sources"]
        }


"""
Example usage of the Simple RAG system.
"""

from src.rag_system import SimpleRAG
import os

def main():
    # Initialize RAG system (all settings auto-load from .env, or can be passed here)
    rag = SimpleRAG()
    
    # Example: Load documents from files
    # Replace these paths with your actual document paths
    file_paths = [
        # "documents/sample.pdf",
        # "documents/sample.txt",
    ]
    
    # If you have documents, uncomment and add paths:
    # rag.load_from_files(file_paths)
    
    # For demonstration, let's create a simple text document
    if not os.path.exists("documents"):
        os.makedirs("documents")
    
    # Create a sample document
    sample_text = """
    LangChain is a framework for developing applications powered by language models.
    It enables applications to be context-aware and reason about how to use a tool.
    RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of
    relevant information with generation of responses. This allows LLMs to access
    external knowledge bases and provide more accurate and up-to-date information.
    """
    
    with open("documents/sample.txt", "w") as f:
        f.write(sample_text)
    
    # Load the sample document
    print("Loading documents...")
    folder_path = "documents/"
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filenames.append(os.path.join(root, file))  # full path
    print(filenames)
    rag.load_from_files(filenames)
    
    # Query the system
    print("\n" + "="*50)
    print("Querying the RAG system...")
    print("="*50 + "\n")
    
    # Demonstrate conversational memory with follow-up questions
    print("\n" + "="*50)
    print("Demonstrating Conversational Memory")
    print("="*50 + "\n")
    
    # First question
    question1 = "What is tabbed tray?"
    print(f"Question 1: {question1}")
    result1 = rag.query(question1)
    
    # Show query routing info if available
    if 'routing' in result1:
        routing = result1['routing']
        print(f"Query Type: {routing.get('query_type', 'unknown')}")
        print(f"Complexity: {routing.get('complexity', 'unknown')}")
        print(f"Recommended K: {routing.get('recommended_k', 'unknown')}")
    
    # Show rewritten query if available
    if 'rewritten_query' in result1 and result1['rewritten_query'] != question1:
        print(f"Rewritten query: {result1['rewritten_query']}")
    
    print(f"Answer: {result1['result']}")
    print(f"Source documents: {len(result1.get('source_documents', []))} found")
    
    # Display validation results if available
    if 'validation' in result1:
        validation = result1['validation']
        print(f"\nAnswer Validation:")
        print(f"  Valid: {validation.get('is_valid', 'Unknown')}")
        print(f"  Confidence: {validation.get('confidence', 0):.2%}")
        if validation.get('warnings'):
            print(f"  Warnings: {', '.join(validation['warnings'])}")
        if validation.get('supporting_evidence'):
            print(f"  Supporting evidence: {len(validation['supporting_evidence'])} items found")
    
    # Display enhanced citations
    if 'citations' in result1 and result1['citations']:
        print("\nCitations:")
        for citation in result1['citations']:
            print(f"  [{citation['index']}] {citation['citation_string']}")
            print(f"      File: {citation['file_name']}")
            if citation.get('page') and citation['page'] != 'N/A':
                print(f"      Page: {citation['page']}")
    print("-" * 50 + "\n")
    
    # Follow-up question that references the previous conversation
    question2 = "Tell me more about it"
    print(f"Question 2 (follow-up): {question2}")
    result2 = rag.query(question2)
    
    # Show rewritten query if available
    if 'rewritten_query' in result2 and result2['rewritten_query'] != question2:
        print(f"Rewritten query: {result2['rewritten_query']}")
    
    print(f"Answer: {result2['result']}")
    print(f"Source documents: {len(result2.get('source_documents', []))} found")
    
    # Display validation results if available
    if 'validation' in result2:
        validation = result2['validation']
        print(f"\nAnswer Validation:")
        print(f"  Valid: {validation.get('is_valid', 'Unknown')}")
        print(f"  Confidence: {validation.get('confidence', 0):.2%}")
        if validation.get('warnings'):
            print(f"  Warnings: {', '.join(validation['warnings'])}")
    
    # Display enhanced citations
    if 'citations' in result2 and result2['citations']:
        print("\nCitations:")
        for citation in result2['citations']:
            print(f"  [{citation['index']}] {citation['citation_string']}")
            print(f"      File: {citation['file_name']}")
            if citation.get('page') and citation['page'] != 'N/A':
                print(f"      Page: {citation['page']}")
    print("-" * 50 + "\n")
    
    # Demonstrate calculator tool
    print("\n" + "="*50)
    print("Demonstrating Calculator Tool")
    print("="*50 + "\n")
    
    calc_question = "Calculate 25 * 4 + 10"
    print(f"Calculator Question: {calc_question}")
    calc_result = rag.query(calc_question)
    print(f"Answer: {calc_result['result']}")
    if 'tools_used' in calc_result:
        print(f"Tools used: {', '.join(calc_result['tools_used'])}")
    print("-" * 50 + "\n")
    
    # Note about web search
    print("Note: Web search requires tavily-python or duckduckgo-search")
    print("Install with: pip install tavily-python  # or pip install duckduckgo-search")
    print("-" * 50 + "\n")
    
    # Another question
    question3 = "What is USP callout?"
    print(f"Question 3: {question3}")
    result3 = rag.query(question3)
    
    # Show rewritten query if available
    if 'rewritten_query' in result3 and result3['rewritten_query'] != question3:
        print(f"Rewritten query: {result3['rewritten_query']}")
    
    print(f"Answer: {result3['result']}")
    print(f"Source documents: {len(result3.get('source_documents', []))} found")
    
    # Display validation results if available
    if 'validation' in result3:
        validation = result3['validation']
        print(f"\nAnswer Validation:")
        print(f"  Valid: {validation.get('is_valid', 'Unknown')}")
        print(f"  Confidence: {validation.get('confidence', 0):.2%}")
        if validation.get('warnings'):
            print(f"  Warnings: {', '.join(validation['warnings'])}")
    
    # Display enhanced citations
    if 'citations' in result3 and result3['citations']:
        print("\nCitations:")
        for citation in result3['citations']:
            print(f"  [{citation['index']}] {citation['citation_string']}")
            print(f"      File: {citation['file_name']}")
            if citation.get('page') and citation['page'] != 'N/A':
                print(f"      Page: {citation['page']}")
    print("-" * 50 + "\n")
    
    # Show conversation history
    print("\n" + "="*50)
    print("Conversation History")
    print("="*50)
    history = rag.get_memory_summary()
    if history:
        print(history)
    else:
        print("No conversation history available.")
    print("="*50 + "\n")
    
    # Clear memory example
    # rag.clear_memory()
    # print("Memory cleared!")

if __name__ == "__main__":
    # Make sure you have GEMINI_API_KEY (or GROQ_API_KEY / OPENAI_API_KEY) set in your .env file
    main()


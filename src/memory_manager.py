"""
Memory Manager for RAG System
Handles conversational memory management.
"""

from typing import Optional, List
from langchain_core.messages import HumanMessage, AIMessage

# Import memory - use ChatMessageHistory for LangChain 1.x
try:
    from langchain_community.chat_message_histories import ChatMessageHistory
except ImportError:
    # Fallback: create a simple message history
    class ChatMessageHistory:
        def __init__(self):
            self.messages = []
        
        def add_user_message(self, message: str):
            from langchain_core.messages import HumanMessage
            self.messages.append(HumanMessage(content=message))
        
        def add_ai_message(self, message: str):
            from langchain_core.messages import AIMessage
            self.messages.append(AIMessage(content=message))
        
        def clear(self):
            self.messages = []


class ConversationBufferMemory:
    """
    Simple memory wrapper for conversation history.
    """
    
    def __init__(self, return_messages=True, memory_key="chat_history", output_key="answer"):
        self.chat_memory = ChatMessageHistory()
        self.return_messages = return_messages
        self.memory_key = memory_key
        self.output_key = output_key
    
    def clear(self):
        self.chat_memory.clear()
    
    def get_messages(self) -> List:
        """Get all messages from memory."""
        return self.chat_memory.messages


class MemoryManager:
    """
    Manages conversational memory for the RAG system.
    """
    
    def __init__(self, use_memory: bool = True, memory_type: str = "buffer"):
        """
        Initialize the memory manager.
        
        Args:
            use_memory: Whether to use conversational memory
            memory_type: Type of memory to use ("buffer" for now)
        """
        self.use_memory = use_memory
        self.memory_type = memory_type
        
        if self.use_memory:
            if memory_type == "buffer":
                self.memory = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="chat_history",
                    output_key="answer"
                )
            else:
                raise ValueError(f"Unsupported memory type: {memory_type}")
        else:
            self.memory = None
    
    def get_chat_history(self) -> Optional[List]:
        """
        Get chat history from memory.
        
        Returns:
            List of messages or None if memory is disabled
        """
        if self.memory is not None:
            return self.memory.get_messages()
        return None
    
    def add_message(self, question: str, answer: str) -> None:
        """
        Add a question-answer pair to memory.
        
        Args:
            question: User question
            answer: System answer
        """
        if self.memory is not None:
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
    
    def clear(self) -> None:
        """Clear the conversation memory."""
        if self.memory is not None:
            self.memory.clear()
            print("Conversation memory cleared.")
    
    def get_memory_summary(self) -> Optional[str]:
        """
        Get a summary of the conversation history.
        
        Returns:
            Summary string if memory is enabled, None otherwise
        """
        if self.memory is not None and self.memory.chat_memory.messages:
            return "\n".join([
                f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
                for msg in self.memory.chat_memory.messages
            ])
        return None


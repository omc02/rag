"""
RAG chain implementations
Contains custom chain implementations for LangChain compatibility
"""

from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI

from ..config import RAGConfig


class StuffDocumentsChain:
    """Custom implementation of create_stuff_documents_chain with invoke method"""
    
    def __init__(self, llm: ChatOpenAI, prompt: ChatPromptTemplate, document_prompt: PromptTemplate = None):
        self.llm = llm
        self.prompt = prompt
        self.document_prompt = document_prompt
    
    def format_docs(self, docs: List[Any]) -> str:
        """Format documents using the document prompt template"""
        if self.document_prompt:
            formatted = []
            for doc in docs:
                formatted.append(
                    self.document_prompt.format(**doc.metadata, page_content=doc.page_content)
                )
            return "\n\n".join(formatted)
        else:
            return "\n\n".join(doc.page_content for doc in docs)
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the chain with input context and question"""
        docs = inputs["context"]
        context = self.format_docs(docs)
        
        # Create a new dict with the formatted context, avoiding duplicate keys
        prompt_inputs = {k: v for k, v in inputs.items() if k != "context"}
        prompt_inputs["context"] = context
        
        messages = self.prompt.format_messages(**prompt_inputs)
        response = self.llm.invoke(messages)
        
        return {"answer": response.content, "context": docs}


class RetrievalChain:
    """Custom implementation of create_retrieval_chain with invoke method"""
    
    def __init__(self, retriever: BaseRetriever, combine_docs_chain: StuffDocumentsChain):
        self.retriever = retriever
        self.combine_docs_chain = combine_docs_chain
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the retrieval chain"""
        # Get the query
        query = inputs["input"]
        
        # Retrieve relevant documents
        docs = self.retriever.invoke(query)
        
        # Pass to the combine chain
        chain_input = {"context": docs, "input": query}
        result = self.combine_docs_chain.invoke(chain_input)
        
        return result


class ConversationalRAG:
    """
    A conversational RAG system that maintains chat history and can handle follow-up questions.
    """
    
    def __init__(self, rag_chain: RetrievalChain, config: RAGConfig):
        self.rag_chain = rag_chain
        self.config = config
        self.max_history = config.conversation.max_history
    
    def _format_chat_history(self, chat_history: List[tuple]) -> str:
        """Format recent chat history for context"""
        if not chat_history:
            return ""
        
        history_text = "\n\nRecent conversation history:\n"
        for i, (q, a) in enumerate(chat_history[-self.max_history:], 1):
            history_text += f"Q{i}: {q}\n"
            history_text += f"A{i}: {a[:200]}{'...' if len(a) > 200 else ''}\n\n"
        
        return history_text
    
    def ask(self, question: str, chat_history: List[tuple]) -> Dict[str, Any]:
        """
        Ask a question with conversation history context
        
        Args:
            question: The current question
            chat_history: List of (question, answer) tuples
            
        Returns:
            Dictionary with answer and context
        """
        # Add conversation context to the question if there's history
        contextual_question = question
        if chat_history:
            contextual_question = f"""
Previous conversation context:
{self._format_chat_history(chat_history)}

Current question: {question}

Please answer the current question, taking into account the previous conversation context if relevant. If the question refers to something mentioned earlier (like "it", "that", "the company", etc.), use the conversation history to understand what the user is referring to.
"""
        
        # Get response from RAG chain
        response = self.rag_chain.invoke({"input": contextual_question})
        return response


def create_stuff_documents_chain(
    llm: ChatOpenAI, 
    prompt: ChatPromptTemplate, 
    document_prompt: PromptTemplate = None
) -> StuffDocumentsChain:
    """Custom implementation of create_stuff_documents_chain"""
    return StuffDocumentsChain(llm, prompt, document_prompt)


def create_retrieval_chain(
    retriever: BaseRetriever, 
    combine_docs_chain: StuffDocumentsChain
) -> RetrievalChain:
    """Custom implementation of create_retrieval_chain"""
    return RetrievalChain(retriever, combine_docs_chain)


def create_rag_chain(
    config: RAGConfig,
    retriever: BaseRetriever,
    llm: ChatOpenAI
) -> RetrievalChain:
    """
    Create a complete RAG chain from configuration
    
    Args:
        config: RAG configuration
        retriever: Document retriever
        llm: Language model
        
    Returns:
        Complete RAG chain
    """
    # Create prompts from configuration
    prompt = ChatPromptTemplate.from_messages([
        ("system", config.prompts.system_prompt),
        ("human", "{input}"),
    ])

    document_prompt = PromptTemplate.from_template(
        config.prompts.document_prompt_template
    )

    # Create chains
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_prompt=document_prompt,
    )

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )

    return rag_chain
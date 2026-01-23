"""
RAG (Retrieval-Augmented Generation) utilities for the F5 AI Technical Assistant.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import os


class RAGSystem:
    """Manages the RAG pipeline for F5 technical documentation."""

    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        persist_dir: Optional[Path] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        collection_name: str = "f5_docs"
    ):
        """
        Initialize the RAG system.

        Args:
            docs_dir: Directory containing F5 documentation files.
            persist_dir: Directory to persist the vector store.
            embedding_model: Name of the embedding model to use.
            chunk_size: Size of text chunks in characters.
            chunk_overlap: Overlap between chunks.
            collection_name: Name for the ChromaDB collection.
        """
        base_dir = Path(__file__).parent.parent

        self.docs_dir = docs_dir or base_dir / "data" / "f5_docs"
        self.persist_dir = persist_dir or base_dir / "chroma_db"
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name

        self.vectorstore = None
        self.retriever = None
        self.embeddings = None

    def initialize_embeddings(self):
        """Initialize the embedding model."""
        from langchain_huggingface import HuggingFaceEmbeddings

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )

        return self.embeddings

    def load_documents(self) -> List[Any]:
        """
        Load documents from the F5 docs directory.

        Returns:
            List of LangChain Document objects.
        """
        from langchain_community.document_loaders import DirectoryLoader, TextLoader

        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_dir}")

        loader = DirectoryLoader(
            str(self.docs_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )

        documents = loader.load()
        print(f"Loaded {len(documents)} documents from {self.docs_dir}")

        return documents

    def split_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into chunks.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List of chunked Document objects.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")

        return chunks

    def create_vectorstore(self, chunks: List[Any], persist: bool = True) -> Any:
        """
        Create a ChromaDB vector store from document chunks.

        Args:
            chunks: List of document chunks.
            persist: Whether to persist the vector store.

        Returns:
            ChromaDB vector store.
        """
        from langchain_community.vectorstores import Chroma

        if self.embeddings is None:
            self.initialize_embeddings()

        # Ensure persist directory exists
        if persist:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir) if persist else None
        )

        print(f"Created vector store with {len(chunks)} chunks")

        return self.vectorstore

    def load_vectorstore(self) -> Any:
        """
        Load an existing vector store from disk.

        Returns:
            ChromaDB vector store.
        """
        from langchain_community.vectorstores import Chroma

        if not self.persist_dir.exists():
            raise FileNotFoundError(f"Vector store not found at {self.persist_dir}")

        if self.embeddings is None:
            self.initialize_embeddings()

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir)
        )

        return self.vectorstore

    def get_retriever(self, k: int = 3) -> Any:
        """
        Get a retriever for the vector store.

        Args:
            k: Number of documents to retrieve.

        Returns:
            LangChain retriever object.
        """
        if self.vectorstore is None:
            try:
                self.load_vectorstore()
            except FileNotFoundError:
                raise RuntimeError("Vector store not initialized. Call create_vectorstore first.")

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        return self.retriever

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Any, float]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query.
            k: Number of documents to retrieve.

        Returns:
            List of (document, score) tuples.
        """
        if self.vectorstore is None:
            self.load_vectorstore()

        results = self.vectorstore.similarity_search_with_score(query, k=k)

        return results

    def format_context(self, documents: List[Any], include_source: bool = True) -> str:
        """
        Format retrieved documents into a context string.

        Args:
            documents: List of retrieved documents.
            include_source: Whether to include source file names.

        Returns:
            Formatted context string.
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            if include_source:
                source = doc.metadata.get("source", "unknown")
                context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
            else:
                context_parts.append(doc.page_content)

        return "\n\n".join(context_parts)

    def build_rag_chain(self, llm: Any, k: int = 3) -> Any:
        """
        Build a complete RAG chain with the given LLM.

        Args:
            llm: The language model to use.
            k: Number of documents to retrieve.

        Returns:
            LangChain RetrievalQA chain.
        """
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        retriever = self.get_retriever(k=k)

        template = """Use the following context to answer the question. If you cannot find the answer in the context, say so but still try to provide helpful information based on your knowledge.

Context:
{context}

Question: {question}

Answer: """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        return chain

    def query_with_sources(self, chain: Any, question: str) -> Dict[str, Any]:
        """
        Query the RAG chain and return results with sources.

        Args:
            chain: The RAG chain.
            question: The question to ask.

        Returns:
            Dictionary with 'answer' and 'sources' keys.
        """
        result = chain.invoke({"query": question})

        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "unknown")
            })

        return {
            "answer": result["result"],
            "sources": sources
        }

    def setup_complete_pipeline(self) -> Any:
        """
        Set up the complete RAG pipeline from scratch.

        Returns:
            The retriever for use with an LLM.
        """
        # Load and process documents
        documents = self.load_documents()
        chunks = self.split_documents(documents)

        # Create vector store
        self.create_vectorstore(chunks)

        # Return retriever
        return self.get_retriever()

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.

        Returns:
            Dictionary with collection statistics.
        """
        if self.vectorstore is None:
            try:
                self.load_vectorstore()
            except FileNotFoundError:
                return {"error": "Vector store not found"}

        # Get collection info from ChromaDB
        collection = self.vectorstore._collection

        return {
            "name": self.collection_name,
            "count": collection.count(),
            "persist_dir": str(self.persist_dir)
        }

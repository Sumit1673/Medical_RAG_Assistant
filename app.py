"""FastAPI application for Advanced RAG system."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup sys.path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_assistant.core.document_loader import DocumentLoader
from rag_assistant.core.embedding_generator import EmbeddingGenerator
from rag_assistant.core.llm_handler import LLMHandler
from rag_assistant.core.query_handler import QueryHandler
from rag_assistant.core.reranker import CrossEncoderReranker
from rag_assistant.core.retriever import HybridRetriever
from rag_assistant.core.vector_store_manager import VectorStoreManager
from rag_assistant.utils.config_loader import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced RAG System",
    description="Production-grade RAG with hybrid retrieval, reranking, and streaming",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    enable_rewriting: Optional[bool] = True
    enable_reranking: Optional[bool] = True


class SourceDocument(BaseModel):
    """Source document model."""
    content: str
    source: str
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Query response model."""
    request_id: str
    answer: str
    original_query: str
    rewritten_query: str
    latency_ms: float
    source_documents: list[SourceDocument]


class PipelineInfo(BaseModel):
    """Pipeline information model."""
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    retrieval_config: dict


# Global components
config: Optional[ConfigLoader] = None
query_handler: Optional[QueryHandler] = None
pipeline_info: Optional[PipelineInfo] = None


def initialize_pipeline() -> None:
    """Initialize RAG pipeline components."""
    global config, query_handler, pipeline_info

    try:
        # Load configuration
        config_path = Path(__file__).parent / "config" / "config.yaml"
        config = ConfigLoader(str(config_path))
        logger.info("Configuration loaded")

        # Initialize embeddings
        embedding_config = config.get_section("embedding")
        embeddings = EmbeddingGenerator(
            provider=embedding_config.get("provider", "huggingface"),
            model_name=embedding_config.get("model_name", "BAAI/bge-small-en-v1.5"),
            device=embedding_config.get("device", "cpu"),
            normalize_embeddings=embedding_config.get("normalize_embeddings", True),
        )
        logger.info(f"Embeddings initialized: {embedding_config.get('model_name')}")

        # Initialize vector store
        vs_config = config.get_section("vector_store")
        vector_store = VectorStoreManager(
            embeddings=embeddings.embeddings,
            collection_name=vs_config.get("collection_name", "medical_rag"),
            persist_directory=str(Path(__file__).parent / vs_config.get("vector_store_dir", "vector_store/")),
            chroma_server_url=vs_config.get("chroma_server_url"),
        )
        logger.info("Vector store initialized")

        # Load documents
        data_dir = Path(__file__).parent / "dataset" / "medical"
        documents = DocumentLoader.load_documents(str(data_dir))

        if documents:
            logger.info(f"Loaded {len(documents)} documents")
        else:
            logger.warning(f"No documents found in {data_dir}")

        # Initialize retriever
        retrieval_config = config.get_section("retrieval")
        retriever = HybridRetriever(
            documents=documents,
            embeddings=embeddings.embeddings,
            dense_top_k=retrieval_config.get("dense_top_k", 10),
            sparse_top_k=retrieval_config.get("sparse_top_k", 10),
            top_p=retrieval_config.get("top_p"),
        )
        logger.info("Retriever initialized")

        # Initialize LLM
        llm_config = config.get_section("llm")
        llm_handler = LLMHandler(
            provider=llm_config.get("provider", "ollama"),
            model_name=llm_config.get("model_name", "mistral"),
            temperature=llm_config.get("temperature", 0.2),
            max_tokens=llm_config.get("max_tokens", 1024),
            model_base_url=llm_config.get("model_base_url"),
        )
        logger.info(f"LLM initialized: {llm_config.get('model_name')}")

        # Initialize reranker
        reranker_config = config.get_section("reranker")
        reranker = CrossEncoderReranker(
            model_name=reranker_config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            top_n=reranker_config.get("top_n", 5),
            top_p=reranker_config.get("top_p"),
        )
        logger.info("Reranker initialized")

        # Initialize query handler
        query_handler = QueryHandler(
            retriever=retriever,
            llm_handler=llm_handler,
            reranker=reranker,
            enable_query_rewriting=retrieval_config.get("enable_query_rewriting", True),
            enable_reranking=retrieval_config.get("enable_reranking", True),
            final_top_k=retrieval_config.get("final_top_k", 5),
        )
        logger.info("Query handler initialized")

        # Store pipeline info
        pipeline_info = PipelineInfo(
            llm_provider=llm_config.get("provider", "ollama"),
            llm_model=llm_config.get("model_name", "mistral"),
            embedding_provider=embedding_config.get("provider", "huggingface"),
            embedding_model=embedding_config.get("model_name", "BAAI/bge-small-en-v1.5"),
            retrieval_config=retrieval_config,
        )

        logger.info("Pipeline initialization complete")

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        raise


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize on startup."""
    initialize_pipeline()


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Advanced RAG System",
    }


@app.get("/pipeline/info")
async def get_pipeline_info() -> dict:
    """Get pipeline information."""
    if not pipeline_info:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return pipeline_info.model_dump()


@app.post("/query")
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Query endpoint with full RAG pipeline."""
    if not query_handler:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Process query
        result = query_handler.answer_query(request.query)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Convert source documents
        source_docs = [
            SourceDocument(
                content=doc["content"],
                source=doc["source"],
                rerank_score=doc.get("rerank_score"),
            )
            for doc in result["source_documents"]
        ]

        return QueryResponse(
            request_id=request_id,
            answer=result["answer"],
            original_query=result["original_query"],
            rewritten_query=result["rewritten_query"],
            latency_ms=latency_ms,
            source_documents=source_docs,
        )

    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream", response_model=None)
async def query_stream_endpoint(request: QueryRequest):
    """Streaming query endpoint."""
    if not query_handler:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Stream response
        async for token in query_handler.answer_query_stream(request.query):
            yield f"data: {token}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming query failed: {e}", exc_info=True)
        yield f"data: ERROR: {str(e)}\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )

import pytest
from unittest.mock import Mock, patch, AsyncMock
import sys
import os
from fastapi.testclient import TestClient

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from rag_system import RAGSystem
from ai_generator import AIGenerator
from search_tools import ToolManager
from session_manager import SessionManager
from vector_store import VectorStore
from document_processor import DocumentProcessor


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Mock(spec=Config)
    config.OLLAMA_BASE_URL = "http://localhost:11434"
    config.OLLAMA_MODEL = "mistral:7b"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_DB_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    vector_store = Mock(spec=VectorStore)
    vector_store.search.return_value = [
        {
            "content": "Test content about machine learning",
            "course": "Test Course",
            "lesson": "Lesson 1",
            "distance": 0.2
        }
    ]
    vector_store.get_course_count.return_value = 2
    vector_store.get_existing_course_titles.return_value = ["Test Course", "Another Course"]
    return vector_store


@pytest.fixture
def mock_ai_generator():
    """Mock AI generator for testing"""
    ai_generator = Mock(spec=AIGenerator)
    ai_generator.generate_response.return_value = "This is a test AI response"
    return ai_generator


@pytest.fixture
def mock_tool_manager():
    """Mock tool manager for testing"""
    tool_manager = Mock(spec=ToolManager)
    tool_manager.get_tool_definitions.return_value = [
        {
            "type": "function",
            "function": {
                "name": "search_course_content",
                "description": "Search course materials",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    tool_manager.get_last_sources.return_value = [
        {
            "content": "Test source content",
            "course": "Test Course", 
            "lesson": "Lesson 1",
            "distance": 0.2
        }
    ]
    tool_manager.execute_tool.return_value = "Tool execution result"
    return tool_manager


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    session_manager = Mock(spec=SessionManager)
    session_manager.create_session.return_value = "test-session-123"
    session_manager.get_conversation_history.return_value = "Previous conversation"
    return session_manager


@pytest.fixture
def mock_document_processor():
    """Mock document processor for testing"""
    processor = Mock(spec=DocumentProcessor)
    processor.process_course_document.return_value = [
        {
            "content": "Test chunk content",
            "course": "Test Course",
            "lesson": "Lesson 1"
        }
    ]
    return processor


@pytest.fixture
def mock_rag_system(mock_config, mock_vector_store, mock_ai_generator, 
                   mock_tool_manager, mock_session_manager, mock_document_processor):
    """Mock RAG system with all dependencies mocked"""
    rag_system = Mock(spec=RAGSystem)
    rag_system.vector_store = mock_vector_store
    rag_system.ai_generator = mock_ai_generator
    rag_system.tool_manager = mock_tool_manager
    rag_system.session_manager = mock_session_manager
    rag_system.document_processor = mock_document_processor
    
    # Mock methods
    rag_system.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course", "Another Course"]
    }
    rag_system.add_course_folder.return_value = (2, 10)
    
    return rag_system


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app with mocked dependencies"""
    # Create a minimal test app without static file mounting
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any
    
    app = FastAPI(title="Test RAG System")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Dict[str, Any]]
        session_id: str
        tools_used: List[str]

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # Define endpoints inline to avoid import issues
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag_system.session_manager.create_session()
            
            history = mock_rag_system.session_manager.get_conversation_history(session_id)
            tools = mock_rag_system.tool_manager.get_tool_definitions()
            
            answer = mock_rag_system.ai_generator.generate_response(
                query=request.query,
                conversation_history=history,
                tools=tools,
                tool_manager=mock_rag_system.tool_manager
            )
            
            sources = mock_rag_system.tool_manager.get_last_sources()
            tools_used = ["ai_directed_search"] if sources else ["ai_general_knowledge"]
            
            mock_rag_system.session_manager.add_exchange(session_id, request.query, answer)
            mock_rag_system.tool_manager.reset_sources()
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id,
                tools_used=tools_used
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "RAG System API"}
    
    return app


@pytest.fixture
def client(test_app):
    """FastAPI test client"""
    return TestClient(test_app)


@pytest.fixture
def sample_query_request():
    """Sample query request for testing"""
    return {
        "query": "What is machine learning?",
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_query_response():
    """Sample query response for testing"""
    return {
        "answer": "Machine learning is a subset of artificial intelligence...",
        "sources": [
            {
                "content": "Machine learning content",
                "course": "AI Course",
                "lesson": "Lesson 1",
                "distance": 0.1
            }
        ],
        "session_id": "test-session-123",
        "tools_used": ["ai_directed_search"]
    }


# Cleanup fixture for any test databases or temporary files
@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    yield
    # Clean up any test files or databases if needed
    test_db_path = "./test_chroma_db"
    if os.path.exists(test_db_path):
        import shutil
        try:
            shutil.rmtree(test_db_path)
        except Exception:
            pass
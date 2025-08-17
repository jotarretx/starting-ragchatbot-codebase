import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json


@pytest.mark.api
class TestAPIEndpoints:
    """Test suite for FastAPI endpoints"""

    def test_root_endpoint(self, client):
        """Test the root endpoint returns correct response"""
        response = client.get("/")
        
        assert response.status_code == 200
        assert response.json() == {"message": "RAG System API"}

    def test_query_endpoint_success(self, client, sample_query_request, mock_rag_system):
        """Test successful query processing"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert "tools_used" in data
        
        # Verify content
        assert data["answer"] == "This is a test AI response"
        assert data["session_id"] == "test-session-123"
        assert isinstance(data["sources"], list)
        assert isinstance(data["tools_used"], list)

    def test_query_endpoint_without_session_id(self, client, mock_rag_system):
        """Test query without session_id creates new session"""
        request_data = {"query": "What is machine learning?"}
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have created a new session
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called()

    def test_query_endpoint_with_session_id(self, client, sample_query_request, mock_rag_system):
        """Test query with existing session_id"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should use provided session ID
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.get_conversation_history.assert_called_with("test-session-123")

    def test_query_endpoint_ai_generator_called(self, client, sample_query_request, mock_rag_system):
        """Test that AI generator is called with correct parameters"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        
        # Verify AI generator was called with expected parameters
        mock_rag_system.ai_generator.generate_response.assert_called_once()
        call_args = mock_rag_system.ai_generator.generate_response.call_args
        
        assert call_args[1]["query"] == "What is machine learning?"
        assert call_args[1]["conversation_history"] == "Previous conversation"
        assert "tools" in call_args[1]
        assert "tool_manager" in call_args[1]

    def test_query_endpoint_tools_flow(self, client, sample_query_request, mock_rag_system):
        """Test the tools flow in query processing"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify tool-related calls
        mock_rag_system.tool_manager.get_tool_definitions.assert_called_once()
        mock_rag_system.tool_manager.get_last_sources.assert_called_once()
        mock_rag_system.tool_manager.reset_sources.assert_called_once()
        
        # Check tools_used classification
        assert data["tools_used"] == ["ai_directed_search"]  # Since sources are returned by mock

    def test_query_endpoint_no_sources(self, client, sample_query_request, mock_rag_system):
        """Test query when no sources are found"""
        # Mock no sources returned
        mock_rag_system.tool_manager.get_last_sources.return_value = []
        
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["tools_used"] == ["ai_general_knowledge"]
        assert data["sources"] == []

    def test_query_endpoint_session_management(self, client, sample_query_request, mock_rag_system):
        """Test session management in query processing"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        
        # Verify session management calls
        mock_rag_system.session_manager.add_exchange.assert_called_once_with(
            "test-session-123",
            "What is machine learning?",
            "This is a test AI response"
        )

    def test_query_endpoint_invalid_request(self, client):
        """Test query endpoint with invalid request data"""
        # Missing required 'query' field
        invalid_request = {"session_id": "test-123"}
        
        response = client.post("/api/query", json=invalid_request)
        
        assert response.status_code == 422  # Validation error

    def test_query_endpoint_empty_query(self, client):
        """Test query endpoint with empty query string"""
        request_data = {"query": ""}
        
        response = client.post("/api/query", json=request_data)
        
        # Should still process but with empty query
        assert response.status_code == 200

    def test_query_endpoint_exception_handling(self, client, sample_query_request, mock_rag_system):
        """Test query endpoint exception handling"""
        # Mock an exception in AI generator
        mock_rag_system.ai_generator.generate_response.side_effect = Exception("AI service error")
        
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 500
        assert "AI service error" in response.json()["detail"]

    def test_courses_endpoint_success(self, client, mock_rag_system):
        """Test successful course statistics retrieval"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Verify content matches mock
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Test Course", "Another Course"]

    def test_courses_endpoint_analytics_called(self, client, mock_rag_system):
        """Test that course analytics method is called"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_courses_endpoint_exception_handling(self, client, mock_rag_system):
        """Test courses endpoint exception handling"""
        # Mock an exception in get_course_analytics
        mock_rag_system.get_course_analytics.side_effect = Exception("Database error")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]

    def test_courses_endpoint_empty_response(self, client, mock_rag_system):
        """Test courses endpoint with no courses"""
        # Mock empty analytics
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []


@pytest.mark.api
class TestAPIValidation:
    """Test API request/response validation"""

    def test_query_request_validation(self, client):
        """Test query request model validation"""
        # Test various invalid payloads
        invalid_payloads = [
            {},  # Missing query
            {"query": None},  # Null query
            {"query": 123},  # Wrong type
            {"query": "valid", "session_id": 123},  # Wrong session_id type
        ]
        
        for payload in invalid_payloads:
            response = client.post("/api/query", json=payload)
            assert response.status_code == 422

    def test_query_response_structure(self, client, sample_query_request):
        """Test query response model structure"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Required fields
        required_fields = ["answer", "sources", "session_id", "tools_used"]
        for field in required_fields:
            assert field in data
        
        # Type validation
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        assert isinstance(data["tools_used"], list)

    def test_courses_response_structure(self, client):
        """Test courses response model structure"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Required fields
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data
        
        # Type validation
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Ensure all course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)


@pytest.mark.api
class TestAPIIntegration:
    """Integration tests for API endpoints"""

    def test_query_sources_structure(self, client, sample_query_request, mock_rag_system):
        """Test that sources have correct structure"""
        # Mock sources with proper structure
        mock_sources = [
            {
                "content": "Test content 1",
                "course": "Course A",
                "lesson": "Lesson 1",
                "distance": 0.1
            },
            {
                "content": "Test content 2", 
                "course": "Course B",
                "lesson": "Lesson 2",
                "distance": 0.2
            }
        ]
        mock_rag_system.tool_manager.get_last_sources.return_value = mock_sources
        
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["sources"]) == 2
        for source in data["sources"]:
            assert "content" in source
            assert "course" in source
            assert "lesson" in source
            assert "distance" in source

    def test_multiple_queries_same_session(self, client, mock_rag_system):
        """Test multiple queries with the same session ID"""
        session_id = "persistent-session-123"
        
        # First query
        query1 = {"query": "First question", "session_id": session_id}
        response1 = client.post("/api/query", json=query1)
        assert response1.status_code == 200
        assert response1.json()["session_id"] == session_id
        
        # Second query with same session
        query2 = {"query": "Follow-up question", "session_id": session_id}
        response2 = client.post("/api/query", json=query2)
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id
        
        # Verify session manager was used correctly
        assert mock_rag_system.session_manager.get_conversation_history.call_count == 2
        assert mock_rag_system.session_manager.add_exchange.call_count == 2

    def test_courses_and_query_consistency(self, client, mock_rag_system):
        """Test consistency between courses endpoint and query sources"""
        # Get courses
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == 200
        courses_data = courses_response.json()
        
        # Perform query
        query_response = client.post("/api/query", json={"query": "test"})
        assert query_response.status_code == 200
        query_data = query_response.json()
        
        # Sources should reference courses that exist in course list
        for source in query_data["sources"]:
            course_name = source.get("course")
            if course_name:
                assert course_name in courses_data["course_titles"]

    def test_api_cors_headers(self, client):
        """Test that CORS headers are properly set"""
        response = client.get("/api/courses")
        assert response.status_code == 200
        
        # Note: TestClient doesn't automatically include CORS headers,
        # but we can verify the middleware is configured in the app
        # This is more of a smoke test to ensure endpoints work with CORS middleware


# Performance and load testing markers for future expansion
@pytest.mark.performance
class TestAPIPerformance:
    """Performance tests for API endpoints (placeholder for future)"""
    
    def test_query_endpoint_response_time(self, client, sample_query_request):
        """Test query endpoint response time"""
        import time
        
        start_time = time.time()
        response = client.post("/api/query", json=sample_query_request)
        end_time = time.time()
        
        assert response.status_code == 200
        # Basic performance check - should respond within reasonable time
        response_time = end_time - start_time
        assert response_time < 5.0  # 5 seconds max for mocked response
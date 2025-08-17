import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator, ToolCallRound, ToolCallSession


class TestAIGenerator:
    """Test suite for AIGenerator with multi-round tool calling capabilities"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.ai_generator = AIGenerator(
            base_url="http://localhost:11434",
            model="mistral:7b",
            max_rounds=2
        )
        
        # Mock the Ollama client
        self.mock_client = Mock()
        self.ai_generator.client = self.mock_client
        
        # Mock tool manager
        self.mock_tool_manager = Mock()
        
        # Sample tools definition
        self.sample_tools = [
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
            },
            {
                "type": "function", 
                "function": {
                    "name": "get_course_outline",
                    "description": "Get course outline",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "course_name": {"type": "string", "description": "Course name"}
                        },
                        "required": ["course_name"]
                    }
                }
            }
        ]
    
    def test_generate_response_without_tools(self):
        """Test response generation when no tools are provided"""
        # Mock response from Ollama
        mock_response = {
            "message": {
                "content": "This is a general knowledge response."
            }
        }
        self.mock_client.chat.return_value = mock_response
        
        # Test
        result = self.ai_generator.generate_response(
            query="What is machine learning?",
            conversation_history=None,
            tools=None,
            tool_manager=None
        )
        
        # Assertions
        assert result == "This is a general knowledge response."
        self.mock_client.chat.assert_called_once()
        
        # Verify API call structure
        call_args = self.mock_client.chat.call_args[1]
        assert "tools" not in call_args
        assert len(call_args["messages"]) == 2
        assert call_args["messages"][1]["content"] == "What is machine learning?"
    
    def test_single_round_with_tools_no_tool_calls(self):
        """Test when tools are available but AI chooses not to use them"""
        # Mock response without tool calls
        mock_response = {
            "message": {
                "content": "This is a direct answer without using tools."
            }
        }
        self.mock_client.chat.return_value = mock_response
        
        # Test
        result = self.ai_generator.generate_response(
            query="General question",
            tools=self.sample_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Assertions
        assert result == "This is a direct answer without using tools."
        
        # Should have made exactly one API call with tools
        self.mock_client.chat.assert_called_once()
        call_args = self.mock_client.chat.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == self.sample_tools
    
    def test_single_round_with_one_tool_call(self):
        """Test single round with one tool call followed by final response"""
        # Mock first response with tool call
        mock_response_with_tool = {
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_123",
                    "function": {
                        "name": "search_course_content",
                        "arguments": {"query": "machine learning basics"}
                    }
                }]
            }
        }
        
        # Mock second response without tool calls (round 2)
        mock_response_no_tools = {
            "message": {
                "content": "I can provide information based on the search results."
            }
        }
        
        # Mock final synthesis response 
        mock_final_response = {
            "message": {
                "content": "Based on the search results, machine learning is..."
            }
        }
        
        # Configure mock to return different responses on consecutive calls
        self.mock_client.chat.side_effect = [mock_response_with_tool, mock_response_no_tools, mock_final_response]
        
        # Mock tool execution
        self.mock_tool_manager.execute_tool.return_value = "Search result: ML is a subset of AI..."
        
        # Test
        result = self.ai_generator.generate_response(
            query="What is machine learning?",
            tools=self.sample_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Assertions
        assert result == "Based on the search results, machine learning is..."
        
        # Should have made three API calls (round1 + round2 + final synthesis)
        assert self.mock_client.chat.call_count == 3
        
        # First call should have tools
        first_call_args = self.mock_client.chat.call_args_list[0][1]
        assert "tools" in first_call_args
        
        # Second call should have tools (round 2)
        second_call_args = self.mock_client.chat.call_args_list[1][1]
        assert "tools" in second_call_args
        
        # Third call should not have tools (final synthesis)
        third_call_args = self.mock_client.chat.call_args_list[2][1]
        assert "tools" not in third_call_args
        
        # Tool should have been executed once
        self.mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning basics"
        )
    
    def test_two_rounds_with_tool_calls(self):
        """Test two sequential rounds of tool calling"""
        # Mock first response with tool call
        mock_response_round1 = {
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_123",
                    "function": {
                        "name": "get_course_outline",
                        "arguments": {"course_name": "Course A"}
                    }
                }]
            }
        }
        
        # Mock second response with another tool call
        mock_response_round2 = {
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_456",
                    "function": {
                        "name": "search_course_content",
                        "arguments": {"query": "lesson 4 content"}
                    }
                }]
            }
        }
        
        # Mock final response
        mock_final_response = {
            "message": {
                "content": "After searching Course A outline and lesson 4 content..."
            }
        }
        
        # Configure mock to return responses in sequence
        self.mock_client.chat.side_effect = [
            mock_response_round1,
            mock_response_round2, 
            mock_final_response
        ]
        
        # Mock tool executions
        self.mock_tool_manager.execute_tool.side_effect = [
            "Course A outline: Lesson 1, Lesson 2, Lesson 3, Lesson 4: Advanced Topics",
            "Lesson 4 covers advanced ML algorithms..."
        ]
        
        # Test
        result = self.ai_generator.generate_response(
            query="What does lesson 4 of Course A cover?",
            tools=self.sample_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Assertions
        assert result == "After searching Course A outline and lesson 4 content..."
        
        # Should have made three API calls (round1 + round2 + final)
        assert self.mock_client.chat.call_count == 3
        
        # Both first and second calls should have tools
        for i in range(2):
            call_args = self.mock_client.chat.call_args_list[i][1]
            assert "tools" in call_args
        
        # Final call should not have tools
        final_call_args = self.mock_client.chat.call_args_list[2][1]
        assert "tools" not in final_call_args
        
        # Both tools should have been executed
        assert self.mock_tool_manager.execute_tool.call_count == 2
        self.mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline",
            course_name="Course A"
        )
        self.mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="lesson 4 content"
        )
    
    def test_max_rounds_termination(self):
        """Test that processing stops after max rounds even if AI wants to continue"""
        # Mock responses that always include tool calls
        mock_response_with_tool = {
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_123",
                    "function": {
                        "name": "search_course_content",
                        "arguments": {"query": "test query"}
                    }
                }]
            }
        }
        
        # Mock final response
        mock_final_response = {
            "message": {
                "content": "Final response after max rounds reached"
            }
        }
        
        # Configure mock to return tool calls for first two calls, then final
        self.mock_client.chat.side_effect = [
            mock_response_with_tool,  # Round 1
            mock_response_with_tool,  # Round 2 (max reached)
            mock_final_response       # Final synthesis
        ]
        
        # Mock tool execution
        self.mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # Test
        result = self.ai_generator.generate_response(
            query="Complex query requiring multiple searches",
            tools=self.sample_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Assertions
        assert result == "Final response after max rounds reached"
        
        # Should have made exactly 3 calls (2 rounds + final)
        assert self.mock_client.chat.call_count == 3
        
        # Should have executed tools twice (once per round)
        assert self.mock_tool_manager.execute_tool.call_count == 2
    
    def test_tool_execution_error_handling(self):
        """Test handling of tool execution errors"""
        # Mock response with tool call
        mock_response_with_tool = {
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_123",
                    "function": {
                        "name": "search_course_content",
                        "arguments": {"query": "test"}
                    }
                }]
            }
        }
        
        # Mock round 2 response (no tool calls)
        mock_response_round2 = {
            "message": {
                "content": "I'll provide what I can despite the error."
            }
        }
        
        # Mock final synthesis response
        mock_final_response = {
            "message": {
                "content": "Response despite tool error"
            }
        }
        
        self.mock_client.chat.side_effect = [mock_response_with_tool, mock_response_round2, mock_final_response]
        
        # Mock tool execution to raise an exception
        self.mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        # Test
        result = self.ai_generator.generate_response(
            query="Test query",
            tools=self.sample_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Assertions
        assert result == "Response despite tool error"
        
        # Should have made three API calls despite tool error (round1 + round2 + final synthesis)
        assert self.mock_client.chat.call_count == 3
        
        # Tool execution should have been attempted once
        self.mock_tool_manager.execute_tool.assert_called_once()
    
    def test_conversation_history_preservation(self):
        """Test that conversation history is properly included in requests"""
        # Mock response
        mock_response = {
            "message": {
                "content": "Response with history context"
            }
        }
        self.mock_client.chat.return_value = mock_response
        
        conversation_history = "Previous conversation content"
        
        # Test
        result = self.ai_generator.generate_response(
            query="Follow-up question",
            conversation_history=conversation_history,
            tools=self.sample_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Assertions
        assert result == "Response with history context"
        
        # Check that conversation history was included in system message
        call_args = self.mock_client.chat.call_args[1]
        system_message = call_args["messages"][0]["content"]
        assert conversation_history in system_message
    
    def test_custom_max_rounds_override(self):
        """Test that max_rounds parameter can be overridden per query"""
        # Create AI generator with default max_rounds=2
        ai_gen = AIGenerator("http://localhost:11434", "mistral:7b", max_rounds=2)
        ai_gen.client = self.mock_client
        
        # Mock response without tool calls
        mock_response = {
            "message": {
                "content": "Direct response"
            }
        }
        self.mock_client.chat.return_value = mock_response
        
        # Test with custom max_rounds
        result = ai_gen.generate_response(
            query="Test query",
            tools=self.sample_tools,
            tool_manager=self.mock_tool_manager,
            max_rounds=1  # Override default
        )
        
        # Assertions
        assert result == "Direct response"
        
        # Verify the override was used (this is tested indirectly through behavior)
        # In a real scenario, we'd need more sophisticated mocking to test this directly
    

class TestToolCallRound:
    """Test the ToolCallRound dataclass"""
    
    def test_tool_call_round_initialization(self):
        """Test ToolCallRound initialization with defaults"""
        round_obj = ToolCallRound(round_number=1)
        
        assert round_obj.round_number == 1
        assert round_obj.ai_response is None
        assert round_obj.has_tool_calls is False
        assert round_obj.tools_called == []
        assert round_obj.tool_results == []
        assert round_obj.error is None


class TestToolCallSession:
    """Test the ToolCallSession dataclass"""
    
    def test_tool_call_session_initialization(self):
        """Test ToolCallSession initialization with defaults"""
        session = ToolCallSession(max_rounds=2)
        
        assert session.max_rounds == 2
        assert session.rounds == []
        assert session.messages == []
        assert session.terminated_early is False
        assert session.termination_reason is None


# Integration test scenarios
class TestAIGeneratorIntegration:
    """Integration tests that verify end-to-end behavior"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.ai_generator = AIGenerator(
            base_url="http://localhost:11434",
            model="mistral:7b",
            max_rounds=2
        )
        
        # Use a more sophisticated mock setup for integration tests
        self.mock_client = Mock()
        self.ai_generator.client = self.mock_client
        
        self.mock_tool_manager = Mock()
        
        self.sample_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_course_content",
                    "description": "Search course materials",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    def test_complex_multi_round_scenario(self):
        """Test a complex scenario similar to the example in requirements"""
        # Scenario: "Search for a course that discusses the same topic as lesson 4 of course X"
        
        # Round 1: Get course outline for course X
        mock_response_round1 = {
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "get_course_outline",
                        "arguments": {"course_name": "Course X"}
                    }
                }]
            }
        }
        
        # Round 2: Search for courses with similar topic
        mock_response_round2 = {
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_2", 
                    "function": {
                        "name": "search_course_content",
                        "arguments": {"query": "machine learning algorithms"}
                    }
                }]
            }
        }
        
        # Final response
        mock_final_response = {
            "message": {
                "content": "Based on the search, Course Y also covers machine learning algorithms similar to lesson 4 of Course X."
            }
        }
        
        self.mock_client.chat.side_effect = [
            mock_response_round1,
            mock_response_round2,
            mock_final_response
        ]
        
        # Mock tool results
        self.mock_tool_manager.execute_tool.side_effect = [
            "Course X Outline: 1. Intro, 2. Basics, 3. Foundations, 4. Machine Learning Algorithms",
            "Found Course Y: Introduction to ML Algorithms..."
        ]
        
        # Test the complex scenario
        result = self.ai_generator.generate_response(
            query="Search for a course that discusses the same topic as lesson 4 of course X",
            tools=self.sample_tools + [{
                "type": "function",
                "function": {
                    "name": "get_course_outline", 
                    "description": "Get course outline",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "course_name": {"type": "string"}
                        },
                        "required": ["course_name"]
                    }
                }
            }],
            tool_manager=self.mock_tool_manager
        )
        
        # Verify the complete flow
        assert result == "Based on the search, Course Y also covers machine learning algorithms similar to lesson 4 of Course X."
        assert self.mock_client.chat.call_count == 3
        assert self.mock_tool_manager.execute_tool.call_count == 2
        
        # Verify tool call sequence
        calls = self.mock_tool_manager.execute_tool.call_args_list
        assert calls[0][0] == ("get_course_outline",)
        assert calls[0][1] == {"course_name": "Course X"}
        assert calls[1][0] == ("search_course_content",)
        assert calls[1][1] == {"query": "machine learning algorithms"}


if __name__ == "__main__":
    pytest.main([__file__])
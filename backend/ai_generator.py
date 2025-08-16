import ollama
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Ollama for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage:
- Use the **search_course_content** tool for questions about specific course content or detailed educational materials
- Use the **get_course_outline** tool for questions about course structure, outlines, lesson lists, or course overview
- **One tool call per query maximum**
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use search_course_content tool first, then answer
- **Course outline questions**: Use get_course_outline tool first, then answer with course title, course link, and complete lesson list (lesson numbers and titles)
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the tool results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, base_url: str, model: str):
        self.client = ollama.Client(host=base_url)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 800
            }
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ]
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
        
        # Get response from Ollama
        response = self.client.chat(**api_params)
        
        # Handle tool execution if needed
        if tools and "tool_calls" in response["message"] and response["message"]["tool_calls"]:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response["message"]["content"]
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append(initial_response["message"])
        
        # Execute all tool calls and collect results
        tool_calls = initial_response["message"]["tool_calls"]
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = tool_call["function"]["arguments"]
            
            # Execute the tool
            tool_result = tool_manager.execute_tool(function_name, **function_args)
            
            # Add tool result message
            messages.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_call.get("id", "")
            })
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Get final response
        final_response = self.client.chat(**final_params)
        return final_response["message"]["content"]
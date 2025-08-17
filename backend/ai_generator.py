import ollama
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class ToolCallRound:
    """Represents a single round of tool calling"""
    round_number: int
    ai_response: Optional[Dict] = None
    has_tool_calls: bool = False
    tools_called: List[Dict] = None
    tool_results: List[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.tools_called is None:
            self.tools_called = []
        if self.tool_results is None:
            self.tool_results = []

@dataclass
class ToolCallSession:
    """Manages multi-round tool calling session"""
    max_rounds: int
    rounds: List[ToolCallRound] = None
    messages: List[Dict] = None
    terminated_early: bool = False
    termination_reason: Optional[str] = None
    
    def __post_init__(self):
        if self.rounds is None:
            self.rounds = []
        if self.messages is None:
            self.messages = []

class AIGenerator:
    """Handles interactions with Ollama for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage Protocol:
- You can make **up to 2 rounds** of tool calls to gather comprehensive information
- **Round 1**: Use tools to search for initial information related to the query
- **Round 2** (if needed): Use tools to gather additional or more specific information that would significantly improve your answer
- **Available tools**:
  - **search_course_content**: For specific course content and detailed educational materials
  - **get_course_outline**: For course structure, outlines, lesson lists, or course overview
- **Multi-round strategy**: Use different tools or different parameters to build comprehensive understanding
- **Synthesis**: After tool rounds, provide a complete answer based on all gathered information

Multi-Round Guidelines:
- **First round**: Gather primary information directly related to the user's question
- **Second round**: Only if additional information would significantly enhance the answer (e.g., for comparisons, follow-up details, or course structure after content search)
- **Final response**: Synthesize all tool results into a coherent, complete answer
- **No meta-commentary**: Don't explain your tool usage process or reasoning steps

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use search_course_content tool, then answer based on results
- **Course outline questions**: Use get_course_outline tool, include course title, link, and complete lesson list
- **Complex queries**: Use multiple rounds to gather complete information (e.g., "compare course A with course B" might need 2 searches)
- **No meta-commentary**: Provide direct answers only — no reasoning process, tool explanations, or question-type analysis

All responses must be:
1. **Comprehensive** - Use multiple rounds when needed to provide complete information
2. **Focused** - Stay relevant to the original query across all rounds
3. **Educational** - Maintain instructional value
4. **Clear** - Use accessible language with examples when helpful
Provide only the direct answer to what was asked, leveraging all information gathered across rounds.
"""
    
    def __init__(self, base_url: str, model: str, max_rounds: int = 2):
        self.client = ollama.Client(host=base_url)
        self.model = model
        self.max_rounds = max_rounds
        
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
                         tool_manager=None,
                         max_rounds: Optional[int] = None) -> str:
        """
        Generate AI response with optional multi-round tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Override default max rounds for this query
            
        Returns:
            Generated response as string
        """
        
        # Use instance default or override
        rounds_limit = max_rounds if max_rounds is not None else self.max_rounds
        
        # If no tools available, use simple response path
        if not tools:
            return self._generate_simple_response(query, conversation_history)
        
        # Build initial system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize base messages
        base_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]
        
        # Process multi-round tool calling
        session = self._process_multi_round_tool_calling(base_messages, tools, tool_manager, rounds_limit)
        
        # Generate final response
        final_response = self._generate_final_response(session)
        
        # Log session summary
        self._log_session_summary(session)
        
        return final_response
    
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
            
            print(f"   🔧 Executing {function_name} with args: {function_args}")
            
            # Execute the tool
            tool_result = tool_manager.execute_tool(function_name, **function_args)
            
            print(f"   ✅ Tool result length: {len(str(tool_result))} chars")
            
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
    
    def _generate_simple_response(self, query: str, conversation_history: Optional[str] = None) -> str:
        """
        Generate a simple response without tools.
        
        Args:
            query: The user's question
            conversation_history: Previous conversation context
            
        Returns:
            Generated response as string
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        api_params = {
            **self.base_params,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ]
        }
        
        response = self.client.chat(**api_params)
        print(f"📝 AI responded without tools (no tools available)")
        return response["message"]["content"]
    
    def _process_multi_round_tool_calling(self, base_messages: List[Dict], 
                                        tools: List, tool_manager, max_rounds: int) -> ToolCallSession:
        """
        Handle multi-round tool calling with proper state management.
        
        Args:
            base_messages: Initial conversation messages
            tools: Available tools for the AI
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of rounds allowed
            
        Returns:
            Complete session with all rounds and final response
        """
        session = ToolCallSession(max_rounds=max_rounds)
        session.messages = base_messages.copy()
        
        for round_num in range(1, max_rounds + 1):
            round_obj = ToolCallRound(round_number=round_num)
            
            # Make API call with current messages and tools
            response = self._make_api_call_with_tools(session.messages, tools)
            round_obj.ai_response = response
            
            # Check for tool calls
            if self._has_tool_calls(response):
                round_obj.has_tool_calls = True
                round_obj = self._execute_round_tools(round_obj, response, tool_manager)
                
                # Add AI message and tool results to session
                session.messages.append(response["message"])
                session.messages.extend(self._build_tool_result_messages(round_obj))
                
                session.rounds.append(round_obj)
                
                # Check if this was the last allowed round
                if round_num == max_rounds:
                    session.termination_reason = "max_rounds_reached"
                    break
                    
            else:
                # No tool calls - this is the final response
                round_obj.has_tool_calls = False
                session.rounds.append(round_obj)
                session.termination_reason = "no_tool_calls"
                break
        
        return session
    
    def _make_api_call_with_tools(self, messages: List[Dict], tools: List) -> Dict:
        """
        Make API call to Ollama with tools available.
        
        Args:
            messages: Current conversation messages
            tools: Available tools
            
        Returns:
            API response from Ollama
        """
        api_params = {
            **self.base_params,
            "messages": messages,
            "tools": tools
        }
        
        return self.client.chat(**api_params)
    
    def _has_tool_calls(self, response: Dict) -> bool:
        """
        Check if the response contains tool calls.
        
        Args:
            response: API response from Ollama
            
        Returns:
            True if response has tool calls, False otherwise
        """
        return ("tool_calls" in response["message"] and 
                response["message"]["tool_calls"] and 
                len(response["message"]["tool_calls"]) > 0)
    
    def _execute_round_tools(self, round_obj: ToolCallRound, 
                           response: Dict, tool_manager) -> ToolCallRound:
        """
        Execute all tool calls for a specific round.
        
        Args:
            round_obj: Current round object
            response: API response containing tool calls
            tool_manager: Manager to execute tools
            
        Returns:
            Updated round object with tool execution results
        """
        tool_calls = response["message"]["tool_calls"]
        
        print(f"🔄 Round {round_obj.round_number}: AI decided to use {len(tool_calls)} tool(s)")
        
        for tool_call in tool_calls:
            try:
                function_name = tool_call["function"]["name"]
                function_args = tool_call["function"]["arguments"]
                
                print(f"   🔧 Executing {function_name} with args: {function_args}")
                
                # Execute tool
                tool_result = tool_manager.execute_tool(function_name, **function_args)
                
                # Track tool execution
                round_obj.tools_called.append({
                    "name": function_name,
                    "args": function_args,
                    "tool_call_id": tool_call.get("id", "")
                })
                round_obj.tool_results.append(tool_result)
                
                print(f"   ✅ Tool result length: {len(str(tool_result))} chars")
                
            except Exception as e:
                error_msg = f"Error executing {function_name}: {str(e)}"
                round_obj.error = error_msg
                round_obj.tool_results.append(error_msg)
                print(f"   ❌ {error_msg}")
        
        return round_obj
    
    def _build_tool_result_messages(self, round_obj: ToolCallRound) -> List[Dict]:
        """
        Build tool result messages for conversation history.
        
        Args:
            round_obj: Round object with tool execution results
            
        Returns:
            List of tool result messages
        """
        messages = []
        
        for i, tool_call in enumerate(round_obj.tools_called):
            tool_result = round_obj.tool_results[i] if i < len(round_obj.tool_results) else "No result"
            
            messages.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_call.get("tool_call_id", "")
            })
        
        return messages
    
    def _generate_final_response(self, session: ToolCallSession) -> str:
        """
        Generate final response after all rounds complete.
        
        Args:
            session: Complete tool calling session
            
        Returns:
            Final synthesized response
        """
        # Check if ANY tools were used in ANY round
        any_tools_used = any(round_obj.has_tool_calls for round_obj in session.rounds)
        
        if any_tools_used:
            # Need final call without tools to get synthesized response
            final_params = {
                **self.base_params,
                "messages": session.messages
                # Note: No tools provided for final synthesis
            }
            
            final_response = self.client.chat(**final_params)
            return final_response["message"]["content"]
        else:
            # No tools were used, return the response from the first (and likely only) round
            first_round = session.rounds[0] if session.rounds else None
            return first_round.ai_response["message"]["content"] if first_round else "No response generated"
    
    def _log_session_summary(self, session: ToolCallSession):
        """
        Log summary of the multi-round session.
        
        Args:
            session: Completed tool calling session
        """
        if not session.rounds:
            print("🧠 AI responded without using any tools (general knowledge)")
            return
        
        total_tools = sum(len(round_obj.tools_called) for round_obj in session.rounds if round_obj.has_tool_calls)
        
        if total_tools == 0:
            print("🧠 AI responded without using any tools (general knowledge)")
        else:
            print(f"🔄 Multi-round session completed:")
            print(f"   Total rounds: {len(session.rounds)}")
            print(f"   Total tools used: {total_tools}")
            print(f"   Termination: {session.termination_reason}")
            
            for round_obj in session.rounds:
                if round_obj.has_tool_calls:
                    tools_summary = ", ".join([tool["name"] for tool in round_obj.tools_called])
                    print(f"   Round {round_obj.round_number}: {tools_summary}")
                else:
                    print(f"   Round {round_obj.round_number}: Final response (no tools)")
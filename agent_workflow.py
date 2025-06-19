# agent_workflow.py
import os
import re  # Added missing import
from typing import TypedDict, Annotated, List, Union, Dict  # Added Dict import
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
# Removed unused import - tool_to_json_callable not available in this version
from dotenv import load_dotenv

from Services.tools import agent_tools # Import our custom tools

# Load environment variables (e.g., for Ollama host if changed from default)
load_dotenv()

# 1. Define the Agent State
# This TypedDict defines the structure of our agent's state, which persists across nodes.
class AgentState(TypedDict):
    """
    Represents the state of our agent's workflow.
    Messages: Conversation history.
    sections_to_analyze: List of document sections requested for analysis.
    document_path: Path to the document being analyzed.
    analysis_results: Dictionary to store results from sentiment analysis.
    """
    messages: Annotated[List[BaseMessage], add_messages]  # Fixed: Added List[BaseMessage] and correct brackets
    sections_to_analyze: List[str]
    document_path: str
    analysis_results: Dict[str, Dict]  # Fixed: Added proper type parameters

# 2. Configure the Local LLM
# Using Ollama for our local LLM
llm = ChatOllama(model="qwen2:7b", temperature=0)

# Bind tools to the LLM so it knows how to use them
llm_with_tools = llm.bind_tools(agent_tools)

# 3. Define the Nodes of the Graph

def call_llm(state: AgentState):
    """
    Node to invoke the LLM with the current messages and tools.
    The LLM decides whether to respond directly or call a tool.
    """
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def call_tool(state: AgentState):
    """
    Node to execute the tool called by the LLM.
    Handles tool invocation and returns the result.
    """
    last_message = state['messages'][-1]
    tool_calls = last_message.tool_calls
    
    if not tool_calls:
        # This should ideally not happen if LLM correctly calls a tool
        return {"messages": [HumanMessage(content="No tool calls detected.")]}

    tool_outputs = []  # Fixed: Added missing initialization
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Find the tool function by name
        tool_function = next((t for t in agent_tools if t.name == tool_name), None)
        if tool_function:
            try:
                output = tool_function.invoke(tool_args)
                tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_call['id']))
            except Exception as e:
                tool_outputs.append(ToolMessage(content=f"Tool '{tool_name}' failed: {e}", tool_call_id=tool_call['id']))
        else:
            tool_outputs.append(ToolMessage(content=f"Tool '{tool_name}' not found.", tool_call_id=tool_call['id']))
            
    return {"messages": tool_outputs}

# Define a helper function for conditional routing based on LLM's response
def should_continue(state: AgentState) -> str:
    """
    Determines the next step based on whether the LLM decided to call a tool or respond directly.
    """
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "continue_tool_use"
    else:
        return "end_conversation"

def parse_user_query_and_set_state(state: AgentState) -> AgentState:
    """
    Initial node to parse the user's query and set the initial state variables.
    This simulates an initial 'perception' step.
    """
    user_query = state['messages'][-1].content
    
    # Simple regex to extract document path and sections
    doc_path_match = re.search(r"document '([^']+)'", user_query)
    doc_path_match = "Doc/report.txt"
    sections_match = re.search(r"sections of the document '([^']+)' and summarize", user_query)
    
    # document_path = doc_path_match.group(1) if doc_path_match else "Doc/report.txt" # Default
    document_path = doc_path_match
    sections_str = sections_match.group(1) if sections_match else "Introduction, Conclusion" # Default
    sections_to_analyze = [s.strip() for s in sections_str.split(',')]

    return {
        "document_path": document_path,
        "sections_to_analyze": sections_to_analyze,
        "analysis_results": {}
    }

def orchestrate_analysis(state: AgentState):
    """
    Orchestrates the document reading and sentiment analysis.
    This node will iteratively call tools based on the sections_to_analyze.
    """
    document_path = state["document_path"]
    sections_to_analyze = state["sections_to_analyze"]
    analysis_results = state["analysis_results"]
    messages = state["messages"]

    # Check if all sections have been analyzed
    all_analyzed = True
    for section in sections_to_analyze:
        if section not in analysis_results:
            all_analyzed = False
            break
    
    if all_analyzed and sections_to_analyze:
        # All sections processed, now summarize
        summary_text = "Analysis Summary:\n"
        for section, result in analysis_results.items():
            if "error" not in result:
                summary_text += f"\n{section}:\n"
                summary_text += f"Content: {result['content'][:200]}...\n"
                summary_text += f"Sentiment: {result['sentiment']}\n"
        
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that summarizes document analysis results."),
            ("user", f"Please provide a comprehensive summary of the following analysis results:\n{summary_text}")
        ])
        summary_response = llm.invoke(summary_prompt)
        return {"messages": [summary_response]}
    
    # If not all sections analyzed, find the next one to process
    for section in sections_to_analyze:
        if section not in analysis_results:
            # Plan: Read section -> Analyze sentiment
            # This is a simplified iterative call for demonstration
            # In a more complex graph, this would be handled by a dedicated planning node
            
            try:
                # First, read the section - Import the specific tools
                from Services.tools import read_document_section, analyze_text_sentiment
                
                # Read the section
                read_tool_output = read_document_section.invoke({"file_path": document_path, "section_name": section})
                messages.append(ToolMessage(content=str(read_tool_output), tool_call_id=f"read_{section}"))
                
                if "Error" in str(read_tool_output):
                    analysis_results[section] = {"error": read_tool_output}
                    messages.append(HumanMessage(content=f"Could not read section {section}: {read_tool_output}"))
                    return {"messages": messages, "analysis_results": analysis_results}
                
                # Then, analyze sentiment
                sentiment_output = analyze_text_sentiment.invoke({"text": read_tool_output})
                messages.append(ToolMessage(content=str(sentiment_output), tool_call_id=f"sentiment_{section}"))
                
                analysis_results[section] = {"content": read_tool_output, "sentiment": sentiment_output}
                messages.append(HumanMessage(content=f"Analyzed sentiment for section '{section}': {sentiment_output}"))
                
            except Exception as e:
                analysis_results[section] = {"error": f"Failed to process section: {str(e)}"}
                messages.append(HumanMessage(content=f"Error processing section '{section}': {str(e)}"))
            
            return {"messages": messages, "analysis_results": analysis_results} # Return to re-evaluate state

    return {"messages": messages} # Should not be reached if all_analyzed logic works


# Rebuild graph with refined orchestration
workflow = StateGraph(AgentState)
workflow.add_node("parse_query", parse_user_query_and_set_state)
workflow.add_node("orchestrate_analysis", orchestrate_analysis)
workflow.add_node("llm_agent", call_llm) # This node will now handle final synthesis if orchestrate_analysis is done
workflow.add_node("tool_executor", call_tool) # This node is for general tool calls

# Entry point
workflow.set_entry_point("parse_query")

# Define edges
workflow.add_edge("parse_query", "orchestrate_analysis")

# Conditional routing from orchestrate_analysis:
# If more sections need processing, loop back. Otherwise, go to llm_agent for final synthesis.
# This requires a conditional edge that checks if 'next_section_to_process' was found.
# Since orchestrate_analysis_node directly returns the updated state,
# we need a router function to decide the next step.

def route_analysis_flow(state: AgentState) -> str:
    sections_to_analyze = state["sections_to_analyze"]
    analysis_results = state["analysis_results"]
    
    # Check if any section is still unanalyzed
    for section in sections_to_analyze:
        if section not in analysis_results:
            return "continue_analysis" # Loop back to orchestrate_analysis
    
    return "final_synthesis" # Move to LLM agent for final summary

workflow.add_conditional_edges(
    "orchestrate_analysis",
    route_analysis_flow,
    {
        "continue_analysis": "orchestrate_analysis", # Loop back
        "final_synthesis": "llm_agent", # Go to LLM for final summary
    },
)

# After llm_agent generates the final summary, the workflow ends.
workflow.add_edge("llm_agent", END)

# Compile the graph
app = workflow.compile()

# 5. Run the Agent
if __name__ == "__main__":
    # Create the report.txt if it doesn't exist for testing
    if not os.path.exists("Doc/report.txt"):
        with open("Doc/report.txt", "w") as f:
            f.write("""# report.txt
## Introduction
The initial phase of the project was met with significant enthusiasm and positive feedback from early adopters. The team's innovative approach to problem-solving led to several breakthroughs, setting a strong foundation for future development. We anticipate a smooth transition to the next stage.

## Key Findings
Our analysis revealed a strong correlation between user engagement and feature adoption. Data from the first quarter indicates a 20% increase in active users, surpassing our initial projections. This success is largely attributed to the intuitive user interface and robust backend infrastructure. However, some minor performance bottlenecks were identified in the data processing pipeline, which require optimization.

## Challenges
Despite the overall positive outlook, the project faced unexpected challenges related to resource allocation and integration with legacy systems. These issues caused minor delays and some frustration among the development team. Addressing these will be critical for maintaining momentum.

## Conclusion
In conclusion, the project has demonstrated remarkable progress and potential. While there were hurdles, the team's resilience and strategic adjustments ensured that objectives remain within reach. We are optimistic about the project's long-term impact and its ability to deliver substantial value to our stakeholders. The future looks bright, despite the initial setbacks.
""")

    # User query
    user_query = "Analyze the sentiment of the 'Introduction' and 'Conclusion' sections of the document 'report.txt' and summarize the key findings."
    
    # Initial state for the agent
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "sections_to_analyze": [], # Will be populated by parse_query node
        "document_path": "",       # Will be populated by parse_query node
        "analysis_results": {}
    }

    print(f"User Query: {user_query}\n")
    print("--- Agent Workflow Execution ---")

    # Stream the execution to see intermediate steps
    for s in app.stream(initial_state):
        if "__end__" not in s:
            print(s)
            print("---")
    
    # Get the final state to see the complete conversation and results
    final_state = app.invoke(initial_state)
    print("\n--- Final Agent Response ---")
    print(final_state['messages'][-1].content)
    print("\n--- Full Analysis Results ---")
    import json
    print(json.dumps(final_state['analysis_results'], indent=2))
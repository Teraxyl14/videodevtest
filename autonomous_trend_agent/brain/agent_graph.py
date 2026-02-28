# brain/agent_graph.py
import os
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from .schemas import AgentState, EditDecision

# Config
from dotenv import load_dotenv
load_dotenv() # Load from .env file
# os.environ["GOOGLE_API_KEY"] is now set from .env if present

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# --- Nodes ---

def editor_node(state: Dict[str, Any]):
    print(f"--- Editor Node (Iteration {state.get('iteration_count', 0)}) ---")
    transcript = state['transcript']
    feedback = state.get('critique_feedback', "")
    
    prompt = f"""
    You are a viral YouTube Shorts Editor. 
    Analyze the transcript and create an Edit Decision List (EDL) for a 45s compiled short.
    
    Transcript: {transcript[:5000]}...
    
    Previous Feedback (if any): {feedback}
    
    Return a VALID JSON matching the EditDecision schema.
    """
    
    # In a real impl, we'd use structured output parsing
    # For now, simulation stub:
    plan = EditDecision(
        clips=[
            {"start_time": 10.0, "end_time": 25.0, "transcript_text": "Clip 1", "relevance_score": 0.9},
            {"start_time": 40.0, "end_time": 55.0, "transcript_text": "Clip 2", "relevance_score": 0.85}
        ],
        hook_text="You won't believe this!",
        visual_style="hype",
        music_vibe="lofi_beat"
    )
    
    return {
        "current_plan": plan,
        "iteration_count": state.get('iteration_count', 0) + 1
    }

def critic_node(state: Dict[str, Any]):
    print("--- Critic Node ---")
    plan = state['current_plan']
    
    # Simulation: specific feedback
    if state['iteration_count'] < 2:
        return {
            "critique_feedback": "The hook is too generic. Make it more provocative. Also clip 1 is too long.",
            "is_approved": False
        }
    else:
        return {
            "critique_feedback": "Looks great. Approved.",
            "is_approved": True
        }

# --- Conditional Logic ---

def should_continue(state: Dict[str, Any]):
    if state['is_approved']:
        return "approved"
    if state['iteration_count'] > 3: # Max retries
        return "max_retries"
    return "rejected"

# --- Graph Construction ---

workflow = StateGraph(Dict[str, Any]) # Using simple dict for state wrapper

workflow.add_node("editor", editor_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("editor")

workflow.add_edge("editor", "critic")

workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "approved": END,
        "max_retries": END,
        "rejected": "editor"
    }
)

app = workflow.compile()

if __name__ == "__main__":
    # Test Run
    initial_state = {
        "video_path": "test.mp4",
        "transcript": "This is a long test transcript about AI agents...",
        "iteration_count": 0
    }
    for output in app.stream(initial_state):
        pass
    print("Graph execution finished.")

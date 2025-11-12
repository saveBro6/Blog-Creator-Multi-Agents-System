import streamlit as st
import operator
from typing import Annotated, Literal, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_ollama.llms import OllamaLLM
import time


# Initialize LLM
@st.cache_resource
def get_llm():
    return OllamaLLM(model="llama3.2", temperature=0.7)

llm = get_llm()

# State definition
class ContentState(TypedDict):
    """State for content creation system"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    agent_results: dict
    topic: str
    research_done: bool
    outline_done: bool
    draft_done: bool
    review_done: bool
    iteration_count: int

# Agent functions
def research_agent(state: ContentState) -> Command[Literal["outline_agent", "writer_agent"]]:
    """Agent that researches the topic"""
    st.session_state.current_agent = "Research Agent"
    st.session_state.logs.append({
        "agent": "🔍 Research Agent",
        "status": "working",
        "message": "Researching the topic..."
    })
    
    messages = state["messages"]
    topic = state.get("topic")
    
    research_prompt = f"""You are a Research Agent. Research the topic: {topic}.
    Provide 3-4 key points and interesting facts.
    Keep it concise but informative."""
    
    research_result = llm.invoke([
        SystemMessage(content=research_prompt),
        *messages
    ])
    
    # Decide next agent
    if not state.get("outline_done", False):
        next_agent = "outline_agent"
        route_msg = "Routing to Outline Agent"
    else:
        next_agent = "writer_agent"
        route_msg = "Routing to Writer Agent"
    
    st.session_state.logs.append({
        "agent": "🔍 Research Agent",
        "status": "success",
        "message": f"Research completed. {route_msg}"
    })
    
    return Command(
        goto=next_agent,
        update={
            "messages": [research_result],
            "agent_results": {"research_result": research_result},
            "research_done": True
        }
    )

def outline_agent(state: ContentState) -> Command[Literal["writer_agent", "research_agent"]]:
    """Agent that creates content outline"""
    st.session_state.current_agent = "Outline Agent"
    st.session_state.logs.append({
        "agent": "📝 Outline Agent",
        "status": "working",
        "message": "Creating structured outline..."
    })
    
    messages = state["messages"]
    
    outline_prompt = """You are an Outline Agent. Based on the research, create a structured outline.
    Include: Introduction, Main Points (3-4), Conclusion.
    Format as bullet points."""
    
    outline_result = llm.invoke([
        SystemMessage(content=outline_prompt),
        *messages
    ])
    
    # Decide next agent
    if not state.get("research_done", False):
        next_agent = "research_agent"
        route_msg = "Need more research, routing to Research Agent"
    else:
        next_agent = "writer_agent"
        route_msg = "Routing to Writer Agent"
    
    st.session_state.logs.append({
        "agent": "📝 Outline Agent",
        "status": "success",
        "message": f"Outline completed. {route_msg}"
    })
    
    return Command(
        goto=next_agent,
        update={
            "messages": [outline_result],
            "agent_results": {
                **state["agent_results"],
                "outline_result": outline_result
            },
            "outline_done": True
        }
    )

def writer_agent(state: ContentState) -> Command[Literal["editor_agent", "research_agent"]]:
    """Agent that writes the content"""
    st.session_state.current_agent = "Writer Agent"
    st.session_state.logs.append({
        "agent": "✍️ Writer Agent",
        "status": "working",
        "message": "Writing blog post..."
    })
    
    messages = state["messages"]
    
    writer_prompt = """You are a Writer Agent. Based on the outline and research, write a compelling blog post. Do not include anything else.
    Make it engaging, clear, and well-structured. Aim for 200-300 words."""
    
    draft_result = llm.invoke([
        SystemMessage(content=writer_prompt),
        *messages
    ])
    
    next_agent = "editor_agent"
    
    st.session_state.logs.append({
        "agent": "✍️ Writer Agent",
        "status": "success",
        "message": "Draft completed. Routing to Editor Agent for review"
    })
    
    return Command(
        goto=next_agent,
        update={
            "messages": [draft_result],
            "agent_results": {
                **state["agent_results"],
                "content_result": draft_result
            },
            "draft_done": True
        }
    )

def editor_agent(state: ContentState) -> Command[Literal["writer_agent", "research_agent", END]]:
    """Agent that reviews and provides feedback"""
    st.session_state.current_agent = "Editor Agent"
    st.session_state.logs.append({
        "agent": "👀 Editor Agent",
        "status": "working",
        "message": "Reviewing content..."
    })
    
    messages = state["messages"]
    iteration = state.get("iteration_count", 0)
    
    editor_prompt = """You are an Editor Agent. Review the blog post for:
    1. Clarity and flow
    2. Grammar and style
    3. Engagement factor
    
    Choose 1 of 2:
        If it's good and no needs to revise nor suggestions, respond with: "APPROVED! [brief comment (2-3 sentences)]"
        If there are any suggestions to further enhance, respond with: "REVISE: [specific feedback]"
    """
    
    review_result = llm.invoke([
        SystemMessage(content=editor_prompt),
        *messages
    ])
    
    review_text = review_result
    
    # Decide next step
    if "APPROVED" in review_text or iteration >= 2:
        next_agent = END
        st.session_state.logs.append({
            "agent": "👀 Editor Agent",
            "status": "approved",
            "message": "✅ Content APPROVED! Ready for publication."
        })
    elif "REVISE" in review_text and "research" in review_text.lower():
        next_agent = "research_agent"
        st.session_state.logs.append({
            "agent": "👀 Editor Agent",
            "status": "warning",
            "message": "Needs more research. Routing to Research Agent"
        })
    else:
        next_agent = "writer_agent"
        st.session_state.logs.append({
            "agent": "👀 Editor Agent",
            "status": "warning",
            "message": "Needs revision. Routing back to Writer Agent"
        })

    
    return Command(
        goto=next_agent,
        update={
            "messages": [review_result],
            "agent_results": {
                **state["agent_results"],
                "review_result": review_result
            },
            "review_done": True,
            "iteration_count": iteration + 1
        }
    )

# Build graph
@st.cache_resource
def build_network_graph():
    builder = StateGraph(ContentState)
    
    builder.add_node("research_agent", research_agent)
    builder.add_node("outline_agent", outline_agent)
    builder.add_node("writer_agent", writer_agent)
    builder.add_node("editor_agent", editor_agent)
    
    builder.add_edge(START, "research_agent")
    
    return builder.compile()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Multi-Agent Blog Assistant",
        page_icon="📝",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            background-color: #2563eb;
            color: white;
            font-weight: 600;
        }
        .agent-card {
            padding: 1rem;
            border-radius: 0.5rem;
            border: 2px solid #e2e8f0;
            margin: 0.5rem 0;
        }
        .agent-working {
            border-color: #3b82f6;
            background-color: #1e3a8a;
            color: #ffffff;
        }
        .agent-success {
            border-color: #22c55e;
            background-color: #14532d;
            color: #ffffff;
        }
        .agent-warning {
            border-color: #f59e0b;
            background-color: #78350f;
            color: #ffffff;
        }
        .agent-approved {
            border-color: #10b981;
            background-color: #064e3b;
            color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'current_agent' not in st.session_state:
        st.session_state.current_agent = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Header
    st.markdown('<h1 class="main-title">📝 Blog Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by LangGraph Network Architecture")
    st.markdown("---")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        topic = st.text_input(
            "Enter your blog topic:",
            placeholder="e.g., Artificial Intelligence, Climate Change, Space Exploration...",
            disabled=st.session_state.processing,
            key="topic_input",
            label_visibility="collapsed"
        )
    
    with col2:
        generate_btn = st.button(
            "🚀 Generate Content",
            disabled=st.session_state.processing or not topic,
            use_container_width=True
        )
    
    if generate_btn and topic:
        st.session_state.processing = True
        st.session_state.logs = []
        st.session_state.result = None
        st.session_state.current_agent = None
        
        # Create placeholders
        status_placeholder = st.empty()
        logs_placeholder = st.empty()
        result_placeholder = st.empty()
        
        with status_placeholder.container():
            st.info("🔄 Initializing agents...")
        
        # Build graph and run
        graph = build_network_graph()
        
        initial_state = {
            "messages": [HumanMessage(content=f"Create a blog post about {topic}")],
            "topic": topic,
            "research_done": False,
            "outline_done": False,
            "draft_done": False,
            "review_done": False,
            "iteration_count": 0,
            "agent_results": {}
        }
        
        try:
            # Run the graph
            result = graph.invoke(initial_state)
            st.session_state.result = result
            st.session_state.processing = False
            
            status_placeholder.success("✅ Content generation completed!")
            
        except Exception as e:
            status_placeholder.error(f"❌ Error: {str(e)}")
            st.session_state.processing = False
    
    # Display logs
    if st.session_state.logs:
        st.markdown("### 📋 Activity Log")
        
        for log in st.session_state.logs:
            status_class = f"agent-{log['status']}"
            st.markdown(
                f'<div class="agent-card {status_class}">'
                f'<strong>{log["agent"]}</strong>: {log["message"]}'
                f'</div>',
                unsafe_allow_html=True
            )
    
    # Display results
    if st.session_state.result:
        st.markdown("---")
        st.markdown("## 📄 Generated Content")
        
        agent_results = st.session_state.result.get('agent_results', {})
        
        # Main content
        if 'content_result' in agent_results:
            st.markdown("### ✅ Final Blog Post")
            st.markdown(
                f'<div style="background-color: #f8fafc; padding: 2rem; '
                f'border-radius: 0.5rem; border-left: 4px solid #2563eb; color: #000000;">'
                f'{agent_results["content_result"]}'
                f'</div>',
                unsafe_allow_html=True
            )
        
        # Expandable sections for intermediate results
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'research_result' in agent_results:
                with st.expander("🔍 Research Notes"):
                    st.write(agent_results['research_result'])
        
        with col2:
            if 'outline_result' in agent_results:
                with st.expander("📝 Content Outline"):
                    st.write(agent_results['outline_result'])
        
        with col3:
            if 'review_result' in agent_results:
                with st.expander("👀 Editor Review"):
                    st.write(agent_results['review_result'])
        
        # Download button
        if 'content_result' in agent_results:
            st.download_button(
                label="📥 Download Blog Post",
                data=agent_results['content_result'],
                file_name=f"blog_post_{topic.replace(' ', '_')}.txt",
                mime="text/plain"
            )
    
    # Sidebar with info
    with st.sidebar:
        st.markdown("## 🎯 How It Works")
        st.markdown("""
        This app uses a multi-agent system with dynamic routing:
        
        1. **🔍 Research Agent** - Gathers key information
        2. **📝 Outline Agent** - Creates structure
        3. **✍️ Writer Agent** - Writes the content
        4. **👀 Editor Agent** - Reviews and approves
        
        Agents can dynamically route to each other based on content quality!
        """)
        
        st.markdown("---")
        st.markdown("## ⚙️ Configuration")
        st.markdown(f"**Model:** llama3.2")
        st.markdown(f"**Temperature:** 0.7")
        st.markdown(f"**Max Iterations:** 3")
        
        st.markdown("---")
        st.markdown("### 🔄 Agent Status")
        if st.session_state.current_agent:
            st.success(f"Currently: {st.session_state.current_agent}")
        else:
            st.info("Ready to start")

if __name__ == "__main__":
    main()
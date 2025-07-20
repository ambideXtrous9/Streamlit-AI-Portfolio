# langgraph_multiagent.py
from langchain.agents import Tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AIMessage  # import AIMessage
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from langchain_core.tools import tool
import torch
import streamlit as st 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 

load_dotenv()

ddg_search = DuckDuckGoSearchResults()


os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

model_name = "qwen/qwen3-32b"
temperature = 0.7


llm = ChatGroq(
    model_name=model_name,
    temperature=temperature
)   


ddg_search_tool = Tool(
    name="DuckDuckGoSearch",
    func=ddg_search.run,  # Uses the standard `.run()` interface
    description=(
        "Use this tool to perform a DuckDuckGo web search and return JSON-formatted results. "
        "Input: a search query string; Output: a JSON array of search results."
    )
)


# ---------------------------
# 🔄 State
# ---------------------------
class AgentState(TypedDict):
    topic: str
    research: Optional[str]
    mythology : Optional[str]
    draft: Optional[str]
    critique: Optional[str]
    approved: Optional[bool]
    review: Optional[str]



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Initialize BAAI embeddings with GPU support
embedmodel = "BAAI/bge-base-en" # You can also use bge-base for smaller but faster model
model_kwargs = {'device': device}
encode_kwargs = {'batch_size': 128, 'device': device, 'normalize_embeddings': True}


embeddings = HuggingFaceEmbeddings(
    model_name=embedmodel,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


# Load BGE reranker model and tokenizer once (outside the function)
reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
reranker_model = AutoModel.from_pretrained("BAAI/bge-reranker-base")


vectordb_vectr = FAISS.load_local(
        "HPVdb", 
        embeddings, 
        allow_dangerous_deserialization=True
    )


@tool
def retrieve_context(query,n_docs=5):

    """Retrieve relevant documents from Harry Potter books to answer query"""

    # Step 1: Initial similarity search
    retrieved_docs = vectordb_vectr.similarity_search(query, k=10)  # get more docs for reranking
    pairs = [(query, doc.page_content) for doc in retrieved_docs]

    # Step 2: Format input for BGE reranker (BGE expects "[CLS] query [SEP] passage [SEP]")
    texts = [f"[CLS] {q} [SEP] {p} [SEP]" for q, p in pairs]
    inputs = reranker_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Step 3: Compute relevance scores
    with torch.no_grad():
        model_output = reranker_model(**inputs)
        scores = model_output.last_hidden_state[:, 0, :]  # CLS token
        scores = torch.nn.functional.normalize(scores, p=2, dim=1)  # normalize if needed
        scores = scores[:, 0]  # Take only the first dim for ranking (simplified relevance proxy)

    # Step 4: Sort documents by score
    reranked = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc.page_content for _, doc in reranked[:n_docs]]

    return top_docs



# ---------------------------
# 🧑‍🔬 Researcher Agent
# ---------------------------
def researcher_node(state: AgentState) -> AgentState:

    agent = create_react_agent(
        model=llm,
        tools=[retrieve_context],
         prompt=(
            "You are a research assistant. "
            "For each query, you MUST use both tools in a logical sequence:\n"
            "1) Use **retrieve_context** to fetch internal or domain-specific context first.\n"
            "Always think through your reasoning, call a tool, observe the result, then respond or call the next tool.\n"
            "Label each action clearly as `Thought:`, `Action:`, `Observation:`."
        )
    )

    # Prepare the prompt
    user_msg = {"role": "user", "content": f"Research this topic in detail: {state['topic']}"}

    ai_content = ""
    for step in agent.stream({"messages": [user_msg]}, stream_mode="values"):
        msg = step["messages"][-1]
        with st.chat_message("Agent"):
            st.markdown(msg.content)

        # Capture only if it's an assistant message
        if isinstance(msg, AIMessage):
            ai_content = msg.content
            
    return {**state, "research": ai_content}



# ---------------------------
# 🧑‍🔬 Researcher Agent
# ---------------------------
def mythology_node(state: AgentState) -> AgentState:

    agent = create_react_agent(
        model=llm,
        tools=[ddg_search_tool],
          prompt=(
            "You are an expert in **Indian ancient history and mythology**. "
            "Mix and Relate topic and research with Indian Mythology and Harry Potter Universe"
        )
    )

    # Prepare the prompt
    user_msg = {
        "role": "user",
        "content": (
            f"Using the research, topic relate to Indian Mythology:\n\n"
            f"Research : {state['research']}\n\nTopic: {state['topic']}"
        )
    }


    ai_content = ""
    for step in agent.stream({"messages": [user_msg]}, stream_mode="values"):
        msg = step["messages"][-1]
        with st.chat_message("Agent"):
            st.markdown(msg.content)
        # Capture only if it's an assistant message
        if isinstance(msg, AIMessage):
            ai_content = msg.content
            
    return {**state, "mythology": ai_content}



# ---------------------------
# ✍️ Writer Agent
# ---------------------------
def writer_node(state: AgentState) -> AgentState:
    agent = create_react_agent(
        model=llm,
        tools = [ddg_search_tool],
        prompt=(
            "You are an Article Writer. Use the `DuckDuckGoSearch` to find additional relevant facts, "
            "Mix and Relate topic and research with Indian Mythology and Harry Potter Universe"
            "Follow the ReAct pattern: label each step as `Thought:`, `Action:`, `Observation:`, "
            "then finally `Final Answer:` with your article."
        )

    )

    user_msg = {
        "role": "user",
        "content": (
            f"Using the research below, write a well-structured article on the topic:\n\n"
            f"Review from Critique: {state['review']}\n\n Research : {state['mythology']}\n\nTopic: {state['topic']}"
        )
    }

    ai_content = ""
    for step in agent.stream({"messages": [user_msg]}, stream_mode="values"):
        msg = step["messages"][-1]
        with st.chat_message("Agent"):
            st.markdown(msg.content)
        # Capture only if it's an assistant message
        if isinstance(msg, AIMessage):
            ai_content = msg.content

    return {**state, "draft": ai_content}


# ---------------------------
# 🧑‍⚖️ Critic Agent
# ---------------------------
def critic_node(state: AgentState) -> AgentState:
    agent = create_react_agent(
        model=llm,
        tools = [],
        prompt=(
            "You are a Critical Reviewer having Knowledge in Both Harry Potter Universe and Indian Mythology."
            "First Give your 'approval' by saying 'Yes' or 'No' by reading the draft."
            "Use Your Intelligence to evaluate the draft and Give your Comments and Reasoning."
        )
    )

    user_msg = {
        "role": "user",
        "content": (
            "Here is the draft article:\n\n"
            f"{state['draft']}\n\n"
            "Please critique it, checking for factual accuracy and clarity."
            "First Give 'approval' by saying 'yes' or 'no' followed by concise reasoning."
        )
    }

    ai_content = ""
    for step in agent.stream({"messages": [user_msg]}, stream_mode="values"):
        msg = step["messages"][-1]
        with st.chat_message("Agent"):
            st.markdown(msg.content)
        # Capture only if it's an assistant message
        if isinstance(msg, AIMessage):
            ai_content = msg.content

    
    approved = "yes" in ai_content.lower()

    return {**state, "critique": ai_content, "approved": approved}


# ---------------------------
# 🔁 Conditional Flow
# ---------------------------
def check_approval(state: AgentState) -> str:
    if state.get("approved"):
        return "end"
    
    else:
        state["review"] = state["critique"]
        return "mythologist"


# ---------------------------
# 🌐 LangGraph Definition
# ---------------------------
graph = StateGraph(AgentState)

graph.add_node("researcher", researcher_node)
graph.add_node("mythologist", mythology_node)

graph.add_node("writer", writer_node)
graph.add_node("critic", critic_node)

graph.set_entry_point("researcher")

graph.add_edge("researcher", "mythologist")
graph.add_edge("mythologist", "writer")
graph.add_edge("writer", "critic")
graph.add_conditional_edges("critic", check_approval, {
    "end": END,
    "mythologist": "mythologist"  
})



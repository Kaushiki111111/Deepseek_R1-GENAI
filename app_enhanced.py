import streamlit as st
import time
import json
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Set Page Config
st.set_page_config(page_title="DeepSeek Code Companion", page_icon="🧠", layout="wide")

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #1a1a1a; color: #ffffff; }
    .sidebar .sidebar-content { background-color: #2d2d2d; }
    .stTextInput textarea, .stSelectbox div[data-baseweb="select"], .stSelectbox svg, .stSelectbox option, div[role="listbox"] div {
        color: white !important; background-color: #3d3d3d !important;
    }
    .fixed-chat-input { position: fixed; bottom: 0; width: 100%; background: #1a1a1a; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "deepseek-r1:1.5b"
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

# Title & Subtitle
st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar Config
with st.sidebar:
    st.header("⚙️ Configuration")
    
    model_options = {
        "deepseek-r1:1.5b": "🤖 DeepSeek 1.5B",
        "deepseek-r1:3b": "🚀 DeepSeek 3B"
    }
    
    selected_model = st.selectbox("Choose Model", list(model_options.keys()), format_func=lambda x: model_options[x])
    st.session_state.selected_model = selected_model
    
    context_window = st.slider("Context Window (Last N Messages)", 2, 20, 5)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]
        st.rerun()
    
    if st.button("📥 Download Chat"):
        chat_history = json.dumps(st.session_state.message_log, indent=4)
        st.download_button("Download Chat History", chat_history, "chat_history.json", "application/json")
    
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initiate LLM Engine with Streaming
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# System Prompt
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Chat Container
chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat Input
st.markdown('<div class="fixed-chat-input">', unsafe_allow_html=True)
user_query = st.chat_input("Type your coding question here...")
st.markdown('</div>', unsafe_allow_html=True)

# Helper Functions
def display_response_with_typing_effect(response):
    message_placeholder = st.empty()
    displayed_text = ""
    for char in response:
        displayed_text += char
        message_placeholder.markdown(displayed_text)
        time.sleep(0.02)  # Typing effect speed
    return displayed_text

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log[-context_window:]:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

# Process User Input
if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    st.session_state.message_log.append({"role": "ai", "content": display_response_with_typing_effect(ai_response)})
    st.rerun()

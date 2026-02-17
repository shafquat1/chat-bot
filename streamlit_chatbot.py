"""
Streamlit Web App for AI Chatbot
A beautiful web interface for the Groq-powered chatbot
"""

import streamlit as st
from chatbot import AIChat
import os

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .main {
        padding: 2rem;
    }
    div[data-testid="stExpander"] {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key input
    st.subheader("🔑 API Key")
    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get your API key from https://console.groq.com/keys"
    )
    
    # Model selection
    st.subheader("🤖 Model Selection")
    model = st.selectbox(
        "Choose a model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        help="Different models have different capabilities and speeds"
    )
    
    # System prompt
    st.subheader("📝 System Prompt")
    use_system_prompt = st.checkbox("Use custom system prompt")
    
    system_prompt = ""
    if use_system_prompt:
        system_prompt = st.text_area(
            "System Prompt",
            value="You are a helpful AI assistant.",
            height=150,
            help="Guide the AI's behavior and personality"
        )
    
    # Temperature setting
    st.subheader("🌡️ Settings")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random, lower values more focused"
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=4096,
        value=1024,
        step=256,
        help="Maximum length of the response"
    )
    
    st.divider()
    
    # Initialize chatbot button
    if st.button("🚀 Initialize Chatbot", type="primary", use_container_width=True):
        if not api_key_input:
            st.error("Please enter your Groq API key!")
        else:
            try:
                st.session_state.chatbot = AIChat(api_key=api_key_input, model=model)
                st.session_state.api_key_set = True
                st.success(f"✓ Chatbot initialized with {model}!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chatbot:
            st.session_state.chatbot.clear_history()
        st.rerun()
    
    st.divider()
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Enter your Groq API key** (get one at [console.groq.com](https://console.groq.com/keys))
        2. **Choose a model** from the dropdown
        3. **Optional:** Set a custom system prompt
        4. **Click "Initialize Chatbot"**
        5. **Start chatting!**
        
        **Tips:**
        - Use system prompts to customize AI behavior
        - Adjust temperature for creativity
        - Clear conversation to start fresh
        """)
    
    with st.expander("🔗 Links"):
        st.markdown("""
        - [Groq Console](https://console.groq.com/)
        - [Groq Documentation](https://console.groq.com/docs)
        - [GitHub Repository](https://github.com)
        """)

# Main chat interface
st.title("🤖 AI Chatbot")
st.caption("Powered by Groq's Lightning-Fast Inference")

# Check if chatbot is initialized
if not st.session_state.api_key_set or st.session_state.chatbot is None:
    st.info("👈 Please configure and initialize the chatbot in the sidebar to get started!")
    
    # Show some example use cases
    st.subheader("✨ What you can do:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **💬 General Chat**
        - Ask questions
        - Get explanations
        - Have conversations
        """)
    
    with col2:
        st.markdown("""
        **💻 Code Help**
        - Debug code
        - Learn programming
        - Get code examples
        """)
    
    with col3:
        st.markdown("""
        **✍️ Content Creation**
        - Write stories
        - Generate ideas
        - Create outlines
        """)
    
    st.divider()
    
    st.subheader("🚀 Quick Start")
    st.code("""
# 1. Get your API key
Visit: https://console.groq.com/keys

# 2. Enter it in the sidebar
# 3. Click "Initialize Chatbot"
# 4. Start chatting!
    """, language="bash")

else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Update chatbot settings
                st.session_state.chatbot.model = model
                
                # Stream the response
                for chunk in st.session_state.chatbot.stream_chat(
                    prompt, 
                    system_prompt if use_system_prompt else None
                ):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                message_placeholder.error(error_msg)
                full_response = error_msg
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.divider()
st.caption("Built with Streamlit and Groq API | Responses may take a few seconds to generate")
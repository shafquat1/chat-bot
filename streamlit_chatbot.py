"""
Streamlit Web App for AI Chatbot
Powered by Groq — API key managed via Streamlit Secrets (never exposed to users)
"""

import streamlit as st
from groq import Groq
import os

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 0.5rem; }
    .main { padding: 2rem; }
    div[data-testid="stExpander"] { background-color: #f0f2f6; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Load API key from Streamlit Secrets ────────────────────────────────────────
def get_groq_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        st.error(
            "⚠️ Groq API key not found. "
            "Add `GROQ_API_KEY` to `.streamlit/secrets.toml` or Streamlit Cloud secrets."
        )
        st.stop()

    return Groq(api_key=api_key)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "client" not in st.session_state:
    st.session_state.client = get_groq_client()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    st.subheader("🤖 Model")
    model = st.selectbox(
        "Choose a model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        help="All models run free on Groq's LPU inference engine.",
    )

    st.subheader("📝 System Prompt")
    use_system_prompt = st.checkbox("Use custom system prompt")
    system_prompt = ""
    if use_system_prompt:
        system_prompt = st.text_area(
            "System Prompt",
            value="You are a helpful AI assistant.",
            height=150,
        )

    st.subheader("🌡️ Settings")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 256, 4096, 1024, 256)

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        Just **start typing** — no API key needed here!

        The key is securely stored in Streamlit Secrets by the app owner.

        **Model guide:**
        | Model | Best for |
        |---|---|
        | llama-3.3-70b | Most capable |
        | llama-3.1-8b | Fastest replies |
        | mixtral-8x7b | Long contexts |
        | gemma2-9b | Balanced |
        """)

    with st.expander("🔒 Security Note"):
        st.markdown("""
        Your API key is **never shown** in this UI.
        It lives in Streamlit Secrets — server-side only.

        [Learn more](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
        """)

    with st.expander("🔗 Links"):
        st.markdown("""
        - [Groq Console (free)](https://console.groq.com/)
        - [Streamlit Secrets Docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
        """)

# ── Main chat interface ────────────────────────────────────────────────────────
st.title("🤖 AI Chatbot")
st.caption("Powered by Groq's Lightning-Fast Inference · Free to use")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_messages = st.session_state.messages.copy()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            stream = st.session_state.client.chat.completions.create(
                model=model,
                messages=(
                    [{"role": "system", "content": system_prompt}] + api_messages
                    if use_system_prompt and system_prompt.strip()
                    else api_messages
                ),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            placeholder.error(error_msg)
            full_response = error_msg

    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.divider()
st.caption("Built with Streamlit & Groq API · Responses stream in real-time")

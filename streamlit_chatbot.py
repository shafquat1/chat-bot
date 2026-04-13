"""
Streamlit Web App for AI Chatbot
Supports Groq, Claude (Anthropic), OpenAI, and Google Gemini
Users supply their own API key — no secrets required.
"""

import streamlit as st

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding: 2rem; }
    div[data-testid="stExpander"] { border-radius: 0.5rem; }

    /* ── Chat bubble layout ─────────────────────────────────────── */

    /* Strip default card styling from every message row */
    [data-testid="stChatMessage"] {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.25rem 1rem;
        align-items: flex-end;
        gap: 0.5rem;
    }

    /* USER → flip row so avatar + bubble sit on the RIGHT */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
    }

    /* USER bubble — blue, tail bottom-right */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background: #0078FF;
        border-radius: 18px 18px 4px 18px;
        padding: 0.65rem 1rem;
        max-width: 70%;
        margin-left: auto;
        margin-right: 0;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] * {
        color: #ffffff !important;
        margin-bottom: 0;
    }

    /* ASSISTANT bubble — subtle dark, tail bottom-left */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
        background: #2a2d3e;
        border-radius: 18px 18px 18px 4px;
        padding: 0.65rem 1rem;
        max-width: 70%;
        margin-right: auto;
        margin-left: 0;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] p {
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

LLM_PROVIDERS = ["Groq", "Claude", "OpenAI", "Google"]

DEFAULT_MODELS = {
    "Groq":   "llama-3.3-70b-versatile",
    "Claude": "claude-sonnet-4-6",
    "OpenAI": "gpt-4o",
    "Google": "gemini-1.5-pro",
}

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "Groq"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    st.subheader("🤖 LLM Provider")
    llm_provider = st.selectbox(
        "Choose a provider",
        LLM_PROVIDERS,
        index=LLM_PROVIDERS.index(st.session_state.llm_provider),
    )

    # Reset API key when provider changes
    if llm_provider != st.session_state.llm_provider:
        st.session_state.llm_provider = llm_provider
        st.session_state.api_key = ""
        st.rerun()

    api_key = st.text_input(
        f"{llm_provider} API Key",
        type="password",
        placeholder=f"Enter your {llm_provider} API key...",
        value=st.session_state.api_key,
    )
    st.session_state.api_key = api_key

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
        1. Select your **LLM provider** from the dropdown.
        2. Paste your **API key** for that provider.
        3. Start chatting!

        **Default models used:**
        | Provider | Model |
        |---|---|
        | Groq | llama-3.3-70b-versatile |
        | Claude | claude-sonnet-4-6 |
        | OpenAI | gpt-4o |
        | Google | gemini-1.5-pro |
        """)

    with st.expander("🔗 Get API Keys"):
        st.markdown("""
        - [Groq Console (free)](https://console.groq.com/)
        - [Anthropic Console](https://console.anthropic.com/)
        - [OpenAI Platform](https://platform.openai.com/)
        - [Google AI Studio](https://aistudio.google.com/)
        """)


# ── Streaming response generator ───────────────────────────────────────────────
def stream_response(provider, api_key, messages, sys_prompt, temperature, max_tokens):
    model = DEFAULT_MODELS[provider]

    if provider == "Groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        api_messages = (
            [{"role": "system", "content": sys_prompt}] + messages
            if sys_prompt else messages
        )
        stream = client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""

    elif provider == "Claude":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            temperature=temperature,
        )
        if sys_prompt:
            kwargs["system"] = sys_prompt
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    elif provider == "OpenAI":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        api_messages = (
            [{"role": "system", "content": sys_prompt}] + messages
            if sys_prompt else messages
        )
        stream = client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""

    elif provider == "Google":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=sys_prompt if sys_prompt else None,
        )
        history = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
            for m in messages[:-1]
        ]
        chat = gen_model.start_chat(history=history)
        response = chat.send_message(
            messages[-1]["content"],
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            stream=True,
        )
        for chunk in response:
            yield chunk.text


# ── Main chat interface ────────────────────────────────────────────────────────
st.title("🤖 AI Chatbot")
st.caption(f"Powered by {st.session_state.llm_provider} · Enter your API key in the sidebar to start")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message here..."):
    if not st.session_state.api_key.strip():
        st.warning(f"Please enter your {st.session_state.llm_provider} API key in the sidebar to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    sys_prompt = system_prompt.strip() if use_system_prompt and system_prompt.strip() else ""

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            for delta in stream_response(
                st.session_state.llm_provider,
                st.session_state.api_key,
                st.session_state.messages,
                sys_prompt,
                temperature,
                max_tokens,
            ):
                full_response += delta
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            placeholder.error(error_msg)
            full_response = error_msg

    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.divider()
st.caption("Built with Streamlit · Supports Groq, Claude, OpenAI & Google Gemini")

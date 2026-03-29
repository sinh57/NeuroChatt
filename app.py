"""
app.py — NeuroChat: Conversational AI with Memory & Tools
Stack : LangChain + LangGraph + Streamlit
"""

from typing import Dict, List

import streamlit as st

from agent.graph import build_agent
from utils.helpers import memory_label, sanitise

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroChat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');
:root {
    --bg: #0a0a0f; --surface: #111118; --border: #1e1e2e;
    --accent: #7c6af7; --accent2: #3ecfcf; --text: #e2e2f0; --muted: #6b6b8a;
}
html,body,[class*="css"] { background:var(--bg)!important; color:var(--text)!important; font-family:'Sora',sans-serif!important; }
section[data-testid="stSidebar"] { background:var(--surface)!important; border-right:1px solid var(--border)!important; }
.user-msg {
    background:#1a1a2e; border:1px solid var(--accent);
    border-radius:12px 12px 4px 12px; padding:12px 16px; margin:6px 0;
    line-height:1.65; box-shadow:0 0 14px rgba(124,106,247,.15);
}
.ai-msg {
    background:#0f1923; border:1px solid var(--border);
    border-left:3px solid var(--accent2);
    border-radius:4px 12px 12px 12px; padding:12px 16px; margin:6px 0; line-height:1.65;
}
.tool-badge {
    display:inline-block; background:rgba(62,207,207,.1);
    border:1px solid var(--accent2); color:var(--accent2);
    font-family:'Space Mono',monospace; font-size:.7rem;
    padding:2px 9px; border-radius:20px; margin:2px 2px 0 0;
}
.mem-chip {
    background:rgba(124,106,247,.1); border:1px solid var(--accent);
    color:var(--accent); font-size:.72rem; padding:3px 10px;
    border-radius:20px; font-family:'Space Mono',monospace;
}
h1 { font-family:'Space Mono',monospace!important; color:var(--accent)!important; letter-spacing:-1px; }
.stButton>button {
    background:var(--accent)!important; color:#fff!important; border:none!important;
    border-radius:8px!important; font-family:'Sora',sans-serif!important;
    font-weight:600!important; transition:all .2s!important;
}
.stButton>button:hover { background:#9b8df9!important; transform:translateY(-1px)!important; box-shadow:0 4px 18px rgba(124,106,247,.45)!important; }
.stTextInput>div>div>input {
    background:var(--surface)!important; border:1px solid var(--border)!important;
    color:var(--text)!important; border-radius:8px!important;
}
.stTextInput>div>div>input:focus { border-color:var(--accent)!important; box-shadow:0 0 0 2px rgba(124,106,247,.25)!important; }
div[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:10px; }
</style>
""", unsafe_allow_html=True)


# ── Resolve API key (sidebar input OR Streamlit secrets) ──────────────────────
def _get_secret_key() -> str:
    """Read API key from st.secrets if available (Streamlit Cloud deployment)."""
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        return ""


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults: Dict = {
        "messages":     [],   # [{role, content, tools_used}]
        "chat_history": [],   # serialised [{role, content}]
        "graph":        None,
        "memory":       None,
        "tool_log":     [],
        "cfg":          {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🧠 NeuroChat")
    st.caption("Conversational AI · Memory · Tools")
    st.divider()

    st.markdown("### ⚙️ Configuration")

    # API key — prefill from secrets if deployed
    secret_key = _get_secret_key()
    api_key = st.text_input(
        "OpenAI API Key",
        value=secret_key,
        type="password",
        placeholder="sk-…",
        help="Your key is never stored. On Streamlit Cloud add it in App Secrets.",
    )

    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

    st.divider()
    st.markdown("### 🛠️ Tools")
    t_search = st.checkbox("🌐 Web Search",  value=True)
    t_calc   = st.checkbox("🔢 Calculator",  value=True)
    t_wiki   = st.checkbox("📚 Wikipedia",   value=True)
    t_time   = st.checkbox("🕐 DateTime",    value=True)
    t_wx     = st.checkbox("🌤️ Weather",     value=True)

    st.divider()
    st.markdown("### 🧬 Memory")
    mem_type = st.selectbox(
        "Memory Type",
        ["ConversationBuffer", "ConversationSummary", "ConversationWindow"],
    )
    window_k = st.slider("Window (k)", 2, 20, 5) if mem_type == "ConversationWindow" else 5

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.chat_history = []
        st.session_state.tool_log     = []
        st.session_state.graph        = None
        st.session_state.memory       = None
        st.session_state.cfg          = {}
        st.rerun()

    st.divider()
    st.markdown("### 📊 Stats")
    c1, c2 = st.columns(2)
    c1.metric("Messages",   len(st.session_state.messages))
    c2.metric("Tool calls", len(st.session_state.tool_log))


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("## 💬 Conversation")

lbl = memory_label(st.session_state.chat_history)
if lbl:
    st.markdown(f'<span class="mem-chip">🧠 {lbl}</span>', unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-msg">👤 <b>You</b><br>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        badges = "".join(
            f'<span class="tool-badge">⚡ {t}</span>'
            for t in msg.get("tools_used", [])
        )
        badge_row = f'<div style="margin-top:8px">{badges}</div>' if badges else ""
        st.markdown(
            f'<div class="ai-msg">🧠 <b>NeuroChat</b><br>{msg["content"]}{badge_row}</div>',
            unsafe_allow_html=True,
        )

st.divider()

# Input row
col_in, col_btn = st.columns([5, 1])
with col_in:
    user_input: str = st.text_input(
        "msg", placeholder="Ask me anything…",
        label_visibility="collapsed", key="input_box",
    )
with col_btn:
    send: bool = st.button("Send →", use_container_width=True)

# Quick prompts
st.markdown("**Try:**")
qcols = st.columns(4)
quick: List[str] = [
    "What's 15% of 847?",
    "Explain LangGraph",
    "Weather in Tokyo?",
    "What day is today?",
]
for i, q in enumerate(quick):
    with qcols[i]:
        if st.button(q, key=f"q{i}", use_container_width=True):
            user_input, send = q, True


# ── Handle send ───────────────────────────────────────────────────────────────
if send and user_input.strip():
    if not api_key:
        st.error("⚠️ Enter your OpenAI API key in the sidebar.")
        st.stop()

    user_input = sanitise(user_input)

    active_tools: List[str] = []
    if t_search: active_tools.append("web_search")
    if t_calc:   active_tools.append("calculator")
    if t_wiki:   active_tools.append("wikipedia")
    if t_time:   active_tools.append("datetime")
    if t_wx:     active_tools.append("weather")

    # Rebuild agent only when config actually changes
    cfg = {
        "model": model, "temperature": temperature,
        "tools": tuple(active_tools), "mem_type": mem_type, "window_k": window_k,
    }
    if st.session_state.graph is None or st.session_state.cfg != cfg:
        with st.spinner("⚙️ Initialising agent…"):
            graph, memory = build_agent(
                api_key=api_key,
                model=model,
                temperature=temperature,
                selected_tools=active_tools,
                memory_type=mem_type,
                window_k=window_k,
            )
        st.session_state.graph  = graph
        st.session_state.memory = memory
        st.session_state.cfg    = cfg

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("🧠 Thinking…"):
        try:
            result = st.session_state.graph.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history,
                "output": "",
                "tools_used": [],
            })
            output: str          = result["output"]
            tools_used: List[str] = result.get("tools_used", [])
            st.session_state.chat_history = result.get(
                "chat_history", st.session_state.chat_history
            )
        except Exception as e:
            output     = f"⚠️ Error: {e}"
            tools_used = []

    st.session_state.tool_log.extend(tools_used)
    st.session_state.messages.append({
        "role": "assistant",
        "content": output,
        "tools_used": tools_used,
    })
    st.rerun()


# ── Tool log ──────────────────────────────────────────────────────────────────
if st.session_state.tool_log:
    with st.expander(f"🔧 Tool activity log ({len(st.session_state.tool_log)} calls)"):
        for i, t in enumerate(st.session_state.tool_log, 1):
            st.markdown(
                f"`{i}.` <span class='tool-badge'>⚡ {t}</span>",
                unsafe_allow_html=True,
            )

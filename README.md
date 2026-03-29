---
title: NeuroChat AI Agent
emoji: 🧠
colorFrom: purple
colorTo: cyan
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
python_version: 3.11
---

# 🧠 NeuroChat — Conversational AI with Memory & Tools

> A production-ready AI agent built with **LangChain**, **LangGraph**, and **Streamlit**.  
> Features persistent conversation memory, 5 integrated tools, and a dark-themed chat UI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square)

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **Persistent Memory** | Buffer, Summary, and Window memory types — switchable at runtime |
| 🔧 **5 Tools** | Web search, calculator, Wikipedia, datetime, live weather |
| 🗺️ **LangGraph Workflow** | Typed `AgentState`, single compiled graph, memory outside graph |
| 💬 **Streamlit UI** | Dark-themed chat, real-time tool badges, session stats |
| ⚙️ **Runtime Config** | Switch model, temperature, tools, memory — no restart needed |
| 🚀 **Deploy-ready** | Streamlit Cloud + HuggingFace Spaces config included |

---

## 🏗️ Architecture

```
neurochat/
├── app.py                  # Streamlit UI + session management
├── agent/
│   ├── graph.py            # LangGraph StateGraph + agent node factory
│   ├── tools.py            # 5 tool implementations
│   └── memory.py           # Memory helpers
├── utils/
│   └── helpers.py          # Shared utilities
├── .streamlit/
│   └── config.toml         # Theme + server config (deploy-ready)
├── requirements.txt
├── .env.example
└── README.md
```

### LangGraph Flow

```
          ┌─────────────────────────────────────────┐
          │            AgentState (TypedDict)        │
          │  input · chat_history · output ·         │
          │  tools_used                              │
          └──────────────┬──────────────────────────┘
                         │
                         ▼
                   [ agent_node ]
                         │
              ┌──────────┴──────────┐
              │   AgentExecutor     │
              │  (OpenAI Tools)     │
              │                     │
              │  Tool calls?        │
              │  ┌───────────────┐  │
              │  │ calculator    │  │
              │  │ web_search    │  │
              │  │ wikipedia     │  │
              │  │ datetime      │  │
              │  │ weather       │  │
              │  └───────────────┘  │
              └──────────┬──────────┘
                         │
              Memory.save_context()
                         │
                        END
```

### Why memory lives outside the graph

LangGraph recompiles the graph on `invoke()` — if memory were created inside, it would reset every turn. By storing the memory object in `st.session_state` and passing it into the node via closure, it survives all Streamlit reruns.

---

## 🚀 Local Setup

```bash
# 1. Clone
git clone https://github.com/yourusername/neurochat
cd neurochat

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo → `app.py` as entry point
4. Under **Advanced settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-your-key"
   ```
5. Click **Deploy** — live in ~2 minutes ✅

---

## 🤗 Deploy to HuggingFace Spaces (free)

The `---` header at the top of this README is the HF Spaces config.

```bash
# Create a new Space at huggingface.co/new-space
# SDK: Streamlit
git remote add hf https://huggingface.co/spaces/yourusername/neurochat
git push hf main
```
Add `OPENAI_API_KEY` in Space → Settings → Repository secrets.

---

## 🧬 Memory Types Explained

| Type | How it works | Best for |
|---|---|---|
| **ConversationBuffer** | Stores every message verbatim | Short–medium chats |
| **ConversationSummary** | LLM summarises old turns | Long conversations |
| **ConversationWindow** | Keeps last *k* turns only | Focused, token-efficient |

---

## 🔧 Tools

| Tool | Source | API key? |
|---|---|---|
| 🌐 Web Search | DuckDuckGo | ❌ Free |
| 🔢 Calculator | Built-in (safe eval) | ❌ |
| 📚 Wikipedia | `wikipedia` library | ❌ Free |
| 🕐 DateTime | `datetime` stdlib | ❌ |
| 🌤️ Weather | wttr.in | ❌ Free |

---

## 📄 Resume Bullets

```
• Built a production-ready conversational AI agent using LangChain and LangGraph
  with a stateful StateGraph, typed AgentState schema, and persistent cross-turn
  memory (Buffer / Summary / Window).

• Integrated 5 tools (DuckDuckGo search, calculator, Wikipedia, datetime, weather)
  via OpenAI function-calling; implemented safe tool degradation so agent never
  crashes on a failed tool.

• Architected memory lifecycle correctly — stored LangChain memory outside the
  LangGraph compile cycle to prevent per-turn reset, a common production pitfall.

• Shipped a Streamlit chat UI with real-time tool-call badges, session stats,
  and runtime model/memory switching; deployed to Streamlit Cloud.
```

---

## 🔮 Roadmap

- [ ] RAG pipeline with FAISS / Chroma vector store
- [ ] Multi-agent graph (Planner → Researcher → Writer)
- [ ] Persistent memory across sessions (SQLite)
- [ ] Voice input/output (Whisper + TTS)
- [ ] Streaming token output

---

## 📜 License

MIT — free to use, fork, and modify.

"""Streamlit dashboard for the LangGraph Crawler Pipeline.

Provides an interactive UI for:
  - Submitting research queries
  - Real-time pipeline status
  - Viewing processed documents
  - Cost breakdown dashboard

Run:
    uv run streamlit run dashboard.py
"""

from __future__ import annotations

import asyncio
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ── Helper: Fetch History ────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_history():
    """Fetch past sessions and their processed documents from MongoDB."""
    import os
    from motor.motor_asyncio import AsyncIOMotorClient

    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", "langgraph_crawler")
    client = AsyncIOMotorClient(uri)
    db = client[db_name]

    async def _fetch():
        # Get last 20 sessions, sorted newest first
        cursor = db.sessions.find().sort("created_at", -1).limit(20)
        sessions = await cursor.to_list(length=20)

        history_data = []
        for sess in sessions:
            sess_id = str(sess["_id"])
            query = sess.get("user_query", "Unknown Query")
            date = sess.get("created_at")
            if date:
                date_str = date.strftime("%Y-%m-%d %H:%M:%S")
            else:
                date_str = "Unknown"

            # Count entities for this session
            entity_count = await db.extracted_entities.count_documents(
                {"session_id": sess_id}
            )

            history_data.append(
                {
                    "Session ID": sess_id,
                    "Date": date_str,
                    "Query": query,
                    "Entities Found": entity_count,
                }
            )

        return history_data

    return run_async(_fetch())


def fetch_session_docs(session_id: str):
    """Fetch processed docs for a specific session."""
    import os
    from motor.motor_asyncio import AsyncIOMotorClient

    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", "langgraph_crawler")
    client = AsyncIOMotorClient(uri)
    db = client[db_name]

    async def _fetch():
        cursor = db.extracted_entities.find({"session_id": session_id}).sort(
            "priority_score", -1
        )
        entities = await cursor.to_list(length=100)

        table_data = []
        for ent in entities:
            row = {
                "Priority": round(ent.get("priority_score", 0.0), 2),
                "Entity Name": ent.get("name", "Unknown"),
                "Source URL": ent.get("source_url", ""),
                "Description": ent.get("description", ""),
            }
            # Flatten metrics into the row
            metrics = ent.get("metrics", {})
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    # Prefix to group them nicely, or just direct name
                    row[k] = str(v)
            table_data.append(row)
        return table_data

    return run_async(_fetch())


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="🕷️ Crawler Pipeline",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown(
    """
<style>
    .main-title {
        color: #2b6cb0;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        color: #4a5568;
        font-size: 1rem;
        margin-top: -10px;
    }
    .node-step {
        padding: 8px 16px;
        border-left: 3px solid #3182ce;
        margin-bottom: 8px;
        background: #ebf8ff;
        border-radius: 0 8px 8px 0;
        color: #2d3748;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Helper: run async in streamlit ───────────────────────────
def run_async(coro):
    """Run an async coroutine from streamlit (sync context)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Sidebar: Configuration ───────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")

    model = st.selectbox(
        "LLM Model",
        [
            "meta/meta-llama-3-8b-instruct",
            "meta/meta-llama-3-70b-instruct",
        ],
        index=0,
    )

    max_search = st.slider("Max search results per query", 3, 20, 10)
    min_words = st.slider("Min word count (quality gate)", 50, 500, 200)
    min_cred = st.slider("Min credibility score", 0.0, 1.0, 0.6, 0.05)
    max_retries = st.slider("Max retries", 0, 5, 2)
    min_docs = st.slider("Min processed docs target", 1, 10, 3)

    st.markdown("---")
    st.markdown("### 📊 Node Pipeline")
    nodes = [
        ("🧠", "Intent Parser", "LLM extracts search queries"),
        ("🔍", "URL Discovery", "Tavily search API"),
        ("🕷️", "Web Crawler", "crawl4ai + httpx fallback"),
        ("✅", "Source Verifier", "Credibility scoring"),
        ("💾", "MongoDB Logger", "Async upsert"),
        ("📝", "Preprocessor", "Summarise & extract"),
    ]
    for icon, name, desc in nodes:
        st.markdown(
            f'<div class="node-step">{icon} <strong>{name}</strong><br/>'
            f'<small style="color: #a0aec0">{desc}</small></div>',
            unsafe_allow_html=True,
        )

# ── Main content ─────────────────────────────────────────────
st.markdown('<p class="main-title">🕷️ Crawler Pipeline</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">LangGraph-powered web research • Replicate LLM • Tavily Search • MongoDB</p>',
    unsafe_allow_html=True,
)

# ── Layout: Tabs ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 New Search", "🕰️ History"])

with tab1:
    # ── Query input ──────────────────────────────────────────────
    query = st.text_area(
        "🔎 Research Query",
        placeholder="e.g., Latest advancements in AI agents and multi-agent systems 2025",
        height=80,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    # ── Pipeline execution ───────────────────────────────────────
    if run_btn and query:
        from crawler.graph import graph
        from crawler.cost_tracker import tracker

        # Reset motor clients so they bind to the fresh event loop
        from crawler.nodes import mongo_logger as _ml, preprocessor as _pp

        _ml._client = None
        _pp._client = None

        # Reset cost tracker for this run
        tracker._calls.clear()

        config = {
            "configurable": {
                "model": model,
                "max_search_results": max_search,
                "min_word_count": min_words,
                "min_credibility": min_cred,
                "max_retries": max_retries,
                "min_processed_docs": min_docs,
            }
        }

        t0 = time.time()

        try:
            with st.spinner("🚀 Running pipeline... (this may take 1-3 minutes)"):
                result = run_async(graph.ainvoke({"user_query": query}, config=config))

            elapsed = time.time() - t0

            entities = result.get("extracted_entities", [])
            cost = result.get("cost_summary", tracker.get_summary())

            # ── Metrics row ──────────────────────────────────
            st.markdown("---")
            st.markdown("### 📊 Results Summary")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("📦 Entities Found", len(entities))
            with m2:
                st.metric("⏱️ Duration", f"{elapsed:.1f}s")
            with m3:
                total_cost = cost.get("total_cost_usd", 0)
                st.metric("💰 Est. Cost", f"${total_cost:.6f}")
            with m4:
                st.metric("🔢 LLM Calls", cost.get("total_calls", 0))

            # ── Cost breakdown ───────────────────────────────
            if cost.get("by_node"):
                st.markdown("### 💰 Cost Breakdown by Node")
                cost_data = []
                for node, data in cost["by_node"].items():
                    cost_data.append(
                        {
                            "Node": node.replace("_", " ").title(),
                            "Calls": data["calls"],
                            "Input Tokens": data["input_tokens"],
                            "Output Tokens": data["output_tokens"],
                            "Cost (USD)": f"${data['cost_usd']:.6f}",
                            "Latency (s)": f"{data['latency_s']:.2f}",
                        }
                    )
                st.table(cost_data)

            # ── Entities (Structured Dataframe) ──────────────
            if entities:
                st.markdown("### 📦 Extracted Entities (Data Table)")

                # Format data into a list of dicts for the dataframe
                table_data = []
                for ent in entities:
                    row = {
                        "Priority": round(ent.priority_score, 2),
                        "Entity Name": ent.name,
                        "Source URL": ent.source_url,
                        "Description": ent.description,
                    }
                    if isinstance(ent.metrics, dict):
                        for k, v in ent.metrics.items():
                            row[k] = str(v)
                    table_data.append(row)

                # Dynamic column config
                col_config = {
                    "Priority": st.column_config.NumberColumn(
                        "Priority", format="%.2f"
                    ),
                    "Source URL": st.column_config.LinkColumn("Source URL"),
                    "Description": st.column_config.TextColumn(
                        "Description", width="large"
                    ),
                }

                # Build Pandas DataFrame to handle dynamic metrics columns safely
                df = pd.DataFrame(table_data)

                st.dataframe(
                    df,
                    width="stretch",
                    column_config=col_config,
                    hide_index=True,
                )

                # Allow user to download the structured data
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Data as CSV",
                    data=csv,
                    file_name=f"crawled_entities_{int(time.time())}.csv",
                    mime="text/csv",
                )

            else:
                st.warning(
                    "No entities were found. Try a broader query or lower the credibility threshold."
                )

        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            import traceback

            st.code(traceback.format_exc())

    elif run_btn and not query:
        st.warning("Please enter a research query first.")


with tab2:
    st.markdown("### 🕰️ Past Research Sessions")

    if st.button("🔄 Refresh History"):
        fetch_history.clear()

    history = fetch_history()

    if not history:
        st.info("No past sessions found in the database. Run a search first!")
    else:
        # Show recent sessions
        df_history = pd.DataFrame(history)
        st.dataframe(
            df_history,
            width="stretch",
            hide_index=True,
        )

        st.markdown("### 📂 View Session Results")
        selected_session = st.selectbox(
            "Select a session to view its extracted entities:",
            options=[h["Session ID"] for h in history if h["Entities Found"] > 0],
            format_func=lambda x: f"{next(h['Date'] for h in history if h['Session ID'] == x)} — {next(h['Query'] for h in history if h['Session ID'] == x)[:50]}...",
        )

        if selected_session:
            entities_data = fetch_session_docs(selected_session)
            if entities_data:
                st.markdown(f"**Results for Session:** `{selected_session}`")

                col_config = {
                    "Priority": st.column_config.NumberColumn(
                        "Priority", format="%.2f"
                    ),
                    "Source URL": st.column_config.LinkColumn("Source URL"),
                    "Description": st.column_config.TextColumn(
                        "Description", width="large"
                    ),
                }

                df_hist = pd.DataFrame(entities_data)
                st.dataframe(
                    df_hist,
                    width="stretch",
                    column_config=col_config,
                    hide_index=True,
                )

                csv_hist = df_hist.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Session Data as CSV",
                    data=csv_hist,
                    file_name=f"session_{selected_session}.csv",
                    mime="text/csv",
                    key="download_hist",
                )

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #4a5568; font-size: 0.85rem;">'
    "LangGraph Crawler Pipeline v0.1.0 • Replicate + Tavily + MongoDB"
    "</p>",
    unsafe_allow_html=True,
)

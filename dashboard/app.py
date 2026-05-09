"""Streamlit dashboard for the Netflix-style recommender.
Hits the FastAPI service for recommendations + comparisons.
"""
import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Netflix-Style Recommender", layout="wide", page_icon="🎬")

st.title("🎬 Netflix-Style Recommendation Engine")
st.caption("SVD · Content-Based (TF-IDF) · Neural CF (PyTorch) · Hybrid — trained on MovieLens 1M")


@st.cache_data(ttl=60)
def health() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "down", "error": str(e)}


@st.cache_data(ttl=300)
def search_movies(q: str, limit: int = 20) -> list[dict]:
    try:
        r = requests.get(f"{API_BASE_URL}/movies", params={"q": q or "", "limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json().get("movies", [])
    except Exception as e:
        st.warning(f"Movie search failed: {e}")
        return []


def call_recommend(payload: dict) -> dict:
    r = requests.post(f"{API_BASE_URL}/recommend", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def call_compare(payload: dict) -> dict:
    r = requests.post(f"{API_BASE_URL}/compare", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


with st.sidebar:
    st.header("⚙️ Settings")
    st.text_input("API base URL", value=API_BASE_URL, key="api_base")
    if st.session_state.get("api_base") and st.session_state.api_base != API_BASE_URL:
        API_BASE_URL = st.session_state.api_base

    h = health()
    if h.get("status") == "ok":
        st.success(f"API healthy · {h['n_users']:,} users · {h['n_items']:,} items")
        meta = h.get("metadata", {}) or {}
        if meta.get("metrics"):
            with st.expander("Trained model metrics", expanded=False):
                rows = []
                for m, vals in meta["metrics"].items():
                    if isinstance(vals, dict):
                        rows.append({"model": m, **vals})
                if rows:
                    st.dataframe(pd.DataFrame(rows).set_index("model"), use_container_width=True)
                if "svd_test_rmse" in meta:
                    st.metric("SVD test RMSE", f"{meta['svd_test_rmse']:.4f}")
    else:
        st.error(f"API unreachable: {h.get('error', 'unknown')}")

    st.divider()
    mode = st.radio("Mode", ["Single model", "Compare all"], index=1)
    model_choice = st.selectbox(
        "Model",
        ["hybrid", "svd", "ncf", "content", "auto"],
        index=0,
        disabled=(mode == "Compare all"),
    )
    k = st.slider("Top-K", 5, 25, 10)

    st.divider()
    st.subheader("👤 User")
    user_id_str = st.text_input("MovieLens user_id (1–6040)", value="1", help="Leave empty to simulate cold-start")
    user_id = int(user_id_str) if user_id_str.strip().isdigit() else None

    st.divider()
    st.subheader("🎞️ Optional history (cold-start)")
    q = st.text_input("Search movies to add as history")
    history_ids = st.session_state.setdefault("history_ids", [])
    if q:
        results = search_movies(q, limit=10)
        for m in results:
            label = f"{m['title']}"
            if st.button(f"➕ {label}", key=f"add_{m['movie_id']}"):
                if m["movie_id"] not in history_ids:
                    history_ids.append(m["movie_id"])
    if history_ids:
        st.write("Selected history:")
        for mid in list(history_ids):
            cols = st.columns([4, 1])
            cols[0].text(f"movie_id={mid}")
            if cols[1].button("✖", key=f"rm_{mid}"):
                history_ids.remove(mid)
        if st.button("Clear history"):
            st.session_state["history_ids"] = []

    submit = st.button("🚀 Recommend", type="primary", use_container_width=True)


def render_recs(title: str, payload: dict):
    st.subheader(title)
    cold = payload.get("cold_start")
    used = payload.get("model_used")
    badge_cols = st.columns([1, 1, 6])
    badge_cols[0].caption(f"Model: **{used}**")
    if cold:
        badge_cols[1].caption("❄️ cold-start")
    recs = payload.get("recommendations", [])
    if not recs:
        st.info("No recommendations.")
        return
    df = pd.DataFrame(recs)
    cols_to_show = [c for c in ["rank", "title", "movie_id", "score"] if c in df.columns]
    st.dataframe(df[cols_to_show], use_container_width=True, hide_index=True)


if submit:
    payload: dict[str, Any] = {
        "user_id": user_id,
        "history_movie_ids": history_ids or None,
        "k": k,
    }
    if mode == "Single model":
        payload["model"] = model_choice
        try:
            with st.spinner("Scoring..."):
                result = call_recommend(payload)
            render_recs(f"Top-{k} recommendations", result)
        except Exception as e:
            st.error(f"Recommend failed: {e}")
    else:
        try:
            with st.spinner("Scoring all models..."):
                result = call_compare(payload)
        except Exception as e:
            st.error(f"Compare failed: {e}")
            result = None
        if result:
            results = result["results"]
            tab_labels = list(results.keys())
            tabs = st.tabs([t.upper() for t in tab_labels])
            for tab, name in zip(tabs, tab_labels):
                with tab:
                    payload_out = results[name]
                    if "error" in payload_out:
                        st.error(payload_out["error"])
                    else:
                        render_recs(f"{name} top-{k}", payload_out)

            # Overlap chart
            st.divider()
            st.subheader("Overlap between model outputs")
            sets = {
                m: set(r["movie_id"] for r in payload_out.get("recommendations", []))
                for m, payload_out in results.items()
                if "recommendations" in payload_out
            }
            names = list(sets.keys())
            if len(names) >= 2:
                z = []
                for a in names:
                    row = []
                    for b in names:
                        inter = len(sets[a] & sets[b])
                        row.append(inter)
                    z.append(row)
                fig = go.Figure(data=go.Heatmap(z=z, x=names, y=names, colorscale="Blues", text=z, texttemplate="%{text}"))
                fig.update_layout(height=350, title=f"# overlapping recs (out of {k})")
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Pick a user, optionally add history, then click **Recommend** in the sidebar.")

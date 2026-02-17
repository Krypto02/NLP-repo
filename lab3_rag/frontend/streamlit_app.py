"""Streamlit frontend for the RAG Document Q&A System (Bonus +5%)."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://flask_app:5000")

st.set_page_config(page_title="RAG Document Q&A", layout="wide")
st.title("RAG Document Q&A System")

with st.sidebar:
    st.header("Document Management")

    uploaded_file = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])
    if uploaded_file is not None and st.button("Upload & Index"):
        with st.spinner("Parsing, chunking, and indexing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                r = requests.post(f"{API_URL}/documents", files=files, timeout=120)
                if r.status_code == 201:
                    info = r.json()
                    st.success(
                        f"Uploaded **{info['filename']}** -- {info['num_chunks']} chunks indexed."
                    )
                else:
                    st.error(r.json().get("error", "Upload failed."))
            except requests.exceptions.RequestException as exc:
                st.error(f"API error: {exc}")

    st.divider()
    st.subheader("Indexed Documents")
    try:
        r = requests.get(f"{API_URL}/documents", timeout=10)
        docs = r.json().get("documents", [])
        if not docs:
            st.info("No documents uploaded yet.")
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{doc['filename']}**  \n{doc['num_chunks']} chunks -- {doc['upload_date'][:10]}")
            if col2.button("Delete", key=doc["id"]):
                requests.delete(f"{API_URL}/documents/{doc['id']}", timeout=10)
                st.rerun()
    except requests.exceptions.RequestException:
        st.warning("Cannot connect to the API.")

    st.divider()
    st.subheader("System Health")
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        health = r.json()
        for svc, status in health.get("services", {}).items():
            icon = "+" if status == "ok" else "x"
            st.text(f"[{icon}] {svc}: {status}")
    except requests.exceptions.RequestException:
        st.text("[x] API unreachable")

st.header("Ask a Question")

col_mode, col_method = st.columns(2)
mode = col_mode.selectbox("Pipeline mode", ["default", "agent (ReAct)"])
method = col_method.selectbox("Retrieval method", ["hybrid", "dense", "bm25"])

question = st.text_input("Your question:")

if st.button("Ask") and question:
    mode_val = "agent" if "agent" in mode else "default"
    with st.spinner("Searching and generating answer..."):
        try:
            r = requests.post(
                f"{API_URL}/query",
                json={"question": question, "mode": mode_val, "method": method},
                timeout=180,
            )
            if r.status_code == 200:
                data = r.json()
                st.subheader("Answer")
                st.write(data["answer"])
                if "agent_iterations" in data:
                    st.caption(f"Agent completed in {data['agent_iterations']} iteration(s).")

                st.subheader("Source Passages")
                for i, src in enumerate(data.get("sources", []), 1):
                    score_str = f"score: {src['score']:.4f}" if src.get("score") else ""
                    with st.expander(f"[{i}] {src['filename']} ({score_str})"):
                        st.write(src["text"])
            else:
                st.error(r.json().get("error", "Query failed."))
        except requests.exceptions.RequestException as exc:
            st.error(f"API error: {exc}")

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    answer_question,
    build_chunk_displays,
    build_citations,
    build_llm,
    build_vectorstore,
    contextualize_question,
    extract_documents,
    filter_documents_by_source,
    get_embeddings,
    retrieve_context,
    split_documents,
    summarize_pdf,
)

load_dotenv()

APP_DIR = Path(__file__).parent
CHROMA_DIR = APP_DIR / ".chroma"
CHROMA_DIR.mkdir(exist_ok=True)


def initialize_session_state() -> None:
    st.session_state.setdefault("vectorstore", None)
    st.session_state.setdefault("collection_name", None)
    st.session_state.setdefault("chunked_documents", [])
    st.session_state.setdefault("file_names", [])
    st.session_state.setdefault("file_signature", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("summary_history", [])
    st.session_state.setdefault("ingestion_warnings", [])


def build_upload_signature(uploaded_files: list) -> str:
    file_keys = sorted(f"{item.name}:{item.size}" for item in uploaded_files)
    return "|".join(file_keys)


def ingest_uploaded_documents(uploaded_files: list) -> None:
    # Store every supported file in one shared vector store so questions can span documents.
    temp_dir = Path(tempfile.mkdtemp())
    all_documents = []
    warnings = []

    try:
        for uploaded_file in uploaded_files:
            temp_path = temp_dir / uploaded_file.name
            temp_path.write_bytes(uploaded_file.getvalue())

            try:
                extracted_documents = extract_documents(temp_path)
            except ValueError as error:
                warnings.append(f"{uploaded_file.name}: {error}")
                continue
            except Exception as error:
                warnings.append(f"{uploaded_file.name}: failed to read file ({error})")
                continue

            if not extracted_documents:
                warnings.append(f"{uploaded_file.name}: no readable text found")
                continue

            all_documents.extend(extracted_documents)

        chunked_documents = split_documents(all_documents)
        if not chunked_documents:
            st.session_state.ingestion_warnings = warnings
            raise ValueError("No readable content found in the uploaded files.")

        embeddings = get_embeddings()
        vectorstore, collection_name = build_vectorstore(
            documents=chunked_documents,
            embeddings=embeddings,
            persist_root=CHROMA_DIR,
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.collection_name = collection_name
        st.session_state.chunked_documents = chunked_documents
        st.session_state.file_names = [item.name for item in uploaded_files]
        st.session_state.file_signature = build_upload_signature(uploaded_files)
        st.session_state.chat_history = []
        st.session_state.summary_history = []
        st.session_state.ingestion_warnings = warnings
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def render_uploaded_sources() -> None:
    with st.expander("Indexed Sources", expanded=False):
        for file_name in st.session_state.file_names:
            st.markdown(f"- `{file_name}`")


def render_summary_history() -> None:
    for summary in st.session_state.summary_history:
        with st.chat_message("assistant"):
            st.markdown(summary["content"])
            st.caption(summary["label"])


def render_chunk_displays(chunk_displays: list[dict]) -> None:
    for index, chunk in enumerate(chunk_displays):
        if index == 0:
            st.markdown("### Most Relevant Source")
            with st.container(border=True):
                st.markdown(
                    f"**{chunk['source']}**  \n{chunk['location']}  \nRelevance: {chunk['relevance_score']:.2f}"
                )
                st.markdown(chunk["highlighted_excerpt"])
                if chunk["supporting_sentences"]:
                    st.caption("Supporting sentences")
                    for sentence in chunk["supporting_sentences"]:
                        st.markdown(f"- {sentence}")
            if len(chunk_displays) > 1:
                st.markdown("### Other Relevant Sources")
            continue

        st.markdown(
            f"**{chunk['source']}** | {chunk['location']} | Relevance: {chunk['relevance_score']:.2f}"
        )
        st.markdown(chunk["highlighted_excerpt"])
        if chunk["supporting_sentences"]:
            for sentence in chunk["supporting_sentences"]:
                st.markdown(f"- {sentence}")
        st.divider()


def render_chat_history() -> None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] != "assistant":
                continue

            metrics = message.get("metrics")
            citations = message.get("citations") or []
            chunk_displays = message.get("chunk_displays") or []

            if metrics:
                with st.expander("Answer Confidence", expanded=False):
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    metric_col1.metric(
                        "Retrieval Confidence",
                        f"{metrics['confidence_score']:.2f}",
                    )
                    metric_col2.metric("Top-k", metrics["top_k_used"])
                    metric_col3.metric(
                        "Retrieval Precision",
                        f"{metrics['precision_at_k']:.2f}",
                    )
                    metric_col4.metric(
                        "Answer Status",
                        metrics["answer_status"],
                    )
                    st.caption(f"Average relevance score: {metrics['average_relevance']:.2f}")
                    st.caption("Retrieval Precision is estimated from thresholded relevance scores.")
                    st.caption(
                        f"Retrieval Quality: {metrics['quality_label']} (based on relevance + precision)"
                    )

            if citations:
                with st.expander("Citations", expanded=False):
                    for citation in citations:
                        st.markdown(f"- {citation}")

            if chunk_displays:
                with st.expander("Relevant Chunks", expanded=False):
                    render_chunk_displays(chunk_displays)


def add_summary_message(summary: str) -> None:
    st.session_state.summary_history.append(
        {
            "content": summary,
            "label": f"Summary across {len(st.session_state.file_names)} uploaded document(s)",
        }
    )


def add_named_summary_message(summary: str, target_label: str) -> None:
    st.session_state.summary_history.append(
        {
            "content": summary,
            "label": f"Summary target: {target_label}",
        }
    )


def add_assistant_message(
    answer: str,
    metrics,
    citations: list[str],
    chunk_displays: list,
    standalone_question: str,
) -> None:
    # Persist all explainability data with each answer so the chat can be replayed.
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "chunk_displays": [
                {
                    "source": item.source,
                    "location": item.location,
                    "relevance_score": item.relevance_score,
                    "highlighted_excerpt": item.highlighted_excerpt,
                    "supporting_sentences": item.supporting_sentences,
                }
                for item in chunk_displays
            ],
            "metrics": {
                "top_k_used": metrics.top_k_used,
                "precision_at_k": metrics.precision_at_k,
                "quality_label": metrics.quality_label,
                "average_relevance": metrics.average_relevance,
                "confidence_score": metrics.confidence_score,
                "answer_status": (
                    "No Answer Found ❌"
                    if "I could not find enough support" in answer
                    else "Answer Found ✅"
                ),
            },
            "standalone_question": standalone_question,
        }
    )


st.set_page_config(page_title="Document RAG Chatbot", page_icon=":page_facing_up:", layout="wide")
initialize_session_state()

st.title("Multi-Format RAG Chatbot")
st.caption(
    "Upload PDFs, DOCX, Excel, TXT, or CSV files, chat across them, and inspect citations and supporting chunks."
)

with st.sidebar:
    st.subheader("Retrieval Settings")
    top_k = st.slider("Top-k relevance", min_value=2, max_value=10, value=5)
    min_relevance = st.slider(
        "Minimum relevance threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
    )
    st.info(
        "Higher thresholds improve retrieval precision and reduce hallucination risk, "
        "but may refuse more answers."
    )

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "xlsx", "txt", "csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    file_signature = build_upload_signature(uploaded_files)

    if st.session_state.file_signature != file_signature:
        with st.spinner("Parsing documents, chunking content, and creating embeddings..."):
            try:
                ingest_uploaded_documents(uploaded_files)
                st.success(f"Indexed {len(st.session_state.file_names)} document(s) successfully.")
            except ValueError as error:
                st.error(str(error))

        for warning in st.session_state.ingestion_warnings:
            st.warning(warning)

if st.session_state.vectorstore is not None:
    st.write(f"Active knowledge base: **{len(st.session_state.file_names)} document(s)**")
    render_uploaded_sources()

    summary_options = ["All Documents", *st.session_state.file_names]
    selected_summary_target = st.selectbox(
        "Choose which document to summarize",
        options=summary_options,
        index=0,
        help="Pick a single source for a focused summary, or choose All Documents for a combined overview.",
    )

    toolbar_col1, toolbar_col2 = st.columns([1, 1])

    with toolbar_col1:
        if st.button("Summarize Selected Source", use_container_width=True):
            with st.spinner("Generating grounded summary..."):
                llm = build_llm()
                selected_documents = filter_documents_by_source(
                    documents=st.session_state.chunked_documents,
                    selected_source=selected_summary_target,
                )
                summary = summarize_pdf(
                    llm=llm,
                    chunked_documents=selected_documents,
                )
            add_named_summary_message(summary, selected_summary_target)

    with toolbar_col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.summary_history = []
            st.rerun()

    render_summary_history()
    render_chat_history()

    user_prompt = st.chat_input("Ask a question across the uploaded documents")

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks and generating grounded answer..."):
                llm = build_llm()
                prior_turns = st.session_state.chat_history[:-1]
                standalone_question = contextualize_question(
                    llm=llm,
                    question=user_prompt,
                    chat_history=prior_turns,
                )
                retrieved_chunks, metrics = retrieve_context(
                    vectorstore=st.session_state.vectorstore,
                    question=standalone_question,
                    top_k=top_k,
                    min_relevance=min_relevance,
                )
                retrieved_docs = [item.document for item in retrieved_chunks]
                answer = answer_question(
                    llm=llm,
                    question=standalone_question,
                    retrieved_docs=retrieved_docs,
                    chat_history=prior_turns,
                )
                citations = build_citations(retrieved_docs[: min(3, len(retrieved_docs))])
                chunk_displays = build_chunk_displays(retrieved_chunks, standalone_question)

            st.markdown(answer)

            # Confidence is derived from retrieval behavior so users can judge answer reliability.
            with st.expander("Answer Confidence", expanded=True):
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                answer_status = (
                    "No Answer Found ❌"
                    if "I could not find enough support" in answer
                    else "Answer Found ✅"
                )
                metric_col1.metric("Retrieval Confidence", f"{metrics.confidence_score:.2f}")
                metric_col2.metric("Top-k", metrics.top_k_used)
                metric_col3.metric("Retrieval Precision", f"{metrics.precision_at_k:.2f}")
                metric_col4.metric("Answer Status", answer_status)
                st.caption(f"Average relevance score: {metrics.average_relevance:.2f}")
                st.caption("Retrieval Precision is estimated from thresholded relevance scores.")
                st.caption(
                    f"Retrieval Quality: {metrics.quality_label} (based on relevance + precision)"
                )
                if standalone_question != user_prompt:
                    st.caption(f"Resolved question: {standalone_question}")

            if citations:
                with st.expander("Citations", expanded=False):
                    for citation in citations:
                        st.markdown(f"- {citation}")

            if chunk_displays:
                # Retrieved chunks are shown with highlighted evidence for explainability.
                with st.expander("Relevant Chunks", expanded=True):
                    render_chunk_displays(
                        [
                                {
                                    "source": item.source,
                                    "location": item.location,
                                    "relevance_score": item.relevance_score,
                                    "highlighted_excerpt": item.highlighted_excerpt,
                                    "supporting_sentences": item.supporting_sentences,
                            }
                            for item in chunk_displays
                        ]
                    )

        add_assistant_message(
            answer=answer,
            metrics=metrics,
            citations=citations,
            chunk_displays=chunk_displays,
            standalone_question=standalone_question,
        )
else:
    st.info("Upload one or more supported files to start the multi-document chat workflow.")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("Set OPENAI_API_KEY in your .env file before using the app.")

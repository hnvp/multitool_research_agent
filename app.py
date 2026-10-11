from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import streamlit as st

from agent import run_agent
from ingest_rag import clean_source_name, ingest_pdf

st.set_page_config(
    page_title="Multitool Research & Knowledge Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multitool Research & Knowledge Agent")

# session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_source" not in st.session_state:
    st.session_state.active_source = None

if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# pdf upload
uploaded_file = st.file_uploader(
    "Upload a PDF to ask questions about it",
    type=["pdf"],
)

if uploaded_file is None:
    st.session_state.active_source = None
    st.session_state.uploaded_file_id = None
    st.session_state.uploaded_file_name = None

else:
    file_bytes = uploaded_file.getvalue()
    file_id = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.uploaded_file_id != file_id:
        with st.spinner("Indexing document..."):
            temp_path = None

            try:
                original_name = Path(uploaded_file.name).name
                clean_name = clean_source_name(original_name)

                temp_dir = (
                    Path(tempfile.gettempdir()) / "multitool_research_agent_uploads"
                )
                temp_dir.mkdir(parents=True, exist_ok=True)

                source_filename = (
                    f"{Path(clean_name).stem}_{file_id[:12]}"
                    f"{Path(clean_name).suffix}"
                )

                temp_path = temp_dir / source_filename
                temp_path.write_bytes(file_bytes)

                source_name = ingest_pdf(temp_path)

                st.session_state.active_source = source_name
                st.session_state.uploaded_file_id = file_id
                st.session_state.uploaded_file_name = original_name

                # New document = new conversation context.
                st.session_state.messages = []

                st.success(f"Document ready: {original_name}")

            except Exception:
                st.error("Could not process the uploaded document.")

            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)


if st.session_state.uploaded_file_name:
    st.caption(f"Active document: {st.session_state.uploaded_file_name}")

# chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat
query = st.chat_input("Ask the agent something...")

if query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        trace = None

        with st.spinner("Agent is working..."):
            try:
                final_text, trace, _ = run_agent(
                    st.session_state.messages[-12:],
                    source=st.session_state.active_source,
                )
            except Exception:
                final_text = (
                    "I could not complete the request "
                    "because the agent encountered an error."
                )

        st.markdown(final_text)

        if trace:
            with st.expander(f"Execution Trace ({len(trace.events)} tool calls)"):
                st.caption(f"Total latency: " f"{trace.total_latency_ms():.2f} ms")

                for i, event in enumerate(
                    trace.events,
                    start=1,
                ):
                    st.markdown(
                        f"**{i}. `{event.tool}` — "
                        f"{event.status} — "
                        f"{event.latency_ms:.0f} ms**"
                    )

                    st.code(
                        json.dumps(
                            event.args,
                            ensure_ascii=False,
                            indent=2,
                        ),
                        language="json",
                    )

                    if event.error:
                        st.error("Tool execution failed.")
                    elif event.output_preview:
                        st.text(event.output_preview)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_text,
        }
    )

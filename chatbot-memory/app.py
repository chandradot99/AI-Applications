import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag-advanced'))

import streamlit as st
from chat import chat
from memory import clear_session

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Policy Assistant",
    page_icon="📋",
    layout="centered"
)

st.title("📋 Insurance Policy Assistant")
st.caption("Ask questions about your insurance policy documents.")

# ── Session management ────────────────────────────────────────────────────────
# Each browser tab gets a unique session ID stored in Streamlit session state
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_" + os.urandom(4).hex()

if "messages" not in st.session_state:
    st.session_state.messages = []  # stores messages for UI display

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session")
    st.write(f"ID: `{st.session_state.session_id}`")

    if st.button("🗑️ Clear conversation"):
        clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

# ── Chat history display ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask about your policies..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            response = chat(question, session_id=st.session_state.session_id)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

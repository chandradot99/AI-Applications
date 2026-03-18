from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# In-memory store: session_id -> ChatMessageHistory
# Each user/session gets their own separate conversation history
_session_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # Create a new history for this session if it doesn't exist
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def clear_session(session_id: str):
    # Reset a session's memory — like clicking "New Chat"
    if session_id in _session_store:
        del _session_store[session_id]


def list_sessions() -> list[str]:
    return list(_session_store.keys())

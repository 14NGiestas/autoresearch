"""ai_chat_hoarder: discover and archive publicly-shared AI chat conversations.

Platforms supported: Claude, ChatGPT, Grok, Qwen, Deepseek, Gemini.
All HTTP goes through `curl` (the sandbox blocks python `requests` to some
hosts); `--compressed` handles brotli/gzip.
"""
from .config import load_dotenv

load_dotenv()

from .core import Store, HttpClient, fetch_all, reaudit, discover_ddg, crawl_github, ingest_file
from .session_convert import (
    find_opencode_db,
    list_opencode_sessions,
    find_latest_opencode_session,
    read_opencode_session,
    convert_opencode_session,
    write_pi_session,
    convert_latest_opencode_session,
)

__all__ = [
    "Store",
    "HttpClient",
    "fetch_all",
    "reaudit",
    "discover_ddg",
    "crawl_github",
    "ingest_file",
    "find_opencode_db",
    "list_opencode_sessions",
    "find_latest_opencode_session",
    "read_opencode_session",
    "convert_opencode_session",
    "write_pi_session",
    "convert_latest_opencode_session",
    "load_dotenv",
]

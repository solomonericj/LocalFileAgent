import io
import json
import socket
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from LocalfileAgent import stream_ollama_chat


def _fake_stream_response(*tokens: str):
    """Return a file-like object that yields NDJSON lines like Ollama does."""
    lines = []
    for i, token in enumerate(tokens):
        done = i == len(tokens) - 1
        line = json.dumps({"message": {"content": token}, "done": done}).encode() + b"\n"
        lines.append(line)
    buf = io.BytesIO(b"".join(lines))
    buf.__enter__ = lambda s: s
    buf.__exit__ = MagicMock(return_value=False)
    return buf


def test_yields_all_tokens():
    resp = _fake_stream_response("Hello", " world", "!")
    with patch("urllib.request.urlopen", return_value=resp):
        result = list(stream_ollama_chat([{"role": "user", "content": "hi"}], "mistral"))
    assert result == ["Hello", " world", "!"]


def test_raises_connection_error_on_url_error():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        with pytest.raises(ConnectionError):
            list(stream_ollama_chat([{"role": "user", "content": "hi"}], "mistral"))


def test_raises_timeout_on_socket_timeout():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError(socket.timeout())):
        with pytest.raises(TimeoutError):
            list(stream_ollama_chat([{"role": "user", "content": "hi"}], "mistral"))


def test_skips_empty_lines():
    raw = (
        b"\n"
        b'{"message":{"content":"hi"},"done":false}\n'
        b"\n"
        b'{"message":{"content":"!"},"done":true}\n'
    )
    buf = io.BytesIO(raw)
    buf.__enter__ = lambda s: s
    buf.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=buf):
        result = list(stream_ollama_chat([{"role": "user", "content": "x"}], "mistral"))
    assert result == ["hi", "!"]

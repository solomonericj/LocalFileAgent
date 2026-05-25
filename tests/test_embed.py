import io
import json
import socket
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from LocalfileAgent import embed_ollama


def _fake_json_response(payload: dict):
    buf = io.BytesIO(json.dumps(payload).encode())
    buf.__enter__ = lambda s: s
    buf.__exit__ = MagicMock(return_value=False)
    return buf


def test_returns_embeddings_list():
    resp = _fake_json_response({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    with patch("urllib.request.urlopen", return_value=resp):
        vecs = embed_ollama(["a", "b"], "nomic-embed-text")
    assert vecs == [[0.1, 0.2], [0.3, 0.4]]


def test_sends_input_field():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data)
        return _fake_json_response({"embeddings": [[1.0]]})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        embed_ollama(["hello"], "nomic-embed-text")
    assert captured["body"]["input"] == ["hello"]
    assert captured["body"]["model"] == "nomic-embed-text"


def test_raises_connection_error():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        with pytest.raises(ConnectionError):
            embed_ollama(["x"], "nomic-embed-text")

import io
import json
import socket
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

import LocalfileAgent as lfa
from LocalfileAgent import embed_ollama, check_ollama_available


def _fake_raw_response(raw: bytes):
    buf = io.BytesIO(raw)
    buf.__enter__ = lambda s: s
    buf.__exit__ = MagicMock(return_value=False)
    return buf


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


def test_raises_timeout_error():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError(socket.timeout())):
        with pytest.raises(TimeoutError):
            embed_ollama(["x"], "nomic-embed-text")


def test_raises_when_no_embeddings_returned():
    resp = _fake_json_response({"error": "model not found"})
    with patch("urllib.request.urlopen", return_value=resp):
        with pytest.raises(ValueError):
            embed_ollama(["x"], "nomic-embed-text")


def _tags_response(names):
    return _fake_json_response({"models": [{"name": n} for n in names]})


def test_check_passes_when_both_models_present():
    with patch("urllib.request.urlopen", return_value=_tags_response(["mistral", "nomic-embed-text"])):
        check_ollama_available("mistral", embed_model="nomic-embed-text")  # no SystemExit


def test_check_exits_when_embed_model_missing():
    with patch("urllib.request.urlopen", return_value=_tags_response(["mistral"])):
        with pytest.raises(SystemExit):
            check_ollama_available("mistral", embed_model="nomic-embed-text")


def test_check_exits_on_malformed_tags_json():
    """A non-JSON / unexpected response must fail cleanly, not raise an
    unhandled JSONDecodeError/TypeError."""
    with patch("urllib.request.urlopen", return_value=_fake_raw_response(b"<html>nope</html>")):
        with pytest.raises(SystemExit):
            check_ollama_available("mistral")


def test_check_requires_exact_tag_when_one_is_specified():
    """Asking for a specific tag must not be satisfied by a different tag of the
    same base model."""
    with patch("urllib.request.urlopen", return_value=_tags_response(["mistral:13b"])):
        with pytest.raises(SystemExit):
            check_ollama_available("mistral:7b")


def test_check_passes_with_base_name_match():
    """Asking for a bare base name is still satisfied by any pulled tag."""
    with patch("urllib.request.urlopen", return_value=_tags_response(["mistral:latest"])):
        check_ollama_available("mistral")   # no SystemExit


def test_post_uses_request_timeout(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["timeout"] = timeout
        return _fake_json_response({"response": "ok"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    lfa._post(lfa.OLLAMA_GENERATE, {"model": "m"})
    assert captured["timeout"] == lfa.REQUEST_TIMEOUT

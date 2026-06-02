import io

import LocalfileAgent as lfa


def test_force_utf8_lets_emoji_through_on_cp1252_stream():
    """Reconfiguring to UTF-8 lets the decorative glyphs we print survive on a
    console whose locale encoding (e.g. cp1252 on Windows) cannot encode them."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", newline="")

    lfa._force_utf8_output([stream])

    stream.write("📂 ok")   # would raise UnicodeEncodeError under cp1252
    stream.flush()
    assert "📂 ok".encode("utf-8") in raw.getvalue()


def test_force_utf8_ignores_streams_without_reconfigure():
    lfa._force_utf8_output([object()])   # must not raise

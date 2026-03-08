"""Unit tests for web fetch tools."""

import pytest


def test_web_search():
    """Test web_search returns results."""
    from axiom_os.agent.tools import web_search
    r = web_search(query="python physics", max_results=2)
    assert r["ok"] is True
    assert "results" in r
    assert len(r["results"]) <= 2


def test_fetch_url_wikipedia():
    """Test fetch_url with Wikipedia."""
    from axiom_os.agent.tools import fetch_url
    r = fetch_url(url="https://en.wikipedia.org/wiki/Physics", max_chars=5000)
    assert r["ok"] is True
    assert "content" in r
    assert len(r["content"]) > 100


def test_fetch_url_invalid():
    """Test fetch_url with invalid URL."""
    from axiom_os.agent.tools import fetch_url
    r = fetch_url(url="")
    assert r["ok"] is False
    assert "error" in r

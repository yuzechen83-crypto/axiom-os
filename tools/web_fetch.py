"""
Axiom-OS 网页抓取与搜索工具
供 Agent 自主从网上抓取学习。

- fetch_url: 抓取 URL 正文
- web_search: 搜索关键词，返回摘要
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]

# 安全：仅允许的域名（可扩展）
ALLOWED_DOMAINS = {
    "wikipedia.org", "en.wikipedia.org", "zh.wikipedia.org",
    "github.com", "raw.githubusercontent.com",
    "arxiv.org", "docs.python.org", "pypi.org",
    "nasa.gov", "nist.gov", "nature.com", "science.org",
    "*.wikipedia.org", "*.github.com", "*.arxiv.org",
}

MAX_CONTENT_LEN = 30000
FETCH_TIMEOUT = 15


def _is_allowed_url(url: str) -> bool:
    """检查 URL 是否在允许列表（可配置放宽）"""
    try:
        parsed = urlparse(url)
        domain = (parsed.netloc or "").lower()
        if not domain:
            return False
        for allowed in ALLOWED_DOMAINS:
            if allowed.startswith("*."):
                if domain.endswith(allowed[1:]) or domain == allowed[2:]:
                    return True
            elif domain == allowed or domain.endswith("." + allowed):
                return True
        # 放宽：允许 https 且非 localhost/内网
        if parsed.scheme in ("http", "https"):
            if "localhost" in domain or "127.0.0.1" in domain or domain.startswith("192.168.") or domain.startswith("10."):
                return False
            return True
        return False
    except Exception:
        return False


def _extract_text(html: str, url: str) -> str:
    """从 HTML 提取正文"""
    # 优先 trafilatura（更准确）
    try:
        import trafilatura
        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True)
        if text and len(text) > 100:
            return text[:MAX_CONTENT_LEN]
    except ImportError:
        pass

    # 备选：BeautifulSoup
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:MAX_CONTENT_LEN] if text else ""
    except ImportError:
        pass

    # 最后：简单 strip tags
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CONTENT_LEN] if text else ""


def fetch_url(
    url: str,
    save_to_workspace: Optional[str] = None,
    max_chars: int = MAX_CONTENT_LEN,
) -> Dict[str, Any]:
    """
    抓取 URL 正文，供 Agent 学习。
    save_to_workspace: 若指定（如 ENGINE_DESIGN.md），则追加到 workspace 文件。
    """
    url = (url or "").strip()
    if not url:
        return {"ok": False, "error": "URL 为空"}
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    if not _is_allowed_url(url):
        return {"ok": False, "error": f"URL 不在允许列表: {url[:50]}..."}

    try:
        import requests
        headers = {"User-Agent": "AxiomOS/1.0 (Physics-Aware AI; +https://github.com/yuzechen83-crypto/axiom-os)"}
        r = requests.get(url, headers=headers, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        html = r.text
        content_type = r.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return {"ok": False, "error": f"非 HTML 内容: {content_type[:50]}"}

        text = _extract_text(html, url)
        if not text or len(text) < 50:
            return {"ok": False, "error": "无法提取有效正文"}

        text = text[:max_chars]
        result = {"ok": True, "content": text, "url": url, "length": len(text)}

        if save_to_workspace:
            try:
                from axiom_os.config.loader import get_workspace_path
                ws = get_workspace_path()
                ws.mkdir(parents=True, exist_ok=True)
                path = ws / save_to_workspace
                if path.suffix != ".md":
                    path = path.with_suffix(".md")
                # 追加或新建
                header = f"\n\n---\n## 来自 {url}\n\n"
                if path.exists():
                    path.write_text(path.read_text(encoding="utf-8") + header + text[:15000], encoding="utf-8")
                else:
                    path.write_text(f"# {save_to_workspace}\n{header}{text[:15000]}", encoding="utf-8")
                result["saved_to"] = str(path)
            except Exception as e:
                result["save_error"] = str(e)

        return result
    except Exception as e:
        return {"ok": False, "error": str(e), "url": url}


def web_search(
    query: str,
    max_results: int = 5,
) -> Dict[str, Any]:
    """
    网页搜索，返回摘要列表。
    用于 Agent 发现学习资源。
    """
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error": "搜索关键词为空"}

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        items = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")[:500]}
            for r in results[:max_results]
        ]
        return {"ok": True, "query": query, "results": items}
    except ImportError:
        return {
            "ok": False,
            "error": "请安装 duckduckgo-search: pip install duckduckgo-search",
            "query": query,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "query": query}

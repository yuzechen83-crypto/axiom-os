"""
Axiom-OS REST API - 服务化接口
供外部系统通过 HTTP 调用 Axiom 能力：Grid Pulse MPC、基准、RAR、Discovery 等。

Run: axiom-api
     uvicorn axiom_os.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 加载 LLM 配置
_env_path = Path(__file__).resolve().parents[1] / "config" / "axiom_os_llm.env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

app = FastAPI(
    title="Axiom-OS API",
    description="Physics-Aware Neural Network Framework - REST API",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response models ---

class GridMPCRequest(BaseModel):
    disturbance_MW: float = 800.0
    nadir_safe_Hz: float = 49.0


class GridMPCResponse(BaseModel):
    ok: bool
    nadir_no_ctrl: float | None = None
    nadir_after: float | None = None
    load_shed_MW: float | None = None
    storage_MW: float | None = None
    feasible: bool | None = None
    error: str | None = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    ok: bool
    reply: str | None = None
    error: str | None = None


# --- Endpoints ---

@app.get("/")
def root():
    return {"service": "Axiom-OS API", "version": "0.1.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/grid-mpc", response_model=GridMPCResponse)
def run_grid_mpc(req: GridMPCRequest):
    """运行 Grid Pulse MPC：预测-决策-执行闭环"""
    try:
        from axiom_os.experiments.run_grid_mpc import main as grid_main
        result = grid_main(
            disturbance_MW=req.disturbance_MW,
            nadir_safe_Hz=req.nadir_safe_Hz,
        )
        return GridMPCResponse(
            ok=True,
            nadir_no_ctrl=result.get("nadir_no_ctrl"),
            nadir_after=result.get("nadir_after"),
            load_shed_MW=result.get("load_shed"),
            storage_MW=result.get("storage"),
            feasible=result.get("nadir_after", 0) >= req.nadir_safe_Hz,
        )
    except Exception as e:
        return GridMPCResponse(ok=False, error=str(e))


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """自然语言对话（DeepSeek 扩展）"""
    try:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            return ChatResponse(ok=False, error="DEEPSEEK_API_KEY 未配置")
        from axiom_os.agent.deepseek_agent import run_agent_loop
        reply = run_agent_loop(api_key=api_key, user_message=req.message, max_tool_rounds=5)
        return ChatResponse(ok=True, reply=reply or "（无回复）")
    except Exception as e:
        return ChatResponse(ok=False, error=str(e))


@app.get("/api/v1/benchmark/report")
def get_benchmark_report():
    """读取最新基准报告"""
    try:
        from axiom_os.agent.tools import get_benchmark_report as _get
        r = _get()
        if not r.get("ok"):
            raise HTTPException(status_code=404, detail=r.get("error", "报告不存在"))
        return r
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/benchmark/run")
def run_benchmark():
    """运行快速基准"""
    try:
        from axiom_os.agent.tools import run_benchmark_quick
        return run_benchmark_quick()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rar")
def run_rar(n_galaxies: int = 20, epochs: int = 200):
    """运行 RAR 星系旋转发现"""
    try:
        from axiom_os.agent.tools import run_rar
        return run_rar(n_galaxies=n_galaxies, epochs=epochs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CadModelRequest(BaseModel):
    shape: str = "l_bracket"
    output_path: str | None = None


@app.post("/api/v1/cad-model")
def run_cad_model(req: CadModelRequest):
    """CAD 建模：构建 3D 形状并导出 STL"""
    try:
        from axiom_os.agent.tools import run_cad_model as _run
        return _run(shape=req.shape, output_path=req.output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cad-shapes")
def list_cad_shapes():
    """列出支持的 CAD 形状"""
    try:
        from axiom_os.agent.tools import list_cad_shapes
        return list_cad_shapes()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class WorkspaceDocRequest(BaseModel):
    doc_name: str = "IDENTITY.md"


@app.post("/api/v1/workspace/read")
def read_workspace_doc(req: WorkspaceDocRequest):
    """读取 workspace 文档"""
    try:
        from axiom_os.agent.tools import read_workspace_doc
        return read_workspace_doc(doc_name=req.doc_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/workspace/docs")
def list_workspace_docs():
    """列出 workspace 文档"""
    try:
        from axiom_os.agent.tools import list_workspace_docs
        return list_workspace_docs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class HippocampusRequest(BaseModel):
    question: str = ""
    top_k: int = 5


@app.post("/api/v1/hippocampus/retrieve")
def retrieve_hippocampus(req: HippocampusRequest):
    """从 Hippocampus 检索物理定律"""
    try:
        from axiom_os.agent.tools import retrieve_hippocampus
        return retrieve_hippocampus(question=req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FetchUrlRequest(BaseModel):
    url: str
    save_to_workspace: str | None = None
    max_chars: int = 30000


@app.post("/api/v1/fetch-url")
def api_fetch_url(req: FetchUrlRequest):
    """抓取 URL 正文"""
    try:
        from axiom_os.agent.tools import fetch_url
        return fetch_url(url=req.url, save_to_workspace=req.save_to_workspace, max_chars=req.max_chars)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5


@app.post("/api/v1/web-search")
def api_web_search(req: WebSearchRequest):
    """网页搜索"""
    try:
        from axiom_os.agent.tools import web_search
        return web_search(query=req.query, max_results=req.max_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Entry point for axiom-api console script."""
    try:
        import uvicorn
    except ImportError:
        print("Install: pip install 'axiom-os[full]' or pip install fastapi uvicorn")
        sys.exit(1)
    port = int(os.environ.get("AXIOM_API_PORT", "8000"))
    host = os.environ.get("AXIOM_API_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

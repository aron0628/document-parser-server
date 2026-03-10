"""PipelineRunner 프로토콜 인터페이스

노드 함수는 순수 async (state) -> state_update 함수로,
오케스트레이터(LangGraph)에 의존하지 않는다.
"""

from typing import Any, Dict, Protocol

from app.models.state import PipelineState


class PipelineRunner(Protocol):
    """파이프라인 실행 프로토콜"""

    async def run(self, state: PipelineState) -> PipelineState:
        ...


class LangGraphPipelineRunner:
    """LangGraph 기반 파이프라인 실행기"""

    def __init__(self):
        from app.pipeline.graph import build_graph
        self._graph = build_graph()

    async def run(self, state: PipelineState) -> PipelineState:
        result = await self._graph.ainvoke(state)
        return result


# 싱글턴 인스턴스
_runner: LangGraphPipelineRunner | None = None


def get_runner() -> LangGraphPipelineRunner:
    global _runner
    if _runner is None:
        _runner = LangGraphPipelineRunner()
    return _runner


async def run_pipeline(job_id: str, pdf_path: str, params: Dict[str, Any]) -> PipelineState:
    """파이프라인 실행 편의 함수"""
    initial_state: PipelineState = {
        "pdf_path": pdf_path,
        "language": params["language"],
        "include_image": params["include_image"],
        "batch_size": params["batch_size"],
        "test_page": params.get("test_page"),
        "upstage_api_key": params["upstage_api_key"],
        "openai_api_key": params["openai_api_key"],
        "job_id": job_id,
        "pdf_chunks": [],
        "current_batch_index": 0,
        "batch_parse_results": [],
    }
    runner = get_runner()
    return await runner.run(initial_state)

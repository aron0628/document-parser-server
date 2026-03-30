"""PipelineRunner 프로토콜 인터페이스

노드 함수는 순수 async (state) -> state_update 함수로,
오케스트레이터(LangGraph)에 의존하지 않는다.
"""

import logging
from typing import Any, Dict, Protocol

from app.config import settings
from app.models.state import PipelineState

logger = logging.getLogger(__name__)


class PipelineRunner(Protocol):
    """파이프라인 실행 프로토콜"""

    async def run(
        self,
        state: PipelineState,
        configurable: Dict[str, Any] | None = None,
    ) -> PipelineState:
        ...


class LangGraphPipelineRunner:
    """LangGraph 기반 파이프라인 실행기"""

    def __init__(self):
        from app.pipeline.graph import compile_graph
        from app.pipeline.checkpointer import get_checkpointer
        self._graph = compile_graph(checkpointer=get_checkpointer())

    async def run(
        self,
        state: PipelineState,
        configurable: Dict[str, Any] | None = None,
    ) -> PipelineState:
        from app.pipeline.logging import LangGraphCallbackTracker, PipelineTracker

        job_id = state.get("job_id", "unknown")
        on_phase_change = self._make_on_phase_change(job_id)
        tracker = PipelineTracker(job_id, on_phase_change=on_phase_change, configurable=configurable)
        callback = LangGraphCallbackTracker(tracker, state=state)

        try:
            batch_count = len(state.get("pdf_chunks", []))
            recursion_limit = max(100, batch_count * 3 + 50)

            config: Dict[str, Any] = {
                "callbacks": [callback],
                "recursion_limit": recursion_limit,
            }
            if configurable:
                config["configurable"] = configurable

            result = await self._graph.ainvoke(state, config=config)
            tracker.pipeline_summary()
            return result
        except Exception:
            tracker.pipeline_summary()
            raise

    @staticmethod
    def _make_on_phase_change(job_id: str):
        """Phase 전환 시 job_manager를 갱신하는 콜백을 생성한다."""
        def _on_phase_change(phase: int, progress: float, current_node: str | None = None):
            try:
                from app.services import job_manager
                job_manager.update_job(
                    job_id,
                    current_phase=phase,
                    progress=round(progress, 4),
                    current_node=current_node,
                )
            except Exception as e:
                logger.warning(f"[{job_id}] Phase 전환 갱신 실패: {e}")
        return _on_phase_change


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
        "job_id": job_id,
        "pdf_chunks": [],
        "current_batch_index": 0,
        "batch_parse_results": [],
        "embedding_model": params.get("embedding_model", "embedding-passage"),
        "chunk_size": params.get("chunk_size", 1000),
        "chunk_overlap": params.get("chunk_overlap", 200),
        "enable_raptor": params.get("enable_raptor", False),
    }
    configurable = {
        "thread_id": job_id,
        "upstage_api_key": params["upstage_api_key"],
        "openai_api_key": params["openai_api_key"],
        "google_api_key": params.get("google_api_key", ""),
        "xai_api_key": params.get("xai_api_key", ""),
        "vision_model": params.get("vision_model", settings.vision_model),
        "raptor_summarization_model": params.get("raptor_summarization_model", settings.raptor_summarization_model),
    }
    logger.info(
        f"[{job_id}] 파이프라인 시작 — "
        f"embedding_model={initial_state['embedding_model']}, "
        f"chunk_size={initial_state['chunk_size']}, chunk_overlap={initial_state['chunk_overlap']}, "
        f"enable_raptor={initial_state['enable_raptor']}"
    )
    runner = get_runner()
    return await runner.run(initial_state, configurable=configurable)

"""Pipeline logging module for LangGraph-based PDF processing pipeline.

Provides structured logging, progress tracking, and timing for pipeline nodes.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPELINE_NODES: list[tuple[str, int, str]] = [
    # Phase 1
    ("split_pdf", 1, "PDF 분할"),
    ("working_queue", 1, "작업 큐 확인"),
    ("document_parse", 1, "문서 파싱 (Upstage API)"),
    ("post_document_parse", 1, "파싱 결과 병합"),
    # Phase 2
    ("create_elements", 2, "요소 생성"),
    ("keyword_preprocess", 2, "키워드 전처리"),
    ("export_image", 2, "이미지 내보내기"),
    ("page_elements_extractor", 2, "페이지 요소 추출"),
    ("image_entity_extractor", 2, "이미지 엔티티 추출"),
    ("table_entity_extractor", 2, "테이블 엔티티 추출"),
    ("merge_entity", 2, "엔티티 병합"),
    ("reconstruct_elements", 2, "요소 재구성"),
    # Phase 3
    ("langchain_document", 3, "LangChain Document 생성"),
    ("embedding", 3, "벡터 임베딩 생성"),
    ("export_html", 3, "HTML 내보내기"),
    ("export_markdown", 3, "Markdown 내보내기"),
    ("export_table_csv", 3, "CSV 내보내기"),
]

PHASE_WEIGHTS: dict[int, float] = {1: 0.50, 2: 0.35, 3: 0.15}

NODE_META: dict[str, tuple[int, str]] = {
    name: (phase, description) for name, phase, description in PIPELINE_NODES
}

PHASE_NODE_COUNTS: dict[int, int] = {}
for _name, _phase, _desc in PIPELINE_NODES:
    PHASE_NODE_COUNTS[_phase] = PHASE_NODE_COUNTS.get(_phase, 0) + 1

TOTAL_PHASES: int = 3


# ---------------------------------------------------------------------------
# PipelineTracker
# ---------------------------------------------------------------------------


class PipelineTracker:
    """Tracks progress, timing, and phase transitions for a pipeline run.

    Args:
        job_id: Unique identifier for the job being processed.
        on_phase_change: Optional callback invoked when the current phase
            advances. Receives ``(job_id, new_phase)`` as arguments.
    """

    # 모델명을 동적으로 표시할 노드 → configurable 키 매핑
    _LLM_NODE_CONFIG_KEY = {
        "image_entity_extractor": "vision_model",
        "table_entity_extractor": "vision_model",
        "raptor": "raptor_summarization_model",
    }

    def __init__(
        self,
        job_id: str,
        on_phase_change: Callable | None = None,
        configurable: dict | None = None,
    ) -> None:
        self.job_id = job_id
        self._on_phase_change = on_phase_change
        self._configurable = configurable or {}
        self._pipeline_start: float | None = None
        self._node_timings: dict[str, float] = {}
        self._node_start_times: dict[str, float] = {}
        self._completed_nodes: set[str] = set()
        self._current_phase: int = 0
        self._batch_info: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def node_start(self, node_name: str, state: dict | None = None) -> None:
        """Record the start of a node execution.

        Args:
            node_name: Name of the pipeline node being started.
            state: Current pipeline state dict (used for batch extraction).
        """
        now = time.monotonic()
        self._node_start_times[node_name] = now

        if self._pipeline_start is None:
            self._pipeline_start = now

        phase, description = NODE_META.get(node_name, (0, node_name))

        # LLM 노드는 현재 사용 중인 모델명을 동적으로 표시
        config_key = self._LLM_NODE_CONFIG_KEY.get(node_name)
        if config_key and self._configurable:
            model_str = self._configurable.get(config_key, "")
            if model_str:
                description = f"{description} ({model_str})"

        batch_suffix = self._extract_batch_suffix(node_name, state)
        if batch_suffix:
            self._batch_info[node_name] = batch_suffix

        logger.info(
            "[%s] [Phase %d/3] %s 시작 - %s%s",
            self.job_id,
            phase,
            node_name,
            description,
            batch_suffix,
        )

    def node_end(self, node_name: str, error: Exception | None = None) -> None:
        """Record the end of a node execution.

        Args:
            node_name: Name of the pipeline node that finished.
            error: Exception raised during node execution, or ``None`` on success.
        """
        now = time.monotonic()
        start = self._node_start_times.pop(node_name, now)
        duration = now - start

        # Accumulate timing for repeated nodes (e.g. document_parse in batches)
        self._node_timings[node_name] = self._node_timings.get(node_name, 0.0) + duration

        self._completed_nodes.add(node_name)

        progress = self._calculate_progress()
        progress_pct = round(progress * 100, 1)

        phase, _ = NODE_META.get(node_name, (0, node_name))
        batch_suffix = self._batch_info.get(node_name, "")
        status = "실패" if error else "완료"

        if error:
            logger.error(
                "[%s] [Phase %d/3] %s %s - %.2fs%s (진행: %s%%)",
                self.job_id,
                phase,
                node_name,
                status,
                duration,
                batch_suffix,
                progress_pct,
            )
        else:
            logger.info(
                "[%s] [Phase %d/3] %s %s - %.2fs%s (진행: %s%%)",
                self.job_id,
                phase,
                node_name,
                status,
                duration,
                batch_suffix,
                progress_pct,
            )

        # Phase transition detection — phase only advances, never retreats
        if phase > self._current_phase:
            self._current_phase = phase
            if self._on_phase_change is not None:
                try:
                    self._on_phase_change(phase, progress, node_name)
                except Exception:
                    logger.exception(
                        "[%s] on_phase_change callback raised an exception",
                        self.job_id,
                    )

    def pipeline_summary(self) -> None:
        """Log a summary of the completed pipeline run."""
        now = time.monotonic()
        total = (now - self._pipeline_start) if self._pipeline_start is not None else 0.0

        phase_durations: dict[int, float] = {}
        for name, timing in self._node_timings.items():
            phase, _ = NODE_META.get(name, (0, name))
            phase_durations[phase] = phase_durations.get(phase, 0.0) + timing

        p1 = phase_durations.get(1, 0.0)
        p2 = phase_durations.get(2, 0.0)
        p3 = phase_durations.get(3, 0.0)

        # Top 3 slowest nodes
        sorted_nodes = sorted(self._node_timings.items(), key=lambda kv: kv[1], reverse=True)
        top3 = sorted_nodes[:3]
        while len(top3) < 3:
            top3.append(("-", 0.0))

        logger.info("[%s] === 파이프라인 완료 요약 ===", self.job_id)
        logger.info("[%s] 총 소요시간: %.2fs", self.job_id, total)
        logger.info(
            "[%s] Phase 1: %.2fs | Phase 2: %.2fs | Phase 3: %.2fs",
            self.job_id,
            p1,
            p2,
            p3,
        )
        logger.info(
            "[%s] 가장 느린 노드: %s (%.2fs), %s (%.2fs), %s (%.2fs)",
            self.job_id,
            top3[0][0],
            top3[0][1],
            top3[1][0],
            top3[1][1],
            top3[2][0],
            top3[2][1],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calculate_progress(self) -> float:
        """Return the overall pipeline progress as a value in [0.0, 1.0]."""
        progress = 0.0
        for phase in range(1, TOTAL_PHASES + 1):
            phase_nodes = [n for n, p, _ in PIPELINE_NODES if p == phase]
            completed_in_phase = len(self._completed_nodes & set(phase_nodes))
            total_in_phase = PHASE_NODE_COUNTS[phase]
            progress += PHASE_WEIGHTS[phase] * (completed_in_phase / total_in_phase)
        return progress

    @staticmethod
    def _extract_batch_suffix(node_name: str, state: dict | None) -> str:
        """Build a batch info string for nodes that iterate over PDF chunks.

        Returns a string like `` [batch 2/5]`` or an empty string.
        """
        if state is None:
            return ""
        if node_name not in ("document_parse", "working_queue"):
            return ""

        current = state.get("current_batch_index")
        chunks = state.get("pdf_chunks")
        if current is None or chunks is None:
            return ""

        try:
            total = len(chunks)
            return f" [batch {int(current)}/{total}]"
        except (TypeError, ValueError):
            return ""


# ---------------------------------------------------------------------------
# wrap_node helper
# ---------------------------------------------------------------------------


def wrap_node(node_name: str, fn: Callable, tracker: PipelineTracker) -> Callable:
    """Wrap an async pipeline node function with start/end tracking.

    Args:
        node_name: Name of the pipeline node (must match a key in NODE_META).
        fn: The async node function to wrap. Must accept a single ``state`` arg.
        tracker: The :class:`PipelineTracker` instance for the current job.

    Returns:
        An async wrapper that calls ``tracker.node_start`` / ``tracker.node_end``
        around the original function and re-raises any exception unchanged.
    """

    async def wrapped(state: dict):
        tracker.node_start(node_name, state)
        try:
            result = await fn(state)
            tracker.node_end(node_name)
            return result
        except Exception as e:
            tracker.node_end(node_name, error=e)
            raise

    return wrapped


# ---------------------------------------------------------------------------
# LangGraphCallbackTracker
# ---------------------------------------------------------------------------


from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402


class LangGraphCallbackTracker(BaseCallbackHandler):
    """LangGraph 콜백 기반 노드 추적기

    ainvoke(state, config={"callbacks": [tracker]}) 형태로 주입하여
    그래프 싱글턴을 유지하면서 노드별 계측을 수행한다.
    """

    def __init__(self, tracker: PipelineTracker, state: dict | None = None):
        super().__init__()
        self._tracker = tracker
        self._state = state  # for batch info extraction
        self._active_steps: dict[str, str] = {}  # graph:step:N -> node_name

    def on_chain_start(self, serialized: dict[str, Any], inputs: Any, **kwargs: Any) -> None:
        tags = kwargs.get('tags', [])
        name = kwargs.get('name', '')

        # Filter out internal events and graph-level events
        if 'langsmith:hidden' in tags:
            return
        if name in ('LangGraph', '__start__', ''):
            return

        step_tag = next((t for t in tags if t.startswith('graph:step:')), None)
        if step_tag and name in NODE_META:
            self._active_steps[step_tag] = name
            self._tracker.node_start(name, self._state)

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        tags = kwargs.get('tags', [])

        if 'langsmith:hidden' in tags:
            return

        step_tag = next((t for t in tags if t.startswith('graph:step:')), None)
        if step_tag and step_tag in self._active_steps:
            name = self._active_steps.pop(step_tag)
            self._tracker.node_end(name)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        tags = kwargs.get('tags', [])

        if 'langsmith:hidden' in tags:
            return

        step_tag = next((t for t in tags if t.startswith('graph:step:')), None)
        if step_tag and step_tag in self._active_steps:
            name = self._active_steps.pop(step_tag)
            self._tracker.node_end(name, error=error if isinstance(error, Exception) else Exception(str(error)))

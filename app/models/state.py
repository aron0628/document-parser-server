"""LangGraph 파이프라인 상태 모델"""

from typing import Any, Dict, List, Optional, TypedDict


class PipelineState(TypedDict, total=False):
    """파이프라인 전체 상태를 관리하는 TypedDict"""

    # Input
    pdf_path: str
    language: str
    include_image: bool
    batch_size: int
    test_page: Optional[int]
    upstage_api_key: str
    openai_api_key: str
    job_id: str
    embedding_model: str            # 임베딩 모델명. 기본값: "embedding-passage"
    chunk_size: int                 # 텍스트 분할 크기. 기본값: 1000
    chunk_overlap: int              # 텍스트 분할 오버랩. 기본값: 200

    # Phase 1: Document Parse
    pdf_chunks: List[str]           # 분할된 PDF 파일 경로 목록
    current_batch_index: int        # working_queue_node 루프 카운터
    batch_parse_results: List[Dict[str, Any]]  # 배치별 Upstage API 결과
    merged_parse_result: Dict[str, Any]        # 병합된 파싱 결과

    # Phase 2: Element Processing
    elements: List[Dict[str, Any]]              # 기본 요소 객체 (type, content, page, position)
    image_paths: List[str]                      # 내보낸 이미지 파일 경로
    page_elements: Dict[int, Dict[str, List]]   # {page_num: {images: [], tables: []}}
    image_entities: List[Dict[str, Any]]        # OpenAI Vision 이미지 설명
    table_entities: List[Dict[str, Any]]        # OpenAI 테이블 구조화 결과
    merged_elements: List[Dict[str, Any]]       # entity 키 추가된 요소
    reconstructed_elements: List[Dict[str, Any]]  # 페이지 순서 재구성된 강화 요소

    # Phase 3: Export paths (병렬 브랜치는 서로 다른 키에만 기록)
    html_path: Optional[str]        # un-enriched 요소 기반
    markdown_path: Optional[str]    # un-enriched 요소 기반
    csv_path: Optional[str]         # un-enriched 요소 기반
    pkl_path: Optional[str]         # enriched 요소 기반
    zip_path: Optional[str]
    embedding_count: Optional[int]  # 임베딩 처리된 Document 수. 기본값: None

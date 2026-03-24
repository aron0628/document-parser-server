"""LangGraph 파이프라인 그래프 정의

아키텍처 다이어그램의 15개 노드를 정확하게 매핑:
- Phase 1: split_pdf → working_queue ⟲ document_parse → post_document_parse
- Phase 2: create_elements → export_image → [parallel: entity extraction + quick exports]
- Phase 3: 4개 export 노드 → END
"""

from langgraph.graph import END, StateGraph

from app.models.state import PipelineState

# Phase 1 nodes
from app.pipeline.nodes.split_pdf import split_pdf_node
from app.pipeline.nodes.working_queue import check_queue, working_queue_node
from app.pipeline.nodes.document_parse import document_parse_node
from app.pipeline.nodes.post_document_parse import post_document_parse_node

# Phase 2 nodes
from app.pipeline.nodes.create_elements import create_elements_node
from app.pipeline.nodes.keyword_preprocess import keyword_preprocess_node
from app.pipeline.nodes.export_image import export_image_node
from app.pipeline.nodes.page_elements_extractor import page_elements_extractor_node
from app.pipeline.nodes.image_entity_extractor import image_entity_extractor_node
from app.pipeline.nodes.table_entity_extractor import table_entity_extractor_node
from app.pipeline.nodes.merge_entity import merge_entity_node
from app.pipeline.nodes.reconstruct_elements import reconstruct_elements_node

# Phase 3 nodes
from app.pipeline.nodes.export_html import export_html_node
from app.pipeline.nodes.export_markdown import export_markdown_node
from app.pipeline.nodes.export_table_csv import export_table_csv_node
from app.pipeline.nodes.langchain_document import langchain_document_node
from app.pipeline.nodes.embedding import embedding_node
from app.pipeline.nodes.raptor import raptor_node


def build_graph() -> StateGraph:
    """파이프라인 그래프 구성 (compile 미포함)"""

    graph = StateGraph(PipelineState)

    # ── Phase 1: Document Parse (sequential with loop) ──

    graph.add_node("split_pdf", split_pdf_node)
    graph.add_node("working_queue", working_queue_node)
    graph.add_node("document_parse", document_parse_node)
    graph.add_node("post_document_parse", post_document_parse_node)

    graph.set_entry_point("split_pdf")
    graph.add_edge("split_pdf", "working_queue")

    # working_queue → True: document_parse (처리할 배치 있음)
    # working_queue → False: post_document_parse (모든 배치 완료)
    graph.add_conditional_edges(
        "working_queue",
        check_queue,
        {True: "document_parse", False: "post_document_parse"},
    )
    graph.add_edge("document_parse", "working_queue")  # loop back

    # ── Phase 2: Element Processing ──

    graph.add_node("create_elements", create_elements_node)
    graph.add_node("keyword_preprocess", keyword_preprocess_node)
    graph.add_node("export_image", export_image_node)
    graph.add_node("page_elements_extractor", page_elements_extractor_node)
    graph.add_node("image_entity_extractor", image_entity_extractor_node)
    graph.add_node("table_entity_extractor", table_entity_extractor_node)
    graph.add_node("merge_entity", merge_entity_node)
    graph.add_node("reconstruct_elements", reconstruct_elements_node)

    graph.add_edge("post_document_parse", "create_elements")
    graph.add_edge("create_elements", "keyword_preprocess")
    graph.add_edge("keyword_preprocess", "export_image")

    # export_image에서 4갈래 분기 (parallel fan-out)
    # Branch A: Entity extraction path (enriched → PKL)
    graph.add_edge("export_image", "page_elements_extractor")

    # page_elements_extractor에서 image/table 병렬 처리
    graph.add_edge("page_elements_extractor", "image_entity_extractor")   # parallel
    graph.add_edge("page_elements_extractor", "table_entity_extractor")   # parallel

    # image/table → merge (fan-in)
    graph.add_edge("image_entity_extractor", "merge_entity")
    graph.add_edge("table_entity_extractor", "merge_entity")

    graph.add_edge("merge_entity", "reconstruct_elements")

    # ── Phase 3: Export ──

    graph.add_node("langchain_document", langchain_document_node)
    graph.add_node("embedding", embedding_node)
    graph.add_node("export_html", export_html_node)
    graph.add_node("export_markdown", export_markdown_node)
    graph.add_node("export_table_csv", export_table_csv_node)

    # Enriched export (after entity extraction)
    graph.add_edge("reconstruct_elements", "langchain_document")
    graph.add_edge("langchain_document", "embedding")
    # RAPTOR (conditional: enable_raptor 플래그에 따라 분기)
    graph.add_node("raptor", raptor_node)

    def check_raptor(state: PipelineState) -> bool:
        return state.get("enable_raptor", False)

    graph.add_conditional_edges(
        "embedding",
        check_raptor,
        {True: "raptor", False: END},
    )
    graph.add_edge("raptor", END)

    # Quick exports (un-enriched, parallel with entity extraction)
    graph.add_edge("export_image", "export_html")
    graph.add_edge("export_image", "export_markdown")
    graph.add_edge("export_image", "export_table_csv")
    graph.add_edge("export_html", END)
    graph.add_edge("export_markdown", END)
    graph.add_edge("export_table_csv", END)

    return graph  # compile() 호출하지 않음


def compile_graph(checkpointer=None):
    """checkpointer를 주입하여 그래프 컴파일"""
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)

"""테이블 멀티모달 확장 단위 테스트"""

import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from langchain_core.messages import HumanMessage

from app.pipeline.external.openai_client import _build_table_messages
from app.pipeline.nodes.table_entity_extractor import table_entity_extractor_node
from app.pipeline.nodes.export_image import _FALLBACK_PNG


# ---------------------------------------------------------------------------
# _build_table_messages 테스트
# ---------------------------------------------------------------------------


def test_build_table_messages_html_and_image(tmp_path):
    """HTML + 이미지 → HumanMessage, content가 리스트, 멀티모달 프롬프트"""
    img_file = tmp_path / "test_table.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n")  # 임의 PNG 바이트

    message = _build_table_messages("html data", "Korean", image_path=str(img_file))

    assert isinstance(message, HumanMessage)
    content = message.content
    assert isinstance(content, list), "HTML+이미지일 때 content는 리스트여야 함"

    text_parts = [p for p in content if p.get("type") == "text"]
    assert text_parts, "텍스트 파트가 존재해야 함"
    assert "추출 데이터" in text_parts[0]["text"]

    image_parts = [p for p in content if p.get("type") == "image_url"]
    assert image_parts, "image_url 파트가 존재해야 함"
    assert "url" in image_parts[0]["image_url"]


def test_build_table_messages_html_only():
    """HTML만 → HumanMessage, content가 문자열, '테이블 데이터' 키워드 포함"""
    message = _build_table_messages("html data", "Korean")

    assert isinstance(message, HumanMessage)
    content = message.content
    assert isinstance(content, str), "HTML만 있을 때 content는 문자열이어야 함"
    assert "테이블 데이터" in content


def test_build_table_messages_image_only(tmp_path):
    """이미지만 → HumanMessage, content가 리스트, '테이블 이미지를 분석' 텍스트 포함"""
    img_file = tmp_path / "test_table.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    message = _build_table_messages("", "Korean", image_path=str(img_file))

    assert isinstance(message, HumanMessage)
    content = message.content
    assert isinstance(content, list), "이미지만 있을 때 content는 리스트여야 함"

    text_parts = [p for p in content if p.get("type") == "text"]
    assert text_parts, "텍스트 파트가 존재해야 함"
    assert "테이블 이미지를 분석" in text_parts[0]["text"]

    image_parts = [p for p in content if p.get("type") == "image_url"]
    assert image_parts, "image_url 파트가 존재해야 함"


def test_build_table_messages_neither():
    """HTML도 이미지도 없음 → HumanMessage, content가 문자열 (기존 텍스트 프롬프트)"""
    message = _build_table_messages("", "Korean")

    assert isinstance(message, HumanMessage)
    content = message.content
    assert isinstance(content, str), "둘 다 없을 때 content는 문자열이어야 함"


# ---------------------------------------------------------------------------
# table_entity_extractor_node 테스트
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("app.pipeline.nodes.table_entity_extractor.extract_table")
async def test_table_extractor_html_and_valid_image(mock_extract, tmp_path):
    """HTML + 유효 이미지 → extract_table이 image_path와 함께 호출됨"""
    mock_extract = AsyncMock(return_value="| col1 | col2 |")

    img_file = tmp_path / "test_page_1_table_0.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n")  # fallback이 아닌 임의 바이트

    state = {
        "job_id": "test-job",
                "language": "Korean",
        "page_elements": {
            1: {
                "tables": [
                    {"id": "tbl-1", "page": 1, "position": 0, "html": "<table>...</table>"}
                ],
                "images": [],
            }
        },
        "image_paths": [str(img_file)],
    }

    with patch(
        "app.pipeline.nodes.table_entity_extractor.extract_table",
        new=mock_extract,
    ):
        config = {"configurable": {"openai_api_key": "test-key"}}
        result = await table_entity_extractor_node(state, config)

    mock_extract.assert_called_once()
    call_kwargs = mock_extract.call_args
    assert call_kwargs.kwargs.get("image_path") == str(img_file) or (
        len(call_kwargs.args) >= 4 and call_kwargs.args[3] == str(img_file)
    ), "유효 이미지가 있을 때 image_path로 경로가 전달되어야 함"

    assert result["table_entities"][0]["structured_table"] == "| col1 | col2 |"


@pytest.mark.asyncio
@patch("app.pipeline.nodes.table_entity_extractor.extract_table")
async def test_table_extractor_fallback_png_treated_as_no_image(mock_extract, tmp_path):
    """fallback PNG → image_path=None으로 extract_table 호출"""
    mock_extract = AsyncMock(return_value="| col1 | col2 |")

    img_file = tmp_path / "test_page_1_table_0.png"
    img_file.write_bytes(_FALLBACK_PNG)

    state = {
        "job_id": "test-job",
                "language": "Korean",
        "page_elements": {
            1: {
                "tables": [
                    {"id": "tbl-1", "page": 1, "position": 0, "html": "<table>...</table>"}
                ],
                "images": [],
            }
        },
        "image_paths": [str(img_file)],
    }

    with patch(
        "app.pipeline.nodes.table_entity_extractor.extract_table",
        new=mock_extract,
    ):
        config = {"configurable": {"openai_api_key": "test-key"}}
        result = await table_entity_extractor_node(state, config)

    mock_extract.assert_called_once()
    call_kwargs = mock_extract.call_args
    passed_image_path = call_kwargs.kwargs.get("image_path")
    assert passed_image_path is None, "fallback PNG일 때 image_path는 None이어야 함"


@pytest.mark.asyncio
@patch("app.pipeline.nodes.table_entity_extractor.extract_table")
async def test_table_extractor_no_html_with_valid_image(mock_extract, tmp_path):
    """HTML 없음 + 유효 이미지 → image_path 전달되어 정상 처리"""
    mock_extract = AsyncMock(return_value="| col1 | col2 |")

    img_file = tmp_path / "test_page_1_table_0.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    state = {
        "job_id": "test-job",
                "language": "Korean",
        "page_elements": {
            1: {
                "tables": [
                    {"id": "tbl-1", "page": 1, "position": 0, "html": ""}
                ],
                "images": [],
            }
        },
        "image_paths": [str(img_file)],
    }

    with patch(
        "app.pipeline.nodes.table_entity_extractor.extract_table",
        new=mock_extract,
    ):
        config = {"configurable": {"openai_api_key": "test-key"}}
        result = await table_entity_extractor_node(state, config)

    mock_extract.assert_called_once()
    call_kwargs = mock_extract.call_args

    table_content_arg = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("table_content", "")
    assert table_content_arg == "", "HTML 없을 때 table_content는 빈 문자열이어야 함"

    passed_image_path = call_kwargs.kwargs.get("image_path")
    assert passed_image_path == str(img_file), "유효 이미지 경로가 전달되어야 함"

    assert "error" not in result["table_entities"][0], "에러 없이 정상 처리되어야 함"


@pytest.mark.asyncio
@patch("app.pipeline.nodes.table_entity_extractor.extract_table")
async def test_table_extractor_no_html_no_image_returns_error(mock_extract):
    """HTML 없음 + 이미지 없음 → extract_table 미호출, error 키 포함"""
    mock_extract = AsyncMock(return_value="| col1 | col2 |")

    state = {
        "job_id": "test-job",
                "language": "Korean",
        "page_elements": {
            1: {
                "tables": [
                    {"id": "tbl-1", "page": 1, "position": 0, "html": ""}
                ],
                "images": [],
            }
        },
        "image_paths": [],
    }

    with patch(
        "app.pipeline.nodes.table_entity_extractor.extract_table",
        new=mock_extract,
    ):
        config = {"configurable": {"openai_api_key": "test-key"}}
        result = await table_entity_extractor_node(state, config)

    mock_extract.assert_not_called()
    assert "error" in result["table_entities"][0], "에러 키가 결과에 포함되어야 함"

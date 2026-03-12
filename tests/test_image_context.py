"""이미지 설명 텍스트 컨텍스트 주입 단위 테스트"""

import base64
from unittest.mock import AsyncMock, patch

import pytest

from app.pipeline.nodes.image_entity_extractor import (
    MAX_CONTEXT_CHARS,
    _build_page_text_map,
)


# ---------------------------------------------------------------------------
# _build_page_text_map
# ---------------------------------------------------------------------------


def test_build_page_text_map_groups_by_page():
    """여러 페이지의 텍스트 요소가 페이지별로 그룹화된다"""
    elements = [
        {"type": "text", "page": 1, "content": "첫 번째"},
        {"type": "text", "page": 1, "content": "두 번째"},
        {"type": "text", "page": 2, "content": "페이지2 텍스트"},
        {"type": "image", "page": 1, "content": ""},
    ]
    result = _build_page_text_map(elements)

    assert result[1] == "첫 번째\n두 번째"
    assert result[2] == "페이지2 텍스트"
    assert len(result) == 2


def test_build_page_text_map_truncates_at_max_chars():
    """2000자 초과 시 잘림 동작"""
    long_text = "가" * 1500
    elements = [
        {"type": "text", "page": 1, "content": long_text},
        {"type": "text", "page": 1, "content": long_text},
    ]
    result = _build_page_text_map(elements)

    assert len(result[1]) == MAX_CONTEXT_CHARS


def test_build_page_text_map_filters_empty_content():
    """빈 content 문자열은 필터링된다"""
    elements = [
        {"type": "text", "page": 1, "content": ""},
        {"type": "text", "page": 1, "content": "유효"},
    ]
    result = _build_page_text_map(elements)

    assert result[1] == "유효"


def test_build_page_text_map_excludes_non_text():
    """type이 text가 아닌 요소는 포함되지 않는다"""
    elements = [
        {"type": "image", "page": 1, "content": "이미지 설명"},
        {"type": "table", "page": 1, "content": "테이블 내용"},
    ]
    result = _build_page_text_map(elements)

    assert len(result) == 0


def test_build_page_text_map_empty_elements():
    """빈 elements 리스트는 빈 딕셔너리를 반환한다"""
    assert _build_page_text_map([]) == {}


def test_build_page_text_map_no_text_page_excluded():
    """텍스트 요소가 없는 페이지는 맵에 포함되지 않는다"""
    elements = [
        {"type": "text", "page": 1, "content": "있음"},
        {"type": "image", "page": 2, "content": "이미지만"},
    ]
    result = _build_page_text_map(elements)

    assert 1 in result
    assert 2 not in result


# ---------------------------------------------------------------------------
# describe_image 메시지 구조
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_describe_image_without_context(tmp_path):
    """context=None → 2개 content 블록 (text + image_url)"""
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    captured_messages = []

    async def mock_call_openai(api_key, messages, **kwargs):
        captured_messages.append(messages)
        return {"choices": [{"message": {"content": "설명"}}]}

    with patch(
        "app.pipeline.external.openai_client._call_openai", side_effect=mock_call_openai
    ):
        from app.pipeline.external.openai_client import describe_image

        await describe_image(str(img), "test-key", "Korean")

    content = captured_messages[0][0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert "맥락을 참고" not in content[0]["text"]


@pytest.mark.asyncio
async def test_describe_image_with_context(tmp_path):
    """context="페이지 텍스트" → 3개 content 블록 (context + prompt + image_url)"""
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    captured_messages = []

    async def mock_call_openai(api_key, messages, **kwargs):
        captured_messages.append(messages)
        return {"choices": [{"message": {"content": "설명"}}]}

    with patch(
        "app.pipeline.external.openai_client._call_openai", side_effect=mock_call_openai
    ):
        from app.pipeline.external.openai_client import describe_image

        await describe_image(str(img), "test-key", "Korean", context="페이지 텍스트")

    content = captured_messages[0][0]["content"]
    assert len(content) == 3
    assert "페이지 텍스트" in content[0]["text"]
    assert "맥락을 참고" in content[1]["text"]
    assert content[2]["type"] == "image_url"


@pytest.mark.asyncio
async def test_describe_image_with_empty_string_context(tmp_path):
    """context="" → None과 동일하게 2개 content 블록"""
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    captured_messages = []

    async def mock_call_openai(api_key, messages, **kwargs):
        captured_messages.append(messages)
        return {"choices": [{"message": {"content": "설명"}}]}

    with patch(
        "app.pipeline.external.openai_client._call_openai", side_effect=mock_call_openai
    ):
        from app.pipeline.external.openai_client import describe_image

        await describe_image(str(img), "test-key", "Korean", context="")

    content = captured_messages[0][0]["content"]
    assert len(content) == 2

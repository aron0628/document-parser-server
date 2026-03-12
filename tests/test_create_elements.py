"""create_elements 노드 단위 테스트"""

import pytest
from app.pipeline.nodes.create_elements import (
    _parse_elements,
    _strip_base64_from_merged,
    create_elements_node,
)


# ---------------------------------------------------------------------------
# _parse_elements
# ---------------------------------------------------------------------------


def test_image_element_preserves_base64():
    """figure 카테고리 + base64_encoding 있는 element → 변환 결과에 base64_encoding 포함"""
    merged = {
        "elements": [
            {
                "id": 1,
                "category": "figure",
                "content": {"text": "", "html": ""},
                "page": 1,
                "base64_encoding": "abc123==",
            }
        ]
    }
    result = _parse_elements(merged)
    assert len(result) == 1
    elem = result[0]
    assert elem["type"] == "image"
    assert elem["base64_encoding"] == "abc123=="


def test_table_element_excludes_base64():
    """table 카테고리 + base64_encoding 있는 element → 변환 결과에 base64_encoding 미포함"""
    merged = {
        "elements": [
            {
                "id": 2,
                "category": "table",
                "content": {"text": "cell", "html": "<table/>"},
                "page": 1,
                "base64_encoding": "shouldnotappear==",
            }
        ]
    }
    result = _parse_elements(merged)
    assert len(result) == 1
    elem = result[0]
    assert elem["type"] == "table"
    assert "base64_encoding" not in elem


def test_text_element_excludes_base64():
    """text 카테고리 → base64_encoding 미포함"""
    merged = {
        "elements": [
            {
                "id": 3,
                "category": "paragraph",
                "content": {"text": "hello", "html": "<p>hello</p>"},
                "page": 2,
            }
        ]
    }
    result = _parse_elements(merged)
    assert len(result) == 1
    elem = result[0]
    assert elem["type"] == "text"
    assert "base64_encoding" not in elem


def test_image_element_without_base64_omits_key():
    """figure 카테고리지만 base64_encoding 없음 → 키 자체가 없어야 함"""
    merged = {
        "elements": [
            {
                "id": 4,
                "category": "figure",
                "content": {"text": "", "html": ""},
                "page": 1,
            }
        ]
    }
    result = _parse_elements(merged)
    elem = result[0]
    assert elem["type"] == "image"
    assert "base64_encoding" not in elem


# ---------------------------------------------------------------------------
# _strip_base64_from_merged
# ---------------------------------------------------------------------------


def test_strip_base64_removes_field():
    """base64_encoding 있는 elements → strip 후 base64_encoding 없음"""
    merged = {
        "elements": [
            {"id": 1, "category": "figure", "base64_encoding": "data=="},
            {"id": 2, "category": "table"},
        ]
    }
    stripped = _strip_base64_from_merged(merged)
    for elem in stripped["elements"]:
        assert "base64_encoding" not in elem


def test_strip_base64_does_not_mutate_original():
    """원본 merged는 strip 후에도 불변"""
    merged = {
        "elements": [
            {"id": 1, "category": "figure", "base64_encoding": "data=="},
        ]
    }
    _strip_base64_from_merged(merged)
    assert merged["elements"][0]["base64_encoding"] == "data=="


# ---------------------------------------------------------------------------
# create_elements_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_elements_node_returns_elements_and_stripped_merged():
    """async 노드 호출 → elements와 stripped merged_parse_result 모두 반환"""
    state = {
        "job_id": "test-job-1",
        "merged_parse_result": {
            "elements": [
                {
                    "id": 1,
                    "category": "figure",
                    "content": {"text": "", "html": ""},
                    "page": 1,
                    "base64_encoding": "imgdata==",
                },
                {
                    "id": 2,
                    "category": "paragraph",
                    "content": {"text": "hello", "html": "<p>hello</p>"},
                    "page": 1,
                },
            ]
        },
    }
    result = await create_elements_node(state)

    assert "elements" in result
    assert "merged_parse_result" in result

    # elements: image에는 base64 있고, text에는 없어야 함
    elements = result["elements"]
    image_elem = next(e for e in elements if e["type"] == "image")
    text_elem = next(e for e in elements if e["type"] == "text")
    assert image_elem["base64_encoding"] == "imgdata=="
    assert "base64_encoding" not in text_elem

    # merged_parse_result에서는 base64 제거됨
    stripped = result["merged_parse_result"]
    for elem in stripped["elements"]:
        assert "base64_encoding" not in elem

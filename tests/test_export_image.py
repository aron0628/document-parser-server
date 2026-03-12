"""export_image 노드 단위 테스트"""

import base64
import pytest
from pathlib import Path
from unittest.mock import patch

from app.pipeline.nodes.export_image import export_image_node, _FALLBACK_PNG


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_work_dir(tmp_path):
    with patch("app.pipeline.nodes.export_image.get_work_dir", return_value=tmp_path):
        yield tmp_path


# ---------------------------------------------------------------------------
# 헬퍼: 최소 유효 PNG base64
# ---------------------------------------------------------------------------

_VALID_PNG_B64 = base64.b64encode(_FALLBACK_PNG).decode()


# ---------------------------------------------------------------------------
# 테스트 케이스
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_base64_creates_png_file(mock_work_dir):
    """유효한 PNG base64 → 파일 생성되고 image_paths에 포함"""
    state = {
        "job_id": "job-1",
        "include_image": True,
        "elements": [
            {
                "type": "image",
                "page": 1,
                "position": 0,
                "base64_encoding": _VALID_PNG_B64,
            }
        ],
    }
    result = await export_image_node(state)

    image_paths = result["image_paths"]
    assert len(image_paths) == 1
    img_path = Path(image_paths[0])
    assert img_path.exists()
    assert img_path.suffix == ".png"
    assert img_path.read_bytes() == _FALLBACK_PNG


@pytest.mark.asyncio
async def test_invalid_base64_saves_fallback_png(mock_work_dir):
    """잘못된 base64 → fallback PNG 저장됨"""
    state = {
        "job_id": "job-2",
        "include_image": True,
        "elements": [
            {
                "type": "image",
                "page": 2,
                "position": 1,
                "base64_encoding": "!!!not-valid-base64!!!",
            }
        ],
    }
    result = await export_image_node(state)

    image_paths = result["image_paths"]
    assert len(image_paths) == 1
    img_path = Path(image_paths[0])
    assert img_path.exists()
    assert img_path.read_bytes() == _FALLBACK_PNG


@pytest.mark.asyncio
async def test_missing_base64_saves_fallback_png(mock_work_dir):
    """base64_encoding 키 없음 → fallback PNG 저장됨"""
    state = {
        "job_id": "job-3",
        "include_image": True,
        "elements": [
            {
                "type": "image",
                "page": 3,
                "position": 2,
                # base64_encoding 키 의도적으로 누락
            }
        ],
    }
    result = await export_image_node(state)

    image_paths = result["image_paths"]
    assert len(image_paths) == 1
    img_path = Path(image_paths[0])
    assert img_path.exists()
    assert img_path.read_bytes() == _FALLBACK_PNG


@pytest.mark.asyncio
async def test_include_image_false_returns_empty_list(mock_work_dir):
    """include_image=False → image_paths는 빈 리스트, 파일 생성 없음"""
    state = {
        "job_id": "job-4",
        "include_image": False,
        "elements": [
            {
                "type": "image",
                "page": 1,
                "position": 0,
                "base64_encoding": _VALID_PNG_B64,
            }
        ],
    }
    result = await export_image_node(state)

    assert result["image_paths"] == []
    # get_work_dir가 호출되지 않으므로 images 디렉토리 없음
    assert not (mock_work_dir / "images").exists()


@pytest.mark.asyncio
async def test_table_element_creates_png_with_table_prefix(mock_work_dir):
    """table 타입 + 유효한 base64 → table_ prefix 파일명으로 PNG 저장"""
    state = {
        "job_id": "job-t1",
        "include_image": True,
        "elements": [
            {
                "type": "table",
                "page": 1,
                "position": 0,
                "base64_encoding": _VALID_PNG_B64,
            }
        ],
    }
    result = await export_image_node(state)

    image_paths = result["image_paths"]
    assert len(image_paths) == 1
    img_path = Path(image_paths[0])
    assert img_path.exists()
    assert "table_" in img_path.name
    assert img_path.read_bytes() == _FALLBACK_PNG


@pytest.mark.asyncio
async def test_mixed_image_and_table_elements(mock_work_dir):
    """image + table 혼합 → 둘 다 PNG로 저장, 파일명 prefix 구분"""
    state = {
        "job_id": "job-t2",
        "include_image": True,
        "elements": [
            {
                "type": "image",
                "page": 1,
                "position": 0,
                "base64_encoding": _VALID_PNG_B64,
            },
            {
                "type": "table",
                "page": 1,
                "position": 1,
                "base64_encoding": _VALID_PNG_B64,
            },
        ],
    }
    result = await export_image_node(state)

    image_paths = result["image_paths"]
    assert len(image_paths) == 2
    names = [Path(p).name for p in image_paths]
    assert any("img_" in n for n in names)
    assert any("table_" in n for n in names)


@pytest.mark.asyncio
async def test_include_image_false_skips_table_too(mock_work_dir):
    """include_image=False → table 타입도 건너뜀"""
    state = {
        "job_id": "job-t3",
        "include_image": False,
        "elements": [
            {
                "type": "table",
                "page": 1,
                "position": 0,
                "base64_encoding": _VALID_PNG_B64,
            }
        ],
    }
    result = await export_image_node(state)

    assert result["image_paths"] == []
    assert not (mock_work_dir / "images").exists()


@pytest.mark.asyncio
async def test_return_value_does_not_contain_elements_key(mock_work_dir):
    """반환값에 'elements' 키 미포함 (elements in-place 수정 없음)"""
    state = {
        "job_id": "job-5",
        "include_image": True,
        "elements": [
            {
                "type": "image",
                "page": 1,
                "position": 0,
                "base64_encoding": _VALID_PNG_B64,
            }
        ],
    }
    result = await export_image_node(state)

    assert "elements" not in result

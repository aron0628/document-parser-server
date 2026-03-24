"""Upstage Document Digitization API 클라이언트 (httpx.AsyncClient)"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)

UPSTAGE_API_URL = "https://api.upstage.ai/v1/document-digitization"
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0

# 기본 요청 파라미터
DEFAULT_PARAMS = {
    "model": "document-parse",
    "mode": "standard",
    "output_formats": "['markdown', 'html', 'text']",
    "ocr": "auto",
    "merge_multipage_tables": "false",
    "coordinates": "true",
    "base64_encoding": "['figure', 'table']",
}


async def parse_document(
    pdf_path: str,
    api_key: str,
    timeout: float = 300.0,
    model: str = DEFAULT_PARAMS["model"],
    mode: str = DEFAULT_PARAMS["mode"],
    output_formats: str = DEFAULT_PARAMS["output_formats"],
    ocr: str = DEFAULT_PARAMS["ocr"],
    merge_multipage_tables: str = DEFAULT_PARAMS["merge_multipage_tables"],
    coordinates: str = DEFAULT_PARAMS["coordinates"],
    base64_encoding: str = DEFAULT_PARAMS["base64_encoding"],
) -> Dict[str, Any]:
    """Upstage Document Digitization API로 PDF 레이아웃 분석

    Args:
        pdf_path: PDF 파일 경로
        api_key: Upstage API 키
        timeout: 요청 타임아웃 (초)
        model: 모델 지정 (기본: "document-parse")
        mode: 처리 모드 (기본: "standard")
        output_formats: 출력 형식 (기본: "['markdown', 'html', 'text']")
        ocr: OCR 모드 (기본: "auto")
        merge_multipage_tables: 멀티페이지 테이블 병합 (기본: "true")
        coordinates: 좌표 포함 여부 (기본: "true")
        base64_encoding: base64 인코딩 대상 (기본: "['figure', 'table']")

    Returns:
        레이아웃 분석 결과 JSON
    """
    headers = {"Authorization": f"Bearer {api_key}"}

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                with open(pdf_path, "rb") as f:
                    files = {"document": (Path(pdf_path).name, f, "application/pdf")}
                    data = {
                        "model": model,
                        "mode": mode,
                        "output_formats": output_formats,
                        "ocr": ocr,
                        "merge_multipage_tables": merge_multipage_tables,
                        "coordinates": coordinates,
                        "base64_encoding": base64_encoding,
                    }
                    response = await client.post(
                        UPSTAGE_API_URL,
                        headers=headers,
                        files=files,
                        data=data,
                    )

                if response.status_code == 429:
                    # Rate limit - exponential backoff
                    wait = INITIAL_BACKOFF * (2**attempt)
                    logger.warning(
                        f"Rate limit hit, retrying in {wait}s (attempt {attempt+1}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500 and attempt < MAX_RETRIES - 1:
                wait = INITIAL_BACKOFF * (2**attempt)
                logger.warning(
                    f"Server error {e.response.status_code}, retrying in {wait}s"
                )
                await asyncio.sleep(wait)
                continue
            raise
        except httpx.RequestError as e:
            if attempt < MAX_RETRIES - 1:
                wait = INITIAL_BACKOFF * (2**attempt)
                logger.warning(f"Request error: {e}, retrying in {wait}s")
                await asyncio.sleep(wait)
                continue
            raise

    raise RuntimeError(f"Upstage API 호출 실패: {MAX_RETRIES}회 재시도 후에도 실패")

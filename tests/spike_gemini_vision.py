"""Gemini Vision spike test — ChatGoogleGenerativeAI가 base64 image_url을 처리하는지 확인.

실행: GOOGLE_API_KEY=xxx uv run python tests/spike_gemini_vision.py
"""
import asyncio
import base64
import os
import sys

async def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY 환경변수가 필요합니다.")
        sys.exit(1)

    # 의존성 확인
    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import HumanMessage
    except ImportError as e:
        print(f"ERROR: 필요한 패키지가 없습니다: {e}")
        print("설치: uv add langchain-google-genai")
        sys.exit(1)

    # 1x1 투명 PNG (최소 이미지)
    tiny_png = base64.b64encode(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
        b'\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
        b'\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    ).decode("utf-8")

    b64_url = f"data:image/png;base64,{tiny_png}"

    # Test 1: OpenAI 호환 image_url 포맷
    print("=== Test 1: image_url 포맷 ===")
    try:
        model = init_chat_model(
            "gemini-2.0-flash",  # 안정 모델 사용
            model_provider="google_genai",
            google_api_key=api_key,
        )
        message = HumanMessage(content=[
            {"type": "text", "text": "이 이미지를 한 문장으로 설명해주세요."},
            {"type": "image_url", "image_url": {"url": b64_url}},
        ])
        response = await model.ainvoke([message])
        print(f"SUCCESS: {response.content[:200]}")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")

        # Test 2: Fallback — inline_data 포맷
        print("\n=== Test 2: inline_data fallback ===")
        try:
            message2 = HumanMessage(content=[
                {"type": "text", "text": "이 이미지를 한 문장으로 설명해주세요."},
                {
                    "type": "media",
                    "mime_type": "image/png",
                    "data": tiny_png,
                },
            ])
            response2 = await model.ainvoke([message2])
            print(f"SUCCESS (fallback): {response2.content[:200]}")
        except Exception as e2:
            print(f"FAIL (fallback): {type(e2).__name__}: {e2}")

    print("\n완료.")

if __name__ == "__main__":
    asyncio.run(main())

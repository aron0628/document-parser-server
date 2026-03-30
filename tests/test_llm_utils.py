"""load_chat_model 유틸리티 단위 테스트"""

from unittest.mock import patch, MagicMock

import pytest


class TestLoadChatModel:
    """load_chat_model 함수 테스트"""

    @patch("app.pipeline.external.llm_utils.init_chat_model")
    def test_openai_model(self, mock_init):
        """OpenAI 모델 로드"""
        from app.pipeline.external.llm_utils import load_chat_model
        mock_init.return_value = MagicMock()
        load_chat_model("openai/gpt-4o")
        mock_init.assert_called_once_with("gpt-4o", model_provider="openai")

    @patch("app.pipeline.external.llm_utils.init_chat_model")
    def test_google_genai_model(self, mock_init):
        """Google GenAI 모델 로드 (정규 이름)"""
        from app.pipeline.external.llm_utils import load_chat_model
        mock_init.return_value = MagicMock()
        load_chat_model("google_genai/gemini-2.5-flash")
        mock_init.assert_called_once_with("gemini-2.5-flash", model_provider="google_genai")

    @patch("app.pipeline.external.llm_utils.init_chat_model")
    def test_gemini_alias(self, mock_init):
        """gemini alias가 google_genai로 정규화"""
        from app.pipeline.external.llm_utils import load_chat_model
        mock_init.return_value = MagicMock()
        load_chat_model("gemini/gemini-2.5-flash")
        mock_init.assert_called_once_with("gemini-2.5-flash", model_provider="google_genai")

    @patch("app.pipeline.external.llm_utils.init_chat_model")
    def test_xai_model(self, mock_init):
        """xAI 모델 로드"""
        from app.pipeline.external.llm_utils import load_chat_model
        mock_init.return_value = MagicMock()
        load_chat_model("xai/grok-3")
        mock_init.assert_called_once_with("grok-3", model_provider="xai")

    @patch("app.pipeline.external.llm_utils.init_chat_model")
    def test_grok_alias(self, mock_init):
        """grok alias가 xai로 정규화"""
        from app.pipeline.external.llm_utils import load_chat_model
        mock_init.return_value = MagicMock()
        load_chat_model("grok/grok-3")
        mock_init.assert_called_once_with("grok-3", model_provider="xai")

    def test_no_provider_prefix(self):
        """프로바이더 접두사 없으면 ValueError"""
        from app.pipeline.external.llm_utils import load_chat_model
        with pytest.raises(ValueError, match="프로바이더 접두사가 없습니다"):
            load_chat_model("gpt-4o")

    @patch("app.pipeline.external.llm_utils.init_chat_model")
    def test_kwargs_forwarded(self, mock_init):
        """추가 kwargs가 init_chat_model에 전달"""
        from app.pipeline.external.llm_utils import load_chat_model
        mock_init.return_value = MagicMock()
        load_chat_model("openai/gpt-4o", temperature=0.5, api_key="sk-test")
        mock_init.assert_called_once_with(
            "gpt-4o", model_provider="openai", temperature=0.5, api_key="sk-test"
        )


class TestLoadChatModelWithRetry:
    """load_chat_model_with_retry 함수 테스트"""

    @patch("app.pipeline.external.llm_utils.init_chat_model")
    def test_returns_model_with_retry(self, mock_init):
        """with_retry가 적용된 모델 반환"""
        from app.pipeline.external.llm_utils import load_chat_model_with_retry
        mock_model = MagicMock()
        mock_model.with_retry.return_value = MagicMock()
        mock_init.return_value = mock_model
        result = load_chat_model_with_retry("openai/gpt-4o")
        mock_model.with_retry.assert_called_once()
        assert result is mock_model.with_retry.return_value

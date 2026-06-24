from __future__ import annotations

import pytest

from customer_service_app.core.exceptions import AppError
from customer_service_app.services.question_preprocessor import QuestionPreprocessor


def test_question_preprocessor_normalizes_whitespace_and_detects_refund() -> None:
    """用户问题进入 LLM 前会先做简单清洗。"""

    profile = QuestionPreprocessor().analyze("  我的订单 202606040001   想申请退款  ")

    assert profile.normalized_question == "我的订单 202606040001 想申请退款"
    assert profile.intent == "refund"
    assert profile.contains_order_id is True
    assert profile.as_trace_metadata()["changed"] is True


def test_question_preprocessor_rejects_blank_question() -> None:
    """空问题不用继续请求模型。"""

    with pytest.raises(AppError) as exc_info:
        QuestionPreprocessor().analyze("   ")

    assert exc_info.value.code == "invalid_question"
    assert exc_info.value.status_code == 400

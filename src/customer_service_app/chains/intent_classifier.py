from __future__ import annotations

from langchain_core.runnables import RunnableLambda

from customer_service_app.services.question_preprocessor import QuestionPreprocessor


def build_intent_classifier() -> RunnableLambda:
    """Wrap the existing keyword-based QuestionPreprocessor as an LCEL Runnable.

    Input: a plain ``str`` (the user question).
    Output: the ``QuestionProfile`` dataclass with normalized text and intent label.

    LLM-based classification can be layered on top later without changing this
    interface — just prepend or replace the RunnableLambda.
    """
    preprocessor = QuestionPreprocessor()
    return RunnableLambda(lambda question: preprocessor.analyze(question))

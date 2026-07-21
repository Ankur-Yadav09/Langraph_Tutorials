from pydantic import BaseModel, Field
from typing import List


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(
        description="~50 word detailed answer to the question."
    )
    search_queries: List[str] = Field(
        description="1 search query for researching improvements to address the critique of your current answer."
    )
    reflection_missing: str = Field(
        description="Critique of what is missing in the answer."
    )
    reflection_superfluous: str = Field(
        description="Critique of what is superfluous or unnecessarily verbose in the answer."
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations or references that support the revised answer."
    )

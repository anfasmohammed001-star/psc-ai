"""
Data models and validation schemas for the Kerala PSC Exam Prep AI system.
"""

from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator


class LanguageEnum(str, Enum):
    MALAYALAM = "Malayalam"
    ENGLISH = "English"


class QuestionTypeEnum(str, Enum):
    MCQ = "MCQ"
    TRUE_FALSE = "TRUE_FALSE"
    DESCRIPTIVE = "DESCRIPTIVE"


class DifficultyEnum(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class QuestionBase(BaseModel):
    """Base model for any extracted/generated question."""
    question_text: str = Field(..., description="The main text of the question.")
    difficulty: DifficultyEnum = Field(default=DifficultyEnum.MEDIUM)
    language: LanguageEnum = Field(default=LanguageEnum.ENGLISH)
    concept_tags: List[str] = Field(default_factory=list, description="Associated syllabus topics.")

    @field_validator("question_text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question text cannot be empty.")
        return v.strip()


class MCQQuestion(QuestionBase):
    """Multiple Choice Question data model."""
    options: List[str] = Field(..., description="List of options, e.g., A, B, C, D.")
    correct_option: str = Field(..., description="The exact correct answer (must match one of the options).")

    @field_validator("options")
    @classmethod
    def validate_options_count(cls, v: List[str]) -> List[str]:
        if len(v) != 4:
            raise ValueError("An MCQ question must have exactly 4 options.")
        return [opt.strip() for opt in v]

    @field_validator("correct_option")
    @classmethod
    def validate_correct_option(cls, v: str, info) -> str:
        options = info.data.get("options")
        if options and v not in options:
            raise ValueError(f"Correct option '{v}' must be one of the provided options: {options}")
        return v


class TrueFalseQuestion(QuestionBase):
    """True or False Question data model."""
    correct_answer: bool = Field(..., description="Correct answer (True or False).")


class DescriptiveQuestion(QuestionBase):
    """Descriptive / Short Answer Question data model."""
    suggested_answer_key: str = Field(..., description="Points or model answer for grading.")
    word_count_guideline: int = Field(default=100, description="Suggested response length.")


class QuestionPair(BaseModel):
    """A pair consisting of the original question and its parallel generated variant."""
    original: MCQQuestion
    generated: MCQQuestion
    similarity_score: float = Field(..., description="Cosine similarity score of their embeddings.")


class QuestionBatch(BaseModel):
    """A batch of processed question pairs from a document."""
    batch_id: UUID = Field(default_factory=uuid4)
    pairs: List[QuestionPair]
    language: LanguageEnum
    total_pages_processed: int


class PDFJobStatus(BaseModel):
    """Representing status tracker for background processing."""
    job_id: str
    status: str  # 'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'
    pages_processed: int
    total_pages: int
    error_message: Optional[str] = None
    output_pdf_path: Optional[str] = None

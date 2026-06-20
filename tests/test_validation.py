"""
Unit tests for the Quality Validation Engine.
Runs the validation engine against both good questions and the adversarial violation generator.
"""
import sys
import os
import pytest

# Ensure flask_app is in Python PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../flask_app')))

from services.validation_service import QualityValidationEngine
from models.schemas import MCQQuestion, LanguageEnum, DifficultyEnum
from adversarial_gen import AdversarialGenerator

from pydantic import ValidationError

def test_valid_question_passes():
    """Verify that a standard, well-formed question passes validation with no errors."""
    q = MCQQuestion(
        question_text="Who was the leader of the Aruvippuram consecration in 1888?",
        options=["Sree Narayana Guru", "Chattampi Swamikal", "Ayyankali", "Sahodaran Ayyappan"],
        correct_option="Sree Narayana Guru",
        difficulty=DifficultyEnum.MEDIUM,
        language=LanguageEnum.ENGLISH,
        concept_tags=["Renaissance"]
    )
    
    errors = QualityValidationEngine.validate_mcq(q)
    assert len(errors) == 0, f"Expected no errors, but found: {errors}"

def test_semantic_ambiguity_fails():
    """Verify that invalid correct options fail at the schema model layer."""
    with pytest.raises(ValidationError):
        MCQQuestion(
            question_text="Which of the following is associated with social reform?",
            options=["Sree Narayana Guru", "Ayyankali", "Sahodaran Ayyappan", "Vagbhadananda"],
            correct_option="Mahatma Gandhi", # Not in options
            difficulty=DifficultyEnum.MEDIUM,
            language=LanguageEnum.ENGLISH
        )

def test_grammatical_violation_fails():
    """Verify that grammatical violations (especially Malayalam morpho-syntax) are flagged."""
    questions = AdversarialGenerator.get_grammatical_violations(intensity=2)
    # Finding the Malayalam question
    malayalam_q = [q for q in questions if q.language == LanguageEnum.MALAYALAM][0]
    
    errors = QualityValidationEngine.validate_mcq(malayalam_q)
    assert "ERR_GRAMMAR_VIOLATION" in errors

def test_duplicate_distractors_fails():
    """Verify that duplicate or too-similar distractors trigger warnings/errors."""
    # Test case 1: Substring containment (e.g. Thiruvananthapuram vs Thiruvananthapuram District)
    q1 = MCQQuestion(
        question_text="What is the capital of Kerala?",
        options=["Thiruvananthapuram", "Thiruvananthapuram District", "Kochi", "Kozhikode"],
        correct_option="Thiruvananthapuram",
        difficulty=DifficultyEnum.EASY,
        language=LanguageEnum.ENGLISH
    )
    errors1 = QualityValidationEngine.validate_mcq(q1)
    assert "ERR_DUPLICATE_DISTRACTORS" in errors1

    # Test case 2: Overlapping words (Jaccard similarity >= 0.50)
    q2 = MCQQuestion(
        question_text="Which renaissance leader formed the Sadhu Jana Paripalana Sangham?",
        options=["Ayyankali", "Leader Ayyankali", "Sree Narayana Guru", "Vagbhadananda"],
        correct_option="Ayyankali",
        difficulty=DifficultyEnum.EASY,
        language=LanguageEnum.ENGLISH
    )
    errors2 = QualityValidationEngine.validate_mcq(q2)
    assert "ERR_DUPLICATE_DISTRACTORS" in errors2

def test_bias_injection_fails():
    """Verify that gender-biased assumptions are caught by the validation rules."""
    questions = AdversarialGenerator.get_bias_injections(intensity=1)
    bias_q = questions[0]
    
    errors = QualityValidationEngine.validate_mcq(bias_q)
    assert "ERR_BIAS_DETECTED" in errors

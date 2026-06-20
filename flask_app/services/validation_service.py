"""
Validation service for question quality control.
Checks questions for factual, grammatical, semantic, and bias issues.
"""
import re
import logging
from typing import List
from models.schemas import MCQQuestion, LanguageEnum

logger = logging.getLogger(__name__)

class QualityValidationEngine:
    """Automates quality and pedagogical heuristics checks on generated MCQs."""
    
    @staticmethod
    def calculate_jaccard_similarity(str1: str, str2: str) -> float:
        """Helper to compute word-level Jaccard similarity between two strings."""
        w1 = set(str1.lower().strip().split())
        w2 = set(str2.lower().strip().split())
        intersection = w1.intersection(w2)
        union = w1.union(w2)
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def calculate_char_overlap(str1: str, str2: str) -> float:
        """Computes character 3-gram similarity for spelling variance detection."""
        s1 = re.sub(r"\s+", "", str1.lower().strip())
        s2 = re.sub(r"\s+", "", str2.lower().strip())
        if len(s1) < 3 or len(s2) < 3:
            return 1.0 if s1 == s2 else 0.0
        g1 = {s1[i:i+3] for i in range(len(s1) - 2)}
        g2 = {s2[i:i+3] for i in range(len(s2) - 2)}
        intersection = g1.intersection(g2)
        union = g1.union(g2)
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def validate_mcq(q: MCQQuestion) -> List[str]:
        """
        Runs validation checks on an MCQ and returns a list of error codes.
        Empty list indicates the question passed validation.
        """
        errors = []
        
        # Rule 1: Correct option must be one of the options
        if q.correct_option not in q.options:
            errors.append("ERR_AMBIGUOUS_ANSWER")
            
        # Rule 2: Check for duplicate/too similar distractors
        for i in range(len(q.options)):
            opt_i = q.options[i].lower().strip()
            for j in range(i + 1, len(q.options)):
                opt_j = q.options[j].lower().strip()
                
                # Check Jaccard word similarity
                word_sim = QualityValidationEngine.calculate_jaccard_similarity(opt_i, opt_j)
                # Check character level overlap
                char_sim = QualityValidationEngine.calculate_char_overlap(opt_i, opt_j)
                
                # Check containment (e.g., "Thiruvananthapuram" and "Thiruvananthapuram City")
                containment = False
                if len(opt_i) > 4 and len(opt_j) > 4:
                    if opt_i in opt_j or opt_j in opt_i:
                        containment = True
                
                if word_sim >= 0.50 or char_sim >= 0.60 or containment:
                    logger.warning(f"Flagged duplicate distractors: '{q.options[i]}' vs '{q.options[j]}' (word_sim={word_sim:.2f}, char_sim={char_sim:.2f}, containment={containment})")
                    errors.append("ERR_DUPLICATE_DISTRACTORS")
                    break
            if "ERR_DUPLICATE_DISTRACTORS" in errors:
                break
                
        # Rule 3: Check language encoding
        if q.language == LanguageEnum.MALAYALAM:
            # Enforce Malayalam Unicode characters
            has_malayalam = bool(re.search(r"[\u0d00-\u0d7f]", q.question_text))
            if not has_malayalam:
                errors.append("ERR_GRAMMAR_VIOLATION")
                
            # Check for common Malayalam morpho-syntax errors (like dative 'ആരെ വേണ്ടി')
            if "ആരെ വേണ്ടി" in q.question_text:
                errors.append("ERR_GRAMMAR_VIOLATION")
                
        # Rule 4: Bias Check (e.g. nurse/her doctor/his gender-role stereotypes)
        bias_keywords_english = [
            r"\bnurse\b.*\bher\b",
            r"\bdoctor\b.*\bhis\b",
            r"\bcollector\b.*\bhis wife\b"
        ]
        for pattern in bias_keywords_english:
            if re.search(pattern, q.question_text.lower()):
                errors.append("ERR_BIAS_DETECTED")
                break
                
        return errors

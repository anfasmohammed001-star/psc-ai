"""
Service layer to parse structured questions, check for duplicates, and validate option formats.
Provides Malayalam character check helper rules.
"""
import re
import logging
from typing import List, Tuple, Optional
import numpy as np

from models.schemas import MCQQuestion, LanguageEnum, DifficultyEnum
from services.gemini_service import GeminiClient

logger = logging.getLogger(__name__)

class DuplicateDetector:
    """Checks semantic similarity of question stems to prevent repeating concepts."""
    def __init__(self, gemini_client: GeminiClient, similarity_threshold: float = 0.85):
        self.client = gemini_client
        self.threshold = similarity_threshold

    def calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        a = np.array(vec1)
        b = np.array(vec2)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def is_duplicate(self, text_a: str, text_b: str) -> Tuple[bool, float]:
        """Compares two question stems using vector embeddings cosine similarity."""
        try:
            emb_a = self.client.get_embedding(text_a)
            emb_b = self.client.get_embedding(text_b)
            score = self.calculate_cosine_similarity(emb_a, emb_b)
            return (score >= self.threshold), score
        except Exception as e:
            logger.error(f"Failed to calculate similarity embedding: {str(e)}")
            # Fallback to Jaccard distance on words if embedding fails
            words_a = set(text_a.lower().split())
            words_b = set(text_b.lower().split())
            intersection = words_a.intersection(words_b)
            union = words_a.union(words_b)
            jaccard = len(intersection) / len(union) if union else 0.0
            return (jaccard >= 0.70), jaccard


class QuestionParser:
    """Regex parsing helper to format text outputs into Pydantic models."""
    
    @staticmethod
    def contains_malayalam(text: str) -> bool:
        """Heuristic check to verify Malayalam Unicode block characters (U+0D00 to U+0D7F)."""
        return bool(re.search(r"[\u0d00-\u0d7f]", text))

    @staticmethod
    def parse_extracted_mcqs(text: str, default_lang: LanguageEnum) -> List[MCQQuestion]:
        """
        Parses text matching the custom [QUESTION_START]...[QUESTION_END] format.
        Yields Pydantic validation structures.
        """
        pattern = re.compile(r"\[QUESTION_START\](.*?)\[QUESTION_END\]", re.DOTALL)
        blocks = pattern.findall(text)
        
        parsed_questions = []
        for block in blocks:
            try:
                # Extract question stem
                stem_match = re.search(r"\*\*Question:\*\*(.*)", block)
                if not stem_match:
                    continue
                # Split off options if they are on same line
                question_text = stem_match.group(1).split("*")[0].strip()
                
                # Extract options
                options = []
                for opt_char in ["A", "B", "C", "D"]:
                    opt_match = re.search(fr"\*\s*{opt_char}\)(.*)", block)
                    if opt_match:
                        options.append(opt_match.group(1).strip())
                        
                if len(options) != 4:
                    # Try alternate matches (without asterisk)
                    options = []
                    for opt_char in ["A", "B", "C", "D"]:
                        opt_match = re.search(fr"{opt_char}\)(.*)", block)
                        if opt_match:
                            options.append(opt_match.group(1).strip())
                
                if len(options) != 4:
                    logger.warning(f"Skipping malformed MCQ block (options count is {len(options)}): {block}")
                    continue
                
                # Validate language specific characters
                if default_lang == LanguageEnum.MALAYALAM:
                    # Original text or options must contain some Malayalam Unicode chars
                    full_block_text = question_text + "".join(options)
                    if not QuestionParser.contains_malayalam(full_block_text):
                        logger.warning(f"Malayalam language mode active but block contains no Malayalam script: {block}")
                        # We don't discard directly as English terms are common in Malayalam science/history, but raise warning
                
                # Deduce correct answer (from generated parallel block format, if available)
                ans_match = re.search(r"\* Correct Option:\s*([A-D])", block)
                correct_opt = ""
                if ans_match:
                    idx = ord(ans_match.group(1).strip()) - ord('A')
                    if 0 <= idx < 4:
                        correct_opt = options[idx]
                else:
                    # Default correct option placeholder for raw extracted exam papers
                    correct_opt = options[0]
                
                q = MCQQuestion(
                    question_text=question_text,
                    options=options,
                    correct_option=correct_opt,
                    difficulty=DifficultyEnum.MEDIUM,
                    language=default_lang,
                    concept_tags=[]
                )
                parsed_questions.append(q)
                
            except Exception as e:
                logger.warning(f"Error parsing single question block: {str(e)}")
                
        return parsed_questions

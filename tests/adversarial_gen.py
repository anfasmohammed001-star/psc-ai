"""
Intentional Quality Violation Generator.
Generates deliberately flawed questions across semantic, grammatical,
factual, distractor, and bias dimensions to stress-test the Validation Engine.
"""
from typing import List, Dict, Any
from models.schemas import MCQQuestion, LanguageEnum, DifficultyEnum

class AdversarialGenerator:
    """Produces flawed questions with configurable intensity levels."""
    
    @staticmethod
    def get_semantic_ambiguities(intensity: int = 1) -> List[MCQQuestion]:
        """Category 1: Semantic Ambiguity (Multiple valid interpretations)"""
        questions = []
        
        if intensity >= 1:
            # Level 1: Subtle ambiguity
            questions.append(MCQQuestion(
                question_text="Which of the following is associated with the social reforms in Kerala?",
                options=["Sree Narayana Guru", "Vagbhadananda", "Chattampi Swamikal", "Ayyankali"],
                correct_option="Sree Narayana Guru",  # But all are highly associated Renaissance leaders
                difficulty=DifficultyEnum.MEDIUM,
                language=LanguageEnum.ENGLISH,
                concept_tags=["Renaissance"]
            ))
            
        if intensity >= 2:
            # Level 2: Moderate ambiguity
            questions.append(MCQQuestion(
                question_text="കേരളത്തിലെ നവോത്ഥാന നായകൻ ആര്?",
                options=["ശ്രീനാരായണഗുരു", "അയ്യങ്കാളി", "ചട്ടമ്പിസ്വാമികൾ", "കുമാരനാശാൻ"],
                correct_option="ശ്രീനാരായണഗുരു",  # All four are famous Renaissance leaders
                difficulty=DifficultyEnum.MEDIUM,
                language=LanguageEnum.MALAYALAM,
                concept_tags=["Renaissance"]
            ))
            
        if intensity >= 3:
            # Level 3: Egregious ambiguity
            questions.append(MCQQuestion(
                question_text="Choose the correct statement about agriculture in India.",
                options=["It is important", "It uses water", "It employs people", "It happens in fields"],
                correct_option="It employs people",  # All options are generically true and correct
                difficulty=DifficultyEnum.EASY,
                language=LanguageEnum.ENGLISH,
                concept_tags=["Agriculture"]
            ))
            
        return questions

    @staticmethod
    def get_grammatical_violations(intensity: int = 1) -> List[MCQQuestion]:
        """Category 2: Grammatical Violations (Malayalam morpho-syntax & English syntax)"""
        questions = []
        
        if intensity >= 1:
            # Level 1: English subject-verb disagreement
            questions.append(MCQQuestion(
                question_text="The members of the constituent assembly has proposed the draft on which date?",
                options=["August 29, 1947", "November 26, 1949", "January 26, 1950", "August 15, 1947"],
                correct_option="August 29, 1947",
                difficulty=DifficultyEnum.MEDIUM,
                language=LanguageEnum.ENGLISH,
                concept_tags=["Constitution"]
            ))
            
        if intensity >= 2:
            # Level 2: Malayalam Case Marker Mismatch
            # Uses 'ആരെ വേണ്ടി' (accusative + postposition) instead of 'ആർക്ക് വേണ്ടി' (dative)
            questions.append(MCQQuestion(
                question_text="ശ്രീനാരായണഗുരു അരുവിപ്പുറം പ്രതിഷ്ഠ നടത്തിയത് ആരെ വേണ്ടി ആണ്?",
                options=["ഈഴവർക്ക് വേണ്ടി", "മനുഷ്യരാശിയുടെ സമാധാനത്തിന് വേണ്ടി", "അടിച്ചമർത്തപ്പെട്ട ജനങ്ങൾക്ക് വേണ്ടി", "ക്ഷേത്രങ്ങൾക്ക് വേണ്ടി"],
                correct_option="അടിച്ചമർത്തപ്പെട്ട ജനങ്ങൾക്ക് വേണ്ടി",
                difficulty=DifficultyEnum.MEDIUM,
                language=LanguageEnum.MALAYALAM,
                concept_tags=["Renaissance"]
            ))
            
        if intensity >= 3:
            # Level 3: Structural grammatical failure
            questions.append(MCQQuestion(
                question_text="Where does the river Bharathapuzha originates and where do it flows into?",
                options=["Anamalai hills, Arabian Sea", "Western Ghats, Bay of Bengal", "Nilgiri hills, Indian Ocean", "Deccan plateau, Laccadive Sea"],
                correct_option="Anamalai hills, Arabian Sea",
                difficulty=DifficultyEnum.MEDIUM,
                language=LanguageEnum.ENGLISH,
                concept_tags=["Geography"]
            ))
            
        return questions

    @staticmethod
    def get_factual_inconsistencies(intensity: int = 1) -> List[MCQQuestion]:
        """Category 3: Factual Inconsistency (Wrong keys, multiple valid keys, or better distractors)"""
        questions = []
        
        if intensity >= 1:
            # Level 1: Marked answer contradicts basic geography
            questions.append(MCQQuestion(
                question_text="Which is the longest river in Kerala?",
                options=["Periyar", "Bharathapuzha", "Pamba", "Chaliyar"],
                correct_option="Bharathapuzha",  # Factual error: Periyar is the longest (244km vs 209km)
                difficulty=DifficultyEnum.EASY,
                language=LanguageEnum.ENGLISH,
                concept_tags=["Geography"]
            ))
            
        if intensity >= 2:
            # Level 2: Mathematical incorrect key
            questions.append(MCQQuestion(
                question_text="If a train runs at 72 km/h, what is its speed in meters per second?",
                options=["20 m/s", "15 m/s", "25 m/s", "30 m/s"],
                correct_option="25 m/s",  # Factual error: 72 * (5/18) = 20 m/s, not 25
                difficulty=DifficultyEnum.EASY,
                language=LanguageEnum.ENGLISH,
                concept_tags=["Math"]
            ))
            
        return questions

    @staticmethod
    def get_implausible_distractors(intensity: int = 1) -> List[MCQQuestion]:
        """Category 4: Implausibly Similar Distractors (Synonyms or trivially identical options)"""
        questions = []
        
        if intensity >= 1:
            # Level 1: Semantic synonyms in English
            questions.append(MCQQuestion(
                question_text="What is the capital of Kerala?",
                options=["Thiruvananthapuram", "Trivandrum", "Capital of Kerala", "Kochi"],
                correct_option="Thiruvananthapuram",  # Distractors 1 & 2 are synonyms
                difficulty=DifficultyEnum.EASY,
                language=LanguageEnum.ENGLISH,
                concept_tags=["Geography"]
            ))
            
        if intensity >= 2:
            # Level 2: Synonyms in Malayalam
            questions.append(MCQQuestion(
                question_text="കേരളത്തിന്റെ തലസ്ഥാനം ഏത്?",
                options=["തിരുവനന്തപുരം", "ത്രിവേന്ദ്രം", "അനന്തപുരി", "കൊച്ചി"],
                correct_option="തിരുവനന്തപുരം",  # ത്രിവേന്ദ്രം and അനന്തപുരി are historical/synonymous names
                difficulty=DifficultyEnum.EASY,
                language=LanguageEnum.MALAYALAM,
                concept_tags=["Geography"]
            ))
            
        return questions

    @staticmethod
    def get_bias_injections(intensity: int = 1) -> List[MCQQuestion]:
        """Category 5: Bias Injections (Gender/Socioeconomic assumptions)"""
        questions = []
        
        if intensity >= 1:
            # Level 1: Gender role assumption
            questions.append(MCQQuestion(
                question_text="A nurse works long shifts to help her doctor. Who coordinates the ward?",
                options=["Head nurse", "Ward boy", "Security", "Receptionist"],
                correct_option="Head nurse",
                difficulty=DifficultyEnum.EASY,
                language=LanguageEnum.ENGLISH,
                concept_tags=["Administration"]
            ))
            
        return questions

    @classmethod
    def generate_all_violations(cls, intensity: int = 2) -> List[Dict[str, Any]]:
        """Compiles a list of labeled problematic test cases for regression testing."""
        all_cases = []
        
        for q in cls.get_semantic_ambiguities(intensity):
            all_cases.append({"category": "semantic_ambiguity", "question": q})
            
        for q in cls.get_grammatical_violations(intensity):
            all_cases.append({"category": "grammatical_violation", "question": q})
            
        for q in cls.get_factual_inconsistencies(intensity):
            all_cases.append({"category": "factual_inconsistency", "question": q})
            
        for q in cls.get_implausible_distractors(intensity):
            all_cases.append({"category": "implausible_distractors", "question": q})
            
        for q in cls.get_bias_injections(intensity):
            all_cases.append({"category": "bias_injection", "question": q})
            
        return all_cases

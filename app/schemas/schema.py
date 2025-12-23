from pydantic import BaseModel, Field

class CandidateData(BaseModel):
    """
    Schema for validating incoming data for inference.
    """

    english_proficiency: float = Field(..., ge = 0, le = 10, description = "English exam score (0-10)")
    cognitive_ability: int = Field(..., ge = 0, le = 200, description = "IQ score (0-200)")
    aptitude_test_result: int = Field(..., ge = 0, le = 100, description = "Psychometric score (0-100)")
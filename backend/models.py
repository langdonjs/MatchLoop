from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class PreferenceState(BaseModel):
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    rank: int
    title: str
    company: str
    url: str
    location: str
    salary: Optional[str]
    yc_batch: Optional[str]
    score: float
    rationale: str
    rationale_bullets: list[str] = Field(default_factory=list)
    """Flat list for backward compat: general + preferences + watchouts in order."""

    rationale_general: list[str] = Field(default_factory=list)
    """Profile / experience overlap — why this role fits you broadly."""

    rationale_preferences: list[str] = Field(default_factory=list)
    """Tie-ins to stated must-have / nice-to-have / feedback themes."""

    rationale_watchouts: list[str] = Field(default_factory=list)
    """Mismatches, tradeoffs, avoid-list tension, geography/comp risks."""

    match_factors: list[str]


class SessionResponse(BaseModel):
    session_id: str
    round: int
    recommendations: list[Recommendation]
    preference_state: PreferenceState
    confidence_message: Optional[str]
    candidate_name: str


class CreateSessionRequest(BaseModel):
    """Exactly one source: numbered preset (`candidate_index`) OR full `candidate` object."""

    candidate_index: Optional[int] = None
    candidate: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def require_candidate_source(self) -> "CreateSessionRequest":
        if self.candidate is None and self.candidate_index is None:
            raise ValueError(
                "Provide candidate_index (preset dropdown) or candidate (custom JSON object)"
            )
        return self


class FeedbackRequest(BaseModel):
    feedback: str


class FeedbackResponse(BaseModel):
    session_id: str
    round: int
    recommendations: list[Recommendation]
    preference_state: PreferenceState
    confidence_message: Optional[str]
    feedback_understood: bool
    clarification_needed: Optional[str]

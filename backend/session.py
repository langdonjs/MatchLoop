import uuid
from dataclasses import dataclass, field

from backend.models import PreferenceState, Recommendation


@dataclass
class SessionState:
    session_id: str
    candidate: dict
    candidate_name: str
    retrieval_summary: str
    ranking_context: str
    preference_state: PreferenceState
    seen_urls: set[str] = field(default_factory=set)
    round_number: int = 0
    round_1_top_score: float = 0.0
    history: list[dict] = field(default_factory=list)


_sessions: dict[str, SessionState] = {}


def create_session(
    candidate: dict,
    retrieval_summary: str,
    ranking_context: str,
) -> SessionState:
    session_id = str(uuid.uuid4())
    name = candidate.get("name") or "Candidate"
    state = SessionState(
        session_id=session_id,
        candidate=candidate,
        candidate_name=name,
        retrieval_summary=retrieval_summary,
        ranking_context=ranking_context,
        preference_state=PreferenceState(),
    )
    _sessions[session_id] = state
    return state


def get_session(session_id: str) -> SessionState:
    if session_id not in _sessions:
        raise KeyError(f"Session {session_id} not found")
    return _sessions[session_id]

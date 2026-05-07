import json
import logging
import os
import secrets
from base64 import b64decode
from pathlib import Path

# Skip optional TensorFlow in Hugging Face stack when local env has Keras 3 / no tf-keras.
os.environ.setdefault("USE_TF", "0")

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.config import retrieve_k
from backend.location_display import shorten_job_location
from backend.feedback import parse_feedback
from backend.models import (
    CreateSessionRequest,
    FeedbackRequest,
    FeedbackResponse,
    Recommendation,
    SessionResponse,
)
from backend.ranking import finalize_ranking_response, stream_rank_jobs
from backend.retrieval import JobRetriever
from backend.session import create_session, get_session
from backend.summarizer import build_ranking_context, build_retrieval_summary

log = logging.getLogger(__name__)

app = FastAPI(title="MatchLoop API")
DEMO_USERNAME = os.getenv("DEMO_USERNAME", "friend")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "CoffeeSpace")

DATA_DIR = Path("data")
with open(DATA_DIR / "jobs.json", encoding="utf-8") as f:
    JOBS = json.load(f)
for _job in JOBS:
    if isinstance(_job, dict) and "location" in _job:
        _job["location"] = shorten_job_location(_job.get("location"))
with open(DATA_DIR / "candidates.json", encoding="utf-8") as f:
    CANDIDATES = json.load(f)

retriever = JobRetriever(JOBS)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


def _auth_challenge() -> PlainTextResponse:
    return PlainTextResponse(
        "Authentication required",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="MatchLoop Demo"'},
    )


def _basic_auth_ok(auth_header: str | None) -> bool:
    if not auth_header:
        return False
    if not auth_header.startswith("Basic "):
        return False
    token = auth_header.split(" ", 1)[1].strip()
    try:
        decoded = b64decode(token).decode("utf-8")
    except Exception:
        return False
    if ":" not in decoded:
        return False
    user, pwd = decoded.split(":", 1)
    return secrets.compare_digest(user, DEMO_USERNAME) and secrets.compare_digest(
        pwd, DEMO_PASSWORD
    )


@app.middleware("http")
async def require_demo_password(request: Request, call_next):
    # Protect every route (UI + API) with one credential pair.
    if not _basic_auth_ok(request.headers.get("authorization")):
        return _auth_challenge()
    return await call_next(request)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _stream_error_payload(exc: BaseException) -> str:
    """
    SSE cannot return HTTP JSON errors mid-body; uncaught exceptions close the TCP
    socket and browsers report 'Network Error'. Always yield this instead.
    """
    msg = str(exc).strip() or type(exc).__name__
    if len(msg) > 520:
        msg = msg[:517] + "..."
    log.exception("Stream failed")
    return _sse({"type": "error", "message": msg})


def _check_confidence(recommendations: list[Recommendation]) -> str | None:
    if recommendations and all(r.score < 0.5 for r in recommendations):
        return (
            "You've seen the strongest matches for your profile in this job pool. "
            "The available roles are becoming limited — consider broadening your preferences, "
            "or these may be the best fits given current openings."
        )
    return None


def _build_retrieval_query_with_prefs(
    base_summary: str, prefs
) -> str:
    parts = [base_summary]
    if prefs.must_have:
        parts.append("Must have: " + " ".join(prefs.must_have))
    if prefs.nice_to_have:
        parts.append("Prefers: " + " ".join(prefs.nice_to_have))
    return " ".join(parts)


@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")


def _candidate_dropdown_subtitle(candidate: dict) -> str:
    """Warren etc. omit `title` in source data — use headline so the picker isn't blank."""
    t = (candidate.get("title") or "").strip()
    if t:
        return t
    headline = (candidate.get("headline") or "").strip()
    if headline:
        return headline if len(headline) <= 80 else headline[:77] + "…"
    return "Profile"


@app.get("/candidates")
def list_candidates():
    return [
        {
            "index": i,
            "name": candidate.get("name") or "Candidate",
            "title": _candidate_dropdown_subtitle(candidate),
        }
        for i, candidate in enumerate(CANDIDATES)
    ]


@app.post("/sessions")
def create_new_session(request: CreateSessionRequest):
    def event_stream():
        try:
            candidate: dict
            if request.candidate is not None:
                if not isinstance(request.candidate, dict):
                    yield _sse({"type": "error", "message": "candidate must be a JSON object"})
                    return
                candidate = request.candidate
            else:
                idx = request.candidate_index
                if idx is None or idx < 0 or idx >= len(CANDIDATES):
                    yield _sse({"type": "error", "message": "Invalid candidate index"})
                    return
                candidate = CANDIDATES[idx]
            retrieval_summary = build_retrieval_summary(candidate)
            ranking_context = build_ranking_context(candidate)
            session = create_session(candidate, retrieval_summary, ranking_context)

            jobs = retriever.retrieve(
                retrieval_summary,
                session.preference_state,
                session.seen_urls,
                k=retrieve_k(),
            )

            full_response = ""
            for chunk in stream_rank_jobs(
                ranking_context,
                jobs,
                session.preference_state,
                round_number=1,
            ):
                full_response += chunk
                yield _sse({"type": "ranking_chunk", "text": chunk})

            recommendations = finalize_ranking_response(
                full_response,
                ranking_context,
                jobs,
                session.preference_state,
                1,
            )

            session.round_number = 1
            for rec in recommendations:
                session.seen_urls.add(rec.url)
            session.round_1_top_score = (
                recommendations[0].score if recommendations else 0.0
            )
            session.history.append(
                {
                    "round": 1,
                    "recommendations": [r.model_dump() for r in recommendations],
                }
            )

            confidence_message = _check_confidence(recommendations)
            payload = SessionResponse(
                session_id=session.session_id,
                round=session.round_number,
                recommendations=recommendations,
                preference_state=session.preference_state,
                confidence_message=confidence_message,
                candidate_name=candidate.get("name") or "Candidate",
            )
            yield _sse({"type": "session_complete", "payload": payload.model_dump()})
        except Exception as e:
            yield _stream_error_payload(e)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/sessions/{session_id}/feedback")
def process_feedback(session_id: str, request: FeedbackRequest):
    def event_stream():
        try:
            try:
                session = get_session(session_id)
            except KeyError:
                yield _sse({"type": "error", "message": "Session not found"})
                return

            prev_recs = session.history[-1]["recommendations"] if session.history else []

            updated_prefs, feedback_useful, clarification = parse_feedback(
                request.feedback,
                session.preference_state,
                prev_recs,
            )

            if not feedback_useful:
                rec_objs = [Recommendation(**r) for r in prev_recs]
                payload = FeedbackResponse(
                    session_id=session_id,
                    round=session.round_number,
                    recommendations=rec_objs,
                    preference_state=session.preference_state,
                    confidence_message=None,
                    feedback_understood=False,
                    clarification_needed=clarification,
                )
                yield _sse({"type": "feedback_complete", "payload": payload.model_dump()})
                return

            # Do not mutate round / prefs until ranking succeeds (avoids corrupt state on API errors).
            next_round = session.round_number + 1
            updated_retrieval_query = _build_retrieval_query_with_prefs(
                session.retrieval_summary,
                updated_prefs,
            )

            jobs = retriever.retrieve(
                updated_retrieval_query,
                updated_prefs,
                session.seen_urls,
                k=retrieve_k(),
            )

            full_response = ""
            for chunk in stream_rank_jobs(
                session.ranking_context,
                jobs,
                updated_prefs,
                round_number=next_round,
            ):
                full_response += chunk
                yield _sse({"type": "ranking_chunk", "text": chunk})

            recommendations = finalize_ranking_response(
                full_response,
                session.ranking_context,
                jobs,
                updated_prefs,
                next_round,
            )

            session.preference_state = updated_prefs
            session.round_number = next_round
            for rec in recommendations:
                session.seen_urls.add(rec.url)
            session.history.append(
                {
                    "round": session.round_number,
                    "feedback": request.feedback,
                    "recommendations": [r.model_dump() for r in recommendations],
                }
            )

            confidence_message = _check_confidence(recommendations)
            payload = FeedbackResponse(
                session_id=session_id,
                round=session.round_number,
                recommendations=recommendations,
                preference_state=updated_prefs,
                confidence_message=confidence_message,
                feedback_understood=True,
                clarification_needed=None,
            )
            yield _sse({"type": "feedback_complete", "payload": payload.model_dump()})
        except Exception as e:
            yield _stream_error_payload(e)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/sessions/{session_id}/state")
def get_session_state(session_id: str):
    try:
        session = get_session(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found") from None
    return {
        "session_id": session_id,
        "round": session.round_number,
        "preference_state": session.preference_state.model_dump(),
        "seen_urls_count": len(session.seen_urls),
        "history_rounds": len(session.history),
    }

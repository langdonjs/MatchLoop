from __future__ import annotations

import json
import re
from collections.abc import Iterator

import anthropic

from backend.config import rank_description_max_chars
from backend.models import PreferenceState, Recommendation

client = anthropic.Anthropic()


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("No valid JSON object found", text, 0)


def _format_jobs_for_prompt(jobs: list[dict]) -> str:
    parts: list[str] = []
    seen_company_intros: dict[str, bool] = {}

    for i, job in enumerate(jobs, 1):
        desc = job.get("description", "")
        company = job.get("company", "")

        if company in seen_company_intros:
            role_markers = [
                "about the role",
                "position summary",
                "the opportunity",
                "what you will do",
                "responsibilities",
                "your role",
            ]
            for marker in role_markers:
                idx = desc.lower().find(marker)
                if idx > 0:
                    desc = desc[idx:]
                    break
        else:
            seen_company_intros[company] = True

        cap = rank_description_max_chars()
        if cap is not None and len(desc) > cap:
            desc = (
                desc[:cap]
                + "\n\n[Truncated — set MATCHLOOP_RANK_DESC_MAX_CHARS=0 for full job text in ranking.]"
            )

        salary = f" | Salary: {job['salary']}" if job.get("salary") else ""
        equity = f" | Equity: {job['equity']}" if job.get("equity") else ""
        batch = f" | YC {job['yc_batch']}" if job.get("yc_batch") else ""

        parts.append(
            f"""
JOB {i}:
Title: {job['title']}
Company: {job.get('company', 'Unknown')}
URL: {job['url']}
Location: {job.get('location', 'Unknown')}{batch}{salary}{equity}
Job Type: {job.get('job_type', 'Unknown')}
Description:
{desc}
---"""
        )

    return "\n".join(parts)


def _norm_bullet_lines(raw: list | None, cap: int = 6) -> list[str]:
    if not isinstance(raw, list):
        return []
    out = [re.sub(r"\s+", " ", str(b).strip()) for b in raw if str(b).strip()]
    return out[:cap]


def _coerce_rationale_bullets(item: dict) -> list[str]:
    """Legacy flat rationale_bullets or paragraph → list."""
    raw = item.get("rationale_bullets")
    if isinstance(raw, list) and len(raw) > 0:
        return _norm_bullet_lines(raw, 12)
    text = str(item.get("rationale") or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 8][:8]


def _extract_three_rationale_buckets(item: dict) -> tuple[list[str], list[str], list[str]]:
    """
    New shape: rationale_general, rationale_preferences, rationale_watchouts.
    Legacy: flat rationale_bullets → treat as general only.
    """
    general = _norm_bullet_lines(item.get("rationale_general"), 5)
    preferences = _norm_bullet_lines(item.get("rationale_preferences"), 5)
    watchouts = _norm_bullet_lines(item.get("rationale_watchouts"), 4)

    if general or preferences or watchouts:
        return general, preferences, watchouts

    legacy_flat = _coerce_rationale_bullets(item)
    return legacy_flat, [], []


def _format_preferences(prefs: PreferenceState, round_number: int) -> str:
    del round_number
    if not prefs.must_have and not prefs.nice_to_have and not prefs.avoid:
        return ""

    lines = ["CANDIDATE PREFERENCES (learned from feedback):"]
    if prefs.must_have:
        lines.append(f"Must have: {', '.join(prefs.must_have)}")
    if prefs.nice_to_have:
        lines.append(f"Nice to have: {', '.join(prefs.nice_to_have)}")
    if prefs.avoid:
        lines.append(f"Strongly avoid: {', '.join(prefs.avoid)}")
    return "\n".join(lines)


def _build_ranking_prompt(
    ranking_context: str,
    jobs: list[dict],
    preference_state: PreferenceState,
    round_number: int,
) -> str:
    jobs_text = _format_jobs_for_prompt(jobs)
    preference_section = _format_preferences(preference_state, round_number)

    return f"""You are an expert technical recruiter. Your job is to find the 3 best matching jobs for this candidate from the list provided.

CANDIDATE PROFILE:
{ranking_context}

{preference_section}

SCORING RUBRIC (you MUST follow this exactly):
Score each job 0.0 to 1.0 using these definitions:
- 0.9-1.0: Near-perfect match. Skills align directly, company stage fits, all stated preferences met.
- 0.7-0.89: Strong match. Core skills align, minor gaps or preference misses.
- 0.5-0.69: Moderate match. Relevant background but meaningful gaps.
- Below 0.5: Weak match. Significant skill or preference mismatch.

PREFERENCE GROUNDING REQUIREMENT (critical):
- Explicitly ground each rationale in the candidate's preference lists when available.
- Mention concrete satisfied (or conflicting) preference signals in plain language, e.g.:
  "This role is remote, in an early-stage team, and aligns with your backend focus."
- Use the FULL candidate dossier above (every current AND past employer block): tie specific shipped work,
  tools, stacks, domains, and company-stage experience to phrases in the JOB description (not vague "strong engineer").
- If preferences mention salary/ compensation (e.g. target $200k+), cite the JOB's listed salary/comp range when judging fit.
  If salary is unknown in the posting, say so briefly instead of pretending.
- If a must-have is not met, lower score accordingly and explain the miss.
- If an avoid item appears in a job, strongly penalize score and avoid recommending unless all alternatives are worse.

AVOID in your recommendations (deprioritize these strongly, do not recommend if better options exist):
{", ".join(preference_state.avoid) if preference_state.avoid else "Nothing to avoid yet"}

JOBS TO RANK:
{jobs_text}

Return ONLY valid JSON with no other text, preamble, or explanation. Format:
{{
  "recommendations": [
    {{
      "rank": 1,
      "url": "exact url from job listing",
      "score": 0.87,
      "rationale_general": ["...", "...", "..."],
      "rationale_preferences": ["...", "..."],
      "rationale_watchouts": ["...", "..."],
      "match_factors": ["remote", "backend", "early-stage", "python"]
    }},
    {{
      "rank": 2,
      "url": "...",
      "score": 0.8,
      "rationale_general": ["...", "...", "..."],
      "rationale_preferences": ["..."],
      "rationale_watchouts": ["..."],
      "match_factors": ["tag"]
    }},
    {{
      "rank": 3,
      "url": "...",
      "score": 0.75,
      "rationale_general": ["...", "...", "..."],
      "rationale_preferences": ["..."],
      "rationale_watchouts": ["..."],
      "match_factors": ["tag"]
    }}
  ]
}}

RATIONALE — THREE BLOCKS ONLY (shown as separate headings in the product):

1) rationale_general — 2–4 bullets: **general fit**.
   Tie specific employers/projects/skills FROM THE CANDIDATE PROFILE to responsibilities or tech FROM THIS JOB (posting wording).
   This block must NOT dwell on user's feedback preferences — only profile ↔ role overlap.
   Max ~22 words per bullet.

2) rationale_preferences — 1–4 bullets: **directly addressing learned preferences / feedback**.
   Must-have / nice-to-have / avoid implications from the preference lists above count as feedback.
   - If preference lists are empty: ONE bullet stating fit is purely profile-driven this round ("No structured preferences logged yet").
   - If non-empty: for each preference theme honestly supported, write one bullet formatted:
     `ASK: <Short preference echo> — <evidence from posting or clarification>`.
   - If something in must-have FAILS partially, note it briefly here WITH the same ASK prefix (honesty).
   Forbidden generic praise here — every bullet should name a preference theme or cite "none yet".

3) rationale_watchouts — 1–3 bullets: **why it might not work / contradictions**.
   Geography vs stated location prefs, compensation vs target band, scope mismatch (e.g. too frontend-heavy vs backend ask),
   company stage, avoid-list tensions, timezone, visa hints from posting unless unknown.
   If no material risk, ONE bullet acknowledging tradeoffs honestly (e.g. "Smaller corpus than FAANG-scale systems you shipped at.").

Across all bullets: forbid vague-only lines ("strong engineer", "great fit"); always anchor to dossier + posting facts.

Rules:
- Return exactly 3 recommendations
- match_factors should be 3–6 short keyword strings explaining why this job was chosen (mix of stack + situational tags)
- Scores must reflect the rubric above — do not inflate scores
- url must be copied exactly from the job listing"""


def _build_recommendations(
    raw_recs: list[dict], jobs: list[dict]
) -> list[Recommendation]:
    by_url = {j["url"]: j for j in jobs}
    out: list[Recommendation] = []
    for item in sorted(raw_recs, key=lambda x: x.get("rank", 0)):
        url = item.get("url")
        if not url or url not in by_url:
            continue
        job = by_url[url]
        general, preferences, watchouts = _extract_three_rationale_buckets(item)
        rationale_bullets = [*general, *preferences, *watchouts]
        if not rationale_bullets:
            rationale_bullets = _coerce_rationale_bullets(item)
            general = rationale_bullets
            preferences, watchouts = [], []
        rationale_plain = " ".join(rationale_bullets) or str(
            item.get("rationale") or ""
        ).strip()

        out.append(
            Recommendation(
                rank=int(item.get("rank", len(out) + 1)),
                title=job["title"],
                company=job.get("company") or "Unknown",
                url=url,
                location=job.get("location") or "",
                salary=job.get("salary"),
                yc_batch=job.get("yc_batch"),
                score=float(item.get("score", 0.0)),
                rationale=rationale_plain,
                rationale_bullets=rationale_bullets,
                rationale_general=general,
                rationale_preferences=preferences,
                rationale_watchouts=watchouts,
                match_factors=list(item.get("match_factors") or []),
            )
        )
        if len(out) >= 3:
            break
    while len(out) < 3 and jobs:
        fallback = jobs[len(out)]
        if any(r.url == fallback["url"] for r in out):
            break
        fb = "Included as fallback — model output was incomplete; verify fit manually."
        out.append(
            Recommendation(
                rank=len(out) + 1,
                title=fallback["title"],
                company=fallback.get("company") or "Unknown",
                url=fallback["url"],
                location=fallback.get("location") or "",
                salary=fallback.get("salary"),
                yc_batch=fallback.get("yc_batch"),
                score=0.4,
                rationale="Included as fallback due to incomplete model output.",
                rationale_bullets=[fb],
                rationale_general=[fb],
                rationale_preferences=[],
                rationale_watchouts=[
                    "Double-check posting for location, compensation, and role scope."
                ],
                match_factors=["fallback"],
            )
        )
    return out[:3]


def _retry_ranking(base_prompt: str) -> dict:
    strict = (
        base_prompt
        + "\n\nIMPORTANT: Output a single JSON object only. No markdown fences. No commentary."
    )
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3500,
        messages=[{"role": "user", "content": strict}],
    )
    text = response.content[0].text
    return _extract_json_object(text)


def stream_rank_jobs(
    ranking_context: str,
    jobs: list[dict],
    preference_state: PreferenceState,
    round_number: int,
) -> Iterator[str]:
    prompt = _build_ranking_prompt(
        ranking_context, jobs, preference_state, round_number
    )
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=3500,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def finalize_ranking_response(
    full_response: str,
    ranking_context: str,
    jobs: list[dict],
    preference_state: PreferenceState,
    round_number: int,
) -> list[Recommendation]:
    try:
        data = _extract_json_object(full_response)
    except json.JSONDecodeError:
        data = _retry_ranking(
            _build_ranking_prompt(
                ranking_context, jobs, preference_state, round_number
            )
        )
    raw = data.get("recommendations") or []
    return _build_recommendations(raw, jobs)


def rank_jobs(
    ranking_context: str,
    jobs: list[dict],
    preference_state: PreferenceState,
    round_number: int,
) -> list[Recommendation]:
    full_response = ""
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=3500,
        messages=[
            {
                "role": "user",
                "content": _build_ranking_prompt(
                    ranking_context, jobs, preference_state, round_number
                ),
            }
        ],
    ) as stream:
        for text in stream.text_stream:
            full_response += text

    return finalize_ranking_response(
        full_response,
        ranking_context,
        jobs,
        preference_state,
        round_number,
    )

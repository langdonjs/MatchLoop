import json
import re

import anthropic

from backend.models import PreferenceState

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


def _looks_like_pure_acknowledgment(text: str) -> bool:
    t = text.strip().lower()
    if len(t) <= 2:
        return True
    return bool(
        re.match(
            r"^(ok|k+|yeah|yep|yes|yup|thanks?|thx|cool|nice|great|perfect|fire|"
            r"looks good|sounds good|good job|love it|nice one|lol|haha|"
            r"got it|sounds great|appreciate it)\.?[!?…]*$",
            t,
        )
    )


def _likely_has_preference_intent(text: str) -> bool:
    """Casual / short preference phrases should still count (remote, team size, etc.)."""
    if _looks_like_pure_acknowledgment(text):
        return False
    tl = text.lower()
    if len(text.strip()) < 6:
        return False
    needles = (
        "remote",
        "hybrid",
        "office",
        "onsite",
        "wfh",
        "team",
        "people",
        "employee",
        "headcount",
        "small",
        "large",
        "10",
        "less than",
        "only",
        "want",
        "need",
        "prefer",
        "rather",
        "avoid",
        "don't",
        "dont",
        "not ",
        "no ",
        "startup",
        "enterprise",
        "early",
        "stage",
        "backend",
        "frontend",
        "full",
        "stack",
        "ml",
        "machine learning",
        "salary",
        "equity",
        "location",
    )
    return any(n in tl for n in needles)


def _retry_actionable_parse(
    feedback: str,
    current_prefs: PreferenceState,
    previous_recommendations: list[dict],
) -> dict | None:
    """Second pass when the model wrongly sets is_actionable false."""
    recs_summary = "\n".join(
        [
            f"- {r['title']} at {r['company']} (score: {r['score']})"
            for r in previous_recommendations
        ]
    )
    retry = f"""The candidate was shown these jobs:
{recs_summary}

They said: "{feedback}"

This message almost certainly states at least ONE job preference (work style, team size, role, company type, etc.).
Set is_actionable to true and populate the delta lists. Do NOT mark non-actionable unless the text is ONLY
praise/thanks with zero preference (e.g. "thanks", "looks great" with nothing else).

Map common phrases:
- remote / WFH / work from home → add_must includes "remote" unless they want office
- small team / under 10 / fewer than 10 people → add_must includes "small team (under 10 people)"
- early stage / startup → add_must or add_nice accordingly

Current prefs: must {current_prefs.must_have}, nice {current_prefs.nice_to_have}, avoid {current_prefs.avoid}

Return the SAME JSON schema:
{{
  "add_must": [],
  "add_nice": [],
  "add_avoid": [],
  "remove_must": [],
  "remove_nice": [],
  "remove_avoid": [],
  "is_actionable": true
}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": retry}],
    )
    try:
        return _extract_json_object(response.content[0].text)
    except json.JSONDecodeError:
        return None


def parse_feedback(
    feedback: str,
    current_prefs: PreferenceState,
    previous_recommendations: list[dict],
) -> tuple[PreferenceState, bool, str | None]:
    recs_summary = "\n".join(
        [
            f"- {r['title']} at {r['company']} (score: {r['score']})"
            for r in previous_recommendations
        ]
    )

    prompt = f"""You are parsing job search feedback to extract preference signals.

Current preferences:
- Must have: {current_prefs.must_have}
- Nice to have: {current_prefs.nice_to_have}
- Avoid: {current_prefs.avoid}

The candidate just saw these job recommendations:
{recs_summary}

The candidate said: "{feedback}"

Extract what this feedback tells us about their preferences. Return ONLY valid JSON:
{{
  "add_must": ["list of things that are now hard requirements"],
  "add_nice": ["list of soft preferences to add"],
  "add_avoid": ["list of things to deprioritize"],
  "remove_must": ["list of existing must-have items to remove — newer signal wins"],
  "remove_nice": ["list of existing nice-to-have items to remove"],
  "remove_avoid": ["list of existing avoid items to remove"],
  "is_actionable": true
}}

is_actionable rules (important):
- Set is_actionable=true whenever the user expresses ANY concrete job preference, even informally:
  remote/hybrid/office, team size, company stage, role type (backend/ML/etc.), industry, comp, location, things to avoid.
  Short natural sentences are valid ("I want a smaller team and remote") — extract structured items from them.
- Set is_actionable=false ONLY for pure acknowledgments / no preference content, for example exactly:
  "ok", "thanks", "looks good", "cool", "yes" (alone), "idk" (alone), or similar with NO new preference.

Examples (all actionable):
- "too enterprise-heavy" → add_avoid: ["enterprise", "large companies"]
- "seed stage only" / "prefer pre-Series A" → add_must: ["seed-stage startup"], add_avoid may include ["Series B+", "growth-stage"]
- "higher pay" / "want at least $200k" → add_must: ["compensation targeting $200k+"] or add_nice: ["maximum listed compensation"]
- "aligns with CoffeeSpace" / "similar to CoffeeSpace" / talent-matching vibe → add_must or add_nice:
  ["mission aligned with cofounder hiring / marketplace for early-stage hires", "CoffeeSpace-adjacent product space"]
- "I want something more backend" → add_must: ["backend-focused"], consider add_avoid: ["full-stack"]
- "prefer earlier stage startups" → add_must: ["early-stage startup"]
- "remote only" / "I need remote" → add_must: ["remote"]
- "smaller team, less than 10 people" → add_must: ["small team (under 10 people)"]
- "actually office is fine" → remove_must: ["remote"] if remote was listed
- "these are great but I want ML" → add_must: ["machine learning", "ML engineering"]"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        delta = _extract_json_object(response.content[0].text)
    except json.JSONDecodeError:
        return current_prefs, False, "Could not parse your feedback. Please try again."

    def lists_empty(d: dict) -> bool:
        return not any(
            d.get(k)
            for k in (
                "add_must",
                "add_nice",
                "add_avoid",
                "remove_must",
                "remove_nice",
                "remove_avoid",
            )
        )

    if not delta.get(
        "is_actionable", True
    ) and _likely_has_preference_intent(feedback):
        retry_delta = _retry_actionable_parse(
            feedback, current_prefs, previous_recommendations
        )
        if retry_delta is not None:
            delta = retry_delta

    if not delta.get("is_actionable", True):
        if _likely_has_preference_intent(feedback) and lists_empty(delta):
            retry_delta = _retry_actionable_parse(
                feedback, current_prefs, previous_recommendations
            )
            if retry_delta is not None and retry_delta.get("is_actionable", False):
                delta = retry_delta

    if not delta.get("is_actionable", True):
        return current_prefs, False, (
            "Could you be more specific? For example: "
            "'I want backend roles', 'prefer early-stage startups', "
            "'remote only', or 'avoid enterprise companies'."
        )

    updated = _apply_delta(current_prefs, delta)
    return updated, True, None


def _dedup_cap(lst: list[str], cap: int = 5) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in lst:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result[:cap]


def _apply_delta(prefs: PreferenceState, delta: dict) -> PreferenceState:
    must = list(prefs.must_have)
    nice = list(prefs.nice_to_have)
    avoid = list(prefs.avoid)

    must = [x for x in must if x not in delta.get("remove_must", [])]
    nice = [x for x in nice if x not in delta.get("remove_nice", [])]
    avoid = [x for x in avoid if x not in delta.get("remove_avoid", [])]

    must = delta.get("add_must", []) + must
    nice = delta.get("add_nice", []) + nice
    avoid = delta.get("add_avoid", []) + avoid

    return PreferenceState(
        must_have=_dedup_cap(must),
        nice_to_have=_dedup_cap(nice),
        avoid=_dedup_cap(avoid),
    )

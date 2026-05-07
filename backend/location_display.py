"""Normalize verbose ATS location strings for UI / prompts (scraped jobs.json noise)."""

from __future__ import annotations

import re


def shorten_job_location(location: str | None, *, max_chars: int = 110) -> str:
    """Compact locations that are essentially country-code lists (e.g. remote EU hires)."""
    s = " ".join(str(location or "").split()).strip()
    if not s:
        return ""

    if _looks_like_country_code_parade(s):
        if re.search(r"\bremote\b", s, re.I):
            return "Remote · multi-country hiring (eligible regions listed on posting)"
        return "Multi-country hiring (see posting for eligible locations)"

    if len(s) <= max_chars:
        return s

    if "/" in s and s.count("/") >= 4:
        first = s.split("/")[0].strip()
        n_rest = s.count("/")
        if len(first) <= 72:
            return f"{first} / … (+{n_rest} more locations)"

    return s[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def _looks_like_country_code_parade(loc: str) -> bool:
    """
    Scrapers sometimes ingest 'GB / EG / RU / … / Remote (GB; EG; …)' as one field.

    Detect many slash-separated uppercase two-letter tokens (ISO-style), not sentences.
    """
    if "/" not in loc or len(loc) < 140:
        return False

    # Count pairs like "ZZ / XX" chained (strictly whitespace around slash).
    chained = len(re.findall(r"\b[A-Z]{2}\b\s*/\s*\b[A-Z]{2}\b", loc))
    if chained >= 15:
        return True

    # Fallback: absurd slash density + many isolated two-letter uppercase tokens (no commas = city-ish).
    if loc.count("/") >= 22:
        iso_tokens = len(re.findall(r"\b[A-Z]{2}\b", loc))
        if iso_tokens >= 28 and "," not in loc[:80]:
            return True

    return False

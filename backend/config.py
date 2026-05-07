"""Runtime tuning (env). Defaults favor staying under tight Anthropic TPM caps."""

import os


def retrieve_k() -> int:
    """
    Jobs retrieved then passed to Sonnet for ranking.
    Default 14 (balanced). Original take-home asked for ~30 — set MATCHLOOP_RETRIEVE_K=30.
    """
    raw = os.environ.get("MATCHLOOP_RETRIEVE_K")
    if raw is None or not str(raw).strip():
        k = 14
    else:
        try:
            k = int(raw)
        except ValueError:
            k = 14
    return max(5, min(50, k))


def rank_description_max_chars() -> int | None:
    """
    Max characters per job description in Sonnet ranking prompt only.
    - Unset → 3200 (TPM-safe)
    - 0 → no truncation (matches original ‘full descriptions’ requirement; burns tokens)
    """
    raw = os.environ.get("MATCHLOOP_RANK_DESC_MAX_CHARS")
    if raw is None:
        return 3200
    raw = raw.strip()
    try:
        v = int(raw)
    except ValueError:
        return 3200
    if v <= 0:
        return None
    return max(500, v)

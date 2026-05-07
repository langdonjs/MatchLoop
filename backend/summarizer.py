def _clip(text: str, max_chars: int) -> str:
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


def _employment_dates(emp: dict) -> str:
    start = emp.get("start_date")
    end = emp.get("end_date")
    if not start and end is None:
        return ""
    left = str(start)[:10] if start else "?"
    if end:
        right = str(end)[:10]
    else:
        right = "present"
    return f"{left} → {right}"


def build_retrieval_summary(candidate: dict) -> str:
    parts: list[str] = []
    parts.extend(candidate.get("skills") or [])
    parts.extend(candidate.get("all_titles") or [])
    parts.extend(candidate.get("all_employers") or [])
    parts.extend(candidate.get("all_schools") or [])
    summary = (candidate.get("summary") or "").strip()
    if summary:
        parts.append(summary[:900])
    for emp in candidate.get("current_employers") or []:
        if emp.get("employee_description"):
            parts.append(str(emp["employee_description"])[:400])
        if emp.get("employer_linkedin_description"):
            parts.append(str(emp["employer_linkedin_description"])[:300])
        if emp.get("employer_name"):
            parts.append(str(emp["employer_name"]))
    for emp in (candidate.get("past_employers") or [])[:6]:
        if emp.get("employee_description"):
            parts.append(str(emp["employee_description"])[:350])
        if emp.get("employer_name"):
            parts.append(str(emp["employer_name"]))
    return " ".join(filter(None, parts))


def _format_current_employers_block(current: list[dict]) -> str:
    if not current:
        return ""
    seen_company_blurb: set[str] = set()
    blocks: list[str] = []
    for i, emp in enumerate(current, 1):
        ename = str(emp.get("employer_name") or "Unknown employer").strip()
        etitle = str(emp.get("employee_title") or "").strip() or "(title not listed)"
        lines = [f"--- Current role #{i}: {etitle} @ {ename} ---"]
        dr = _employment_dates(emp)
        if dr:
            lines.append(f"Dates: {dr}")

        shards: list[str] = []
        ckey = ename.lower()
        if emp.get("employer_linkedin_description") and ckey not in seen_company_blurb:
            blurb = _clip(str(emp["employer_linkedin_description"]), 520)
            if blurb:
                shards.append(f"Company / mission ({ename}): {blurb}")
                seen_company_blurb.add(ckey)
        if emp.get("employee_description"):
            desc = _clip(str(emp["employee_description"]), 1200)
            if desc:
                shards.append(f"What they ship / achievements / scope:\n{desc}")
        if shards:
            lines.append("\n".join(shards))
        blocks.append("\n".join(lines))
    return "CURRENT EMPLOYMENT (every concurrent role — treat all as primary context):\n" + "\n\n".join(
        blocks
    )


def _format_past_employers_block(past: list[dict], *, max_roles: int = 12) -> str:
    if not past:
        return ""

    def sort_key(p: dict) -> str:
        return str(p.get("end_date") or p.get("start_date") or "")

    ordered = sorted(past, key=sort_key, reverse=True)[:max_roles]
    seen_company_blurb: set[str] = set()
    blocks: list[str] = []
    for i, emp in enumerate(ordered, 1):
        ename = str(emp.get("employer_name") or "Unknown employer").strip()
        etitle = str(emp.get("employee_title") or "").strip() or "(title not listed)"
        lines = [f"--- Past role #{i}: {etitle} @ {ename} ---"]
        dr = _employment_dates(emp)
        if dr:
            lines.append(f"Dates: {dr}")

        shards: list[str] = []
        ckey = ename.lower()
        if emp.get("employer_linkedin_description") and ckey not in seen_company_blurb:
            blurb = _clip(str(emp["employer_linkedin_description"]), 320)
            if blurb:
                shards.append(f"Company context ({ename}): {blurb}")
                seen_company_blurb.add(ckey)
        if emp.get("employee_description"):
            desc = _clip(str(emp["employee_description"]), 1000)
            if desc:
                shards.append(f"Role scope / projects / measurable outcomes:\n{desc}")
        if shards:
            lines.append("\n".join(shards))
        blocks.append("\n".join(lines))

    header = (
        "PRIOR EXPERIENCE (recent first — use these for overlap with job postings; "
        "each block is verbatim-style detail, not just a company name):\n"
    )
    return header + "\n\n".join(blocks)


def build_ranking_context(candidate: dict) -> str:
    name = candidate.get("name") or "The candidate"
    title = (candidate.get("title") or "").strip()
    headline = (candidate.get("headline") or "").strip()
    location = candidate.get("location") or ""
    summary = (candidate.get("summary") or "").strip()
    skills = ", ".join((candidate.get("skills") or [])[:42])
    titles = ", ".join((candidate.get("all_titles") or [])[:12])
    langs = candidate.get("languages") or []
    langs_line = ""
    if langs:
        langs_line = "Languages: " + "; ".join(str(x) for x in langs[:8]) + "."

    current = candidate.get("current_employers") or []
    current_block = _format_current_employers_block(current)

    past = candidate.get("past_employers") or []
    past_block = _format_past_employers_block(past)

    edu = candidate.get("education_background") or []
    edu_summary = ""
    if edu:
        degrees = [
            f"{e.get('degree_name', '')} from {e.get('institute_name', '')}"
            for e in edu[:4]
        ]
        edu_summary = " ".join(filter(None, degrees))

    title_line = title or (
        headline[:140] + ("…" if len(headline) > 140 else "") if headline else "professional"
    )

    parts_out: list[str] = [
        f"IDENTITY: {name}",
        f"Listed title field: {title_line}. Location: {location}.",
    ]
    if headline:
        parts_out.append(f"HEADLINE / POSITIONING LINE (often carries ex-employer niches, e.g. ex-Pixar):\n{headline}")
    if summary:
        parts_out.append(
            "SELF-SUMMARY (candidate narrative — weigh heavily when present):\n"
            + _clip(summary, 3200)
        )

    parts_out.extend(
        [
            f"AGGREGATED TITLES (career breadth): {titles}.",
            f"CORE SKILLS (keyword list): {skills}.",
        ]
    )
    if langs_line:
        parts_out.append(langs_line)

    if current_block:
        parts_out.extend(["", current_block])
    else:
        parts_out.extend(
            [
                "",
                "NOTE: Profile lists no roles under current_employers; rely on HEADLINE and PRIOR EXPERIENCE "
                "for recent positioning (many candidates only have headline + history).",
            ]
        )

    if past_block:
        parts_out.extend(["", past_block])

    edu_line = f"Education: {edu_summary}".strip()
    if edu_line != "Education:":
        parts_out.extend(["", edu_line])

    return "\n".join(p for p in parts_out if p).strip()

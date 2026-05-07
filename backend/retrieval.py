import os

os.environ.setdefault("USE_TF", "0")

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.models import PreferenceState


def _cosine_scores(query_vec: np.ndarray, job_matrix: np.ndarray) -> np.ndarray:
    """query_vec shape (1, d), job_matrix (n, d). Returns shape (n,)."""
    q = query_vec[0].astype(np.float64)
    qn = q / (np.linalg.norm(q) + 1e-10)
    jm = job_matrix.astype(np.float64)
    norms = np.linalg.norm(jm, axis=1, keepdims=True) + 1e-10
    jn = jm / norms
    return jn @ qn


class JobRetriever:
    def __init__(self, jobs: list[dict]):
        self.jobs = jobs
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        corpus = [
            f"{j['title']} {j.get('company', '')} {j.get('description', '')}"
            for j in jobs
        ]

        print(f"Embedding {len(jobs)} jobs... (one-time startup cost)")
        self.job_matrix = self.model.encode(corpus, show_progress_bar=True)
        print("Embedding complete. Retrieval ready.")

    def retrieve(
        self,
        candidate_summary: str,
        preference_state: PreferenceState,
        seen_urls: set[str],
        k: int = 30,
    ) -> list[dict]:
        queries = self._build_queries(candidate_summary, preference_state)

        seen_in_results: set[str] = set()
        candidate_jobs: list[tuple[float, dict]] = []

        for query in queries:
            query_vec = self.model.encode([query])
            scores = _cosine_scores(
                np.asarray(query_vec), np.asarray(self.job_matrix)
            )
            top_indices = np.argsort(scores)[::-1][:60]

            for idx in top_indices:
                job = self.jobs[int(idx)]
                url = job["url"]
                if url not in seen_urls and url not in seen_in_results:
                    seen_in_results.add(url)
                    candidate_jobs.append((float(scores[idx]), job))

        candidate_jobs.sort(key=lambda x: x[0], reverse=True)
        return [job for _, job in candidate_jobs[:k]]

    def _build_queries(
        self, candidate_summary: str, prefs: PreferenceState
    ) -> list[str]:
        queries: list[str] = [candidate_summary]

        if prefs.must_have or prefs.nice_to_have:
            pref_query = " ".join(prefs.must_have + prefs.nice_to_have)
            queries.append(pref_query)

        if prefs.must_have:
            blended = candidate_summary + " " + " ".join(prefs.must_have)
            queries.append(blended)

        return queries

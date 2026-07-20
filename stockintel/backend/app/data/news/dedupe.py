"""Near-duplicate detection for syndicated news.

One Reuters story appears verbatim across a dozen outlets. Left alone, an
aggregate sentiment score counts that single event a dozen times, and one
widely-syndicated headline swamps genuinely independent reporting. Since the
brief requires weighting by relevance and recency rather than counting
headlines, deduplication is a correctness requirement, not a cosmetic one.

Approach: character n-gram shingles plus Jaccard similarity, via an inverted
index so the comparison is near-linear rather than all-pairs. Chosen over exact
hashing (misses reworded headlines) and over embeddings (a transformer per
article is far too slow for this stage, and dedup does not need semantics --
syndicated copies are lexically near-identical by construction).
"""

from __future__ import annotations

import re
from collections import defaultdict

from app.core.logging import get_logger
from app.data.news.base import Article

logger = get_logger(__name__)

#: Shingle width in characters. 5 is the standard choice for short text: long
#: enough to be discriminative, short enough to survive small edits like an
#: outlet appending its own name to a headline.
SHINGLE_SIZE = 5

#: Jaccard similarity at or above which two articles are the same story.
#: 0.6 tolerates the usual syndication edits (added dateline, trimmed
#: subclause) without merging two genuinely different stories about one company.
SIMILARITY_THRESHOLD = 0.6

_PUNCTUATION = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    lowered = text.lower()
    lowered = _PUNCTUATION.sub(" ", lowered)
    return _WHITESPACE.sub(" ", lowered).strip()


def _shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    """Character n-gram set."""
    normalised = _normalise(text)
    if len(normalised) < size:
        return {normalised} if normalised else set()
    return {normalised[i : i + size] for i in range(len(normalised) - size + 1)}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """|intersection| / |union|. Zero when either side is empty."""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    return intersection / len(left | right)


def deduplicate(
    articles: list[Article],
    *,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[Article]:
    """Collapse near-duplicate articles, keeping the best representative.

    Representative selection prefers the earliest publication (closest to the
    originating wire) and, among simultaneous copies, the one with the longest
    description, since that carries the most signal into FinBERT.

    The survivor's `duplicate_count` records how many copies were folded in --
    displayed in the UI, and usable later as a crude proxy for how widely an
    event was covered.

    Returns:
        Deduplicated articles, newest first.
    """
    if len(articles) <= 1:
        return list(articles)

    shingle_sets = [_shingles(article.title) for article in articles]

    # Inverted index: shingle -> article positions. Comparing only articles
    # that share at least one shingle avoids the O(n^2) all-pairs scan.
    index: dict[str, list[int]] = defaultdict(list)
    for position, shingles in enumerate(shingle_sets):
        for shingle in shingles:
            index[shingle].append(position)

    assigned: dict[int, int] = {}   # article position -> cluster id
    clusters: list[list[int]] = []

    for position in range(len(articles)):
        if position in assigned:
            continue

        cluster_id = len(clusters)
        clusters.append([position])
        assigned[position] = cluster_id

        candidates: set[int] = set()
        for shingle in shingle_sets[position]:
            candidates.update(index[shingle])

        for candidate in candidates:
            if candidate <= position or candidate in assigned:
                continue
            if jaccard_similarity(shingle_sets[position], shingle_sets[candidate]) >= threshold:
                assigned[candidate] = cluster_id
                clusters[cluster_id].append(candidate)

    survivors: list[Article] = []
    for members in clusters:
        best_position = min(
            members,
            key=lambda p: (articles[p].published_at, -len(articles[p].description)),
        )
        survivor = articles[best_position]
        survivor.duplicate_count = len(members) - 1
        survivors.append(survivor)

    removed = len(articles) - len(survivors)
    if removed:
        logger.info("Deduplicated %d syndicated copies from %d articles", removed, len(articles))

    survivors.sort(key=lambda a: a.published_at, reverse=True)
    return survivors

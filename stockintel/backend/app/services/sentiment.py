"""Financial sentiment via FinBERT.

`ProsusAI/finbert` is BERT further pretrained on financial text and fine-tuned
on the Financial PhraseBank. It is used here rather than a general-purpose
sentiment model because financial language inverts ordinary polarity
constantly: "shares fell less than feared", "narrowed its loss" and "cut
guidance but beat on margins" are all mis-scored by general models trained on
product reviews.

Three engineering decisions worth stating:

* **Lazy loading.** Weights (~440MB) download and load on first use, not at
  import, so the API starts instantly and a user who never opens the news panel
  never pays the cost.
* **Cached by content hash.** Identical text is never re-scored. The brief
  requires this, and it matters: FinBERT on CPU is ~50ms per article.
* **Degrades to UNAVAILABLE, never to neutral.** If the model cannot load, the
  UI says sentiment is unavailable. Silently emitting NEUTRAL would be
  indistinguishable from a real reading of "the news is balanced" -- a
  fabrication in exactly the sense the brief forbids.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.logging import get_logger
from app.data.cache.store import get_cache
from app.data.news.base import Article, TextSentiment

logger = get_logger(__name__)

MODEL_NAME = "ProsusAI/finbert"

#: Articles needed before an aggregate sentiment reading is treated as a
#: reliable measure rather than an indicative one.
MIN_ARTICLES_FOR_CONFIDENT_AGGREGATE = 8

#: FinBERT's positional limit is 512 tokens; titles plus descriptions sit far
#: below this, but truncation is set explicitly so a long description can never
#: raise at inference time.
MAX_TOKENS = 512

_model = None
_tokenizer = None
_load_lock = threading.Lock()
_load_failed = False


@dataclass(frozen=True)
class SentimentResult:
    """FinBERT's reading of one piece of text."""

    label: TextSentiment
    confidence: float
    scores: dict[str, float]

    @property
    def signed_score(self) -> float:
        """Sentiment on a [-1, +1] scale: P(positive) - P(negative).

        More informative than the argmax label for aggregation, since it keeps
        the margin. An article at 0.45/0.45/0.10 and one at 0.90/0.05/0.05 both
        label POSITIVE but carry very different weight.
        """
        return self.scores.get("positive", 0.0) - self.scores.get("negative", 0.0)


def _load_model() -> bool:
    """Load FinBERT once, on first use. Returns False if unavailable."""
    global _model, _tokenizer, _load_failed

    if _model is not None:
        return True
    if _load_failed:
        return False

    with _load_lock:
        if _model is not None:
            return True
        if _load_failed:
            return False

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info("Loading FinBERT (%s) — first run downloads ~440MB", MODEL_NAME)
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            model.eval()
            torch.set_grad_enabled(False)
            _model = model
            logger.info("FinBERT loaded (labels: %s)", model.config.id2label)
            return True
        except Exception as exc:
            # Offline, out of disk, incompatible transformers version -- all
            # degrade to "sentiment unavailable", never to a fake neutral.
            logger.error("FinBERT unavailable: %s", exc)
            _load_failed = True
            return False


def is_available() -> bool:
    """Whether sentiment analysis can run. Triggers a load attempt."""
    return _load_model()


def analyse_text(text: str, *, use_cache: bool = True) -> SentimentResult:
    """Score one piece of text.

    Returns an UNAVAILABLE result rather than raising, so one unscoreable
    article never sinks a whole news panel.
    """
    import hashlib

    cleaned = text.strip()
    if not cleaned:
        return SentimentResult(TextSentiment.UNAVAILABLE, 0.0, {})

    cache = get_cache()
    settings = get_settings()
    key = hashlib.sha256(cleaned.lower().encode("utf-8")).hexdigest()[:32]

    if use_cache:
        hit = cache.get_json("finbert", key)
        if hit is not None:
            payload = hit.value
            return SentimentResult(
                label=TextSentiment(payload["label"]),
                confidence=payload["confidence"],
                scores=payload["scores"],
            )

    if not _load_model():
        return SentimentResult(TextSentiment.UNAVAILABLE, 0.0, {})

    try:
        import torch

        inputs = _tokenizer(
            cleaned, return_tensors="pt", truncation=True, max_length=MAX_TOKENS, padding=True
        )
        logits = _model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

        # Read labels from the model config rather than assuming an order --
        # index-order assumptions are a common and silent source of inverted
        # sentiment.
        scores = {
            _model.config.id2label[index].lower(): float(probabilities[index])
            for index in range(len(probabilities))
        }
    except Exception as exc:
        logger.warning("FinBERT inference failed: %s", exc)
        return SentimentResult(TextSentiment.UNAVAILABLE, 0.0, {})

    best_label = max(scores, key=scores.get)
    result = SentimentResult(
        label=TextSentiment[best_label.upper()] if best_label.upper() in TextSentiment.__members__
        else TextSentiment.NEUTRAL,
        confidence=scores[best_label],
        scores=scores,
    )

    if use_cache:
        cache.set_json(
            "finbert", key,
            {"label": result.label.value, "confidence": result.confidence, "scores": result.scores},
            ttl_seconds=settings.cache_ttl_sentiment,
        )

    return result


def analyse_articles(articles: list[Article]) -> list[Article]:
    """Attach sentiment to each article in place."""
    if not articles:
        return articles

    if not _load_model():
        logger.warning("FinBERT unavailable; articles will report UNAVAILABLE sentiment.")
        return articles

    for article in articles:
        result = analyse_text(article.analysed_text)
        article.text_sentiment = result.label
        article.sentiment_confidence = result.confidence
        article.sentiment_scores = result.scores

    return articles


def aggregate_sentiment(
    articles: list[Article],
    *,
    half_life_hours: float = 48.0,
) -> dict[str, object]:
    """Weighted aggregate sentiment across articles.

    The brief explicitly rules out counting positive and negative headlines, so
    each article's signed score is weighted by the product of:

    * **Recency** — exponential decay with a 48h half-life. Yesterday's
      earnings miss is priced in; this morning's is not.
    * **Relevance** — the relevance engine's score, where available. A macro
      story mentioning a company in passing should not move its sentiment as
      much as its own earnings report.
    * **Model confidence** — a 0.95-confidence reading counts nearly twice a
      0.50 one.

    Syndication is handled upstream by deduplication, so a widely-carried wire
    story contributes once.
    """
    scored = [a for a in articles if a.text_sentiment is not TextSentiment.UNAVAILABLE]

    if not scored:
        return {
            "available": False,
            "reason": "No articles could be scored.",
            "n_articles": len(articles),
        }

    total_weight = 0.0
    weighted_sum = 0.0
    counts = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}

    for article in scored:
        counts[article.text_sentiment.value] += 1

        recency_weight = 0.5 ** (article.age_hours() / half_life_hours)
        relevance_weight = article.relevance_score if article.relevance_score is not None else 1.0
        confidence_weight = article.sentiment_confidence or 0.5

        weight = recency_weight * relevance_weight * confidence_weight
        signed = article.sentiment_scores.get("positive", 0.0) - article.sentiment_scores.get(
            "negative", 0.0
        )

        weighted_sum += signed * weight
        total_weight += weight

    if total_weight <= 1e-9:
        return {
            "available": False,
            "reason": "All articles were too old or too weakly relevant to weight.",
            "n_articles": len(articles),
        }

    net_score = weighted_sum / total_weight

    if net_score > 0.15:
        label = "POSITIVE"
    elif net_score < -0.15:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    # Coverage caveat. An aggregate over 3 articles is not comparable to one
    # over 30: a single outlier dominates it. The news plan in use returns very
    # few articles per request, so this is the common case rather than an edge
    # case, and the reading must be labelled accordingly instead of being
    # presented with the same weight as a well-covered one.
    n_scored = len(scored)
    if n_scored >= MIN_ARTICLES_FOR_CONFIDENT_AGGREGATE:
        coverage = "ADEQUATE"
        coverage_note = ""
    elif n_scored >= 3:
        coverage = "THIN"
        coverage_note = (
            f"Based on only {n_scored} articles — a single story can dominate this "
            f"reading. Treat it as indicative rather than a reliable sentiment measure."
        )
    else:
        coverage = "INSUFFICIENT"
        coverage_note = (
            f"Only {n_scored} article(s) available. Too few to characterise sentiment."
        )

    return {
        "available": True,
        "label": label,
        "net_score": round(net_score, 4),
        "n_articles": n_scored,
        "n_unscoreable": len(articles) - n_scored,
        "counts": counts,
        "coverage": coverage,
        "coverage_note": coverage_note,
        "method": (
            "Signed FinBERT score per article, weighted by recency (48h half-life), "
            "relevance and model confidence. Syndicated duplicates removed first."
        ),
    }

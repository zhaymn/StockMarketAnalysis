"""Event relevance and expected company impact.

Implements the pipeline the brief specifies:

    article -> entity extraction -> company matching -> sector matching
            -> business exposure -> relevance score -> sentiment
            -> expected impact direction -> magnitude -> time horizon

**The central rule: negative text does not imply a bearish stock.** The brief's
own example -- "competitor suffers major production failure" -- is negative
language and plausibly *bullish* for the company being analysed. So this module
keeps two separate fields:

* `text_sentiment` — FinBERT's reading of the language (set upstream).
* `expected_impact` — the directional implication for *this* company, derived
  from sentiment **and** the relationship between the article's subject and the
  company.

The relationship is what flips the sign. An article about a rival's factory
fire is `COMPETITOR`-related, so negative text maps to a bullish (weak) impact.

**Everything here is rule-based and inspectable.** No impact claim is generated
without a matched, citable reason, and every reason is surfaced in the UI. When
the rules cannot establish a relationship, the impact is `UNCERTAIN` -- not a
guess. Per the brief, all impact language is probabilistic: "may", "could",
"historically tends to".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from app.core.logging import get_logger
from app.data.fundamentals.profile import CompanyProfile
from app.data.news.base import Article, NewsCategory, TextSentiment

logger = get_logger(__name__)


class ImpactDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    MIXED = "MIXED"
    UNCERTAIN = "UNCERTAIN"


class ImpactMagnitude(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TimeHorizon(str, Enum):
    IMMEDIATE = "IMMEDIATE"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"


class Relationship(str, Enum):
    """How the article's subject relates to the company being analysed.

    This is what determines whether text polarity carries through or inverts.
    """

    DIRECT = "DIRECT"
    """About the company itself. Sentiment carries through unchanged."""

    COMPETITOR = "COMPETITOR"
    """About a named rival. Sentiment inverts -- a competitor's misfortune is
    not the analysed company's misfortune."""

    SECTOR = "SECTOR"
    """About the company's sector or industry. Sentiment carries through, but
    weakened -- sector news moves a single name less than its own news."""

    MACRO = "MACRO"
    """Economy-wide. Carries through, weakest of all, and only where the
    company has plausible exposure."""

    UNRELATED = "UNRELATED"


#: Event types that move prices, with the magnitude and horizon they typically
#: carry. Keyword-driven and deliberately conservative: an unmatched article
#: gets LOW magnitude rather than a guessed one.
EVENT_PATTERNS: list[tuple[str, tuple[str, ...], ImpactMagnitude, TimeHorizon]] = [
    ("Earnings result",
     ("earnings", "quarterly result", "q1 result", "q2 result", "q3 result", "q4 result",
      "profit", "revenue", "beat estimates", "missed estimates", "eps"),
     ImpactMagnitude.HIGH, TimeHorizon.IMMEDIATE),
    ("Guidance change",
     ("guidance", "outlook", "forecast", "raises target", "cuts target"),
     ImpactMagnitude.HIGH, TimeHorizon.SHORT_TERM),
    ("Merger or acquisition",
     ("acquisition", "acquire", "merger", "takeover", "buyout", "stake sale"),
     ImpactMagnitude.HIGH, TimeHorizon.MEDIUM_TERM),
    ("Regulatory or legal",
     ("lawsuit", "investigation", "regulator", "antitrust", "fine", "penalty",
      "sanction", "export restriction", "ban", "probe"),
     ImpactMagnitude.HIGH, TimeHorizon.MEDIUM_TERM),
    ("Leadership change",
     ("chief executive", "ceo", "cfo", "resign", "steps down", "appoint", "successor"),
     ImpactMagnitude.MEDIUM, TimeHorizon.SHORT_TERM),
    ("Product or partnership",
     ("launch", "unveil", "partnership", "contract", "deal", "order win", "new product"),
     ImpactMagnitude.MEDIUM, TimeHorizon.MEDIUM_TERM),
    ("Analyst action",
     ("upgrade", "downgrade", "price target", "initiates coverage", "rating"),
     ImpactMagnitude.MEDIUM, TimeHorizon.IMMEDIATE),
    ("Dividend or buyback",
     ("dividend", "buyback", "share repurchase", "bonus issue", "split"),
     ImpactMagnitude.MEDIUM, TimeHorizon.SHORT_TERM),
    ("Monetary policy",
     ("interest rate", "rate cut", "rate hike", "central bank", "federal reserve",
      "monetary policy", "repo rate", "inflation", "cpi"),
     ImpactMagnitude.MEDIUM, TimeHorizon.MEDIUM_TERM),
    ("Trade policy",
     ("tariff", "trade war", "import duty", "export control", "trade deal"),
     ImpactMagnitude.MEDIUM, TimeHorizon.MEDIUM_TERM),
    ("Supply chain",
     ("supply chain", "shortage", "production halt", "factory", "plant closure",
      "recall", "disruption"),
     ImpactMagnitude.MEDIUM, TimeHorizon.SHORT_TERM),
]

#: Sector -> themes used for sector-news search and sector matching. Keyed on
#: the sector strings Yahoo actually returns.
SECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Technology": ("semiconductor", "chip", "software", "cloud", "artificial intelligence",
                   "data center", "hardware"),
    "Financial Services": ("bank", "lending", "credit", "interest margin", "npa",
                           "insurance", "fintech"),
    "Healthcare": ("pharmaceutical", "drug", "clinical trial", "fda", "biotech", "medical"),
    "Consumer Cyclical": ("retail", "consumer demand", "auto", "vehicle", "e-commerce"),
    "Consumer Defensive": ("fmcg", "consumer staples", "grocery", "beverage"),
    "Energy": ("oil", "crude", "natural gas", "refinery", "opec", "petroleum"),
    "Basic Materials": ("steel", "cement", "metal", "mining", "commodity"),
    "Industrials": ("manufacturing", "infrastructure", "aerospace", "defence", "logistics"),
    "Communication Services": ("telecom", "spectrum", "5g", "streaming", "advertising"),
    "Utilities": ("power", "electricity", "renewable", "grid"),
    "Real Estate": ("real estate", "property", "reit", "housing"),
}


@dataclass
class EventImpact:
    """Assessed implication of one article for one company."""

    relationship: Relationship
    relevance_score: float
    """0-1. Below `MIN_RELEVANCE` the article is dropped as noise."""

    direction: ImpactDirection
    magnitude: ImpactMagnitude
    horizon: TimeHorizon

    event_type: str | None = None
    what_happened: str = ""
    why_relevant: str = ""
    reasoning: str = ""
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "relationship": self.relationship.value,
            "relevance_score": round(self.relevance_score, 4),
            "expected_impact": self.direction.value,
            "impact_magnitude": self.magnitude.value,
            "time_horizon": self.horizon.value,
            "event_type": self.event_type,
            "what_happened": self.what_happened,
            "why_relevant": self.why_relevant,
            "reasoning": self.reasoning,
            "matched_terms": self.matched_terms,
        }


#: Articles scoring below this are excluded. The brief forbids showing
#: irrelevant generic world news.
MIN_RELEVANCE = 0.25

_TOKEN = re.compile(r"[a-z0-9]+")


def _company_name_variants(profile: CompanyProfile) -> set[str]:
    """Name forms an article might use.

    Strips corporate suffixes so "Reliance Industries Limited" also matches a
    headline that just says "Reliance".
    """
    name = profile.name.lower()
    variants = {name}

    for suffix in (" limited", " ltd.", " ltd", " inc.", " inc", " corporation",
                   " corp.", " corp", " plc", " co.", " company", " holdings",
                   " group", " industries", " technologies", " & co"):
        if name.endswith(suffix):
            variants.add(name[: -len(suffix)].strip())

    # Leading token, when distinctive enough to be unambiguous.
    tokens = _TOKEN.findall(name)
    if tokens and len(tokens[0]) >= 5:
        variants.add(tokens[0])

    base_symbol = profile.symbol.split(".")[0].lower()
    if len(base_symbol) >= 3:
        variants.add(base_symbol)

    return {v for v in variants if len(v) >= 3}


def _match_event_type(text: str) -> tuple[str | None, ImpactMagnitude, TimeHorizon, list[str]]:
    """Classify the event, returning its typical magnitude and horizon."""
    best: tuple[str | None, ImpactMagnitude, TimeHorizon, list[str]] = (
        None, ImpactMagnitude.LOW, TimeHorizon.SHORT_TERM, []
    )
    best_hits = 0

    for event_type, keywords, magnitude, horizon in EVENT_PATTERNS:
        hits = [keyword for keyword in keywords if keyword in text]
        if len(hits) > best_hits:
            best_hits = len(hits)
            best = (event_type, magnitude, horizon, hits)

    return best


def assess_impact(
    article: Article,
    profile: CompanyProfile,
    *,
    competitor_names: tuple[str, ...] = (),
) -> EventImpact:
    """Assess one article's implication for one company.

    Args:
        article: Article with FinBERT sentiment already attached.
        profile: Company being analysed.
        competitor_names: Known rivals. Supplying these enables the
            sentiment-inversion path; without them a competitor story is
            treated as sector news, which weakens but does not invert it.
    """
    text = f"{article.title} {article.description}".lower()

    # --- 1. Entity and company matching -----------------------------------
    variants = _company_name_variants(profile)
    matched_names = [variant for variant in variants if variant in text]
    symbol_tagged = any(
        tag.upper().split(".")[0] == profile.symbol.split(".")[0].upper()
        for tag in article.tagged_symbols
    )

    # --- 2. Competitor matching -------------------------------------------
    matched_competitors = [
        name for name in competitor_names if name.lower() in text
    ]

    # --- 3. Sector matching -----------------------------------------------
    sector_keywords = SECTOR_KEYWORDS.get(profile.sector or "", ())
    matched_sector_terms = [keyword for keyword in sector_keywords if keyword in text]

    # --- 4. Relationship and base relevance -------------------------------
    reasons: list[str] = []

    if symbol_tagged or matched_names:
        relationship = Relationship.DIRECT
        relevance = 0.95 if symbol_tagged else 0.85
        reasons.append(
            f"Article names {profile.name}" if matched_names
            else f"Provider tagged this article with {profile.symbol}"
        )
    elif matched_competitors:
        relationship = Relationship.COMPETITOR
        relevance = 0.55
        reasons.append(f"Concerns competitor(s): {', '.join(matched_competitors)}")
    elif matched_sector_terms:
        relationship = Relationship.SECTOR
        relevance = 0.45
        reasons.append(
            f"Relates to {profile.sector} themes: {', '.join(matched_sector_terms[:3])}"
        )
    elif article.category is NewsCategory.MACRO:
        relationship = Relationship.MACRO
        relevance = 0.30
        reasons.append("Macroeconomic development with broad equity-market exposure")
    else:
        return EventImpact(
            relationship=Relationship.UNRELATED,
            relevance_score=0.0,
            direction=ImpactDirection.UNCERTAIN,
            magnitude=ImpactMagnitude.LOW,
            horizon=TimeHorizon.SHORT_TERM,
            what_happened=article.title,
            why_relevant="No established link to this company, its sector, or the macro picture.",
            reasoning="Excluded as not relevant.",
        )

    # --- 5. Event type, magnitude, horizon --------------------------------
    event_type, magnitude, horizon, matched_terms = _match_event_type(text)
    if event_type:
        reasons.append(f"Identified as: {event_type}")
        relevance = min(1.0, relevance + 0.05)

    # Indirect news moves a single stock less, whatever the event type.
    if relationship in (Relationship.SECTOR, Relationship.MACRO) and magnitude is ImpactMagnitude.HIGH:
        magnitude = ImpactMagnitude.MEDIUM
    if relationship is Relationship.MACRO and magnitude is ImpactMagnitude.MEDIUM:
        magnitude = ImpactMagnitude.LOW

    # --- 6. Expected impact — where text sentiment may invert -------------
    direction, impact_reason = _derive_direction(article, relationship, profile)
    reasons.append(impact_reason)

    return EventImpact(
        relationship=relationship,
        relevance_score=relevance,
        direction=direction,
        magnitude=magnitude,
        horizon=horizon,
        event_type=event_type,
        what_happened=article.title,
        why_relevant="; ".join(reasons[:-1]) if len(reasons) > 1 else reasons[0],
        reasoning=impact_reason,
        matched_terms=matched_terms,
    )


def _derive_direction(
    article: Article,
    relationship: Relationship,
    profile: CompanyProfile,
) -> tuple[ImpactDirection, str]:
    """Map text sentiment to expected share-price impact.

    The inversion for `COMPETITOR` is the point of the whole module. All
    returned language is probabilistic, per the brief.
    """
    sentiment = article.text_sentiment

    if sentiment is TextSentiment.UNAVAILABLE:
        return (
            ImpactDirection.UNCERTAIN,
            "Sentiment could not be determined, so the share-price implication is unclear.",
        )

    if sentiment is TextSentiment.NEUTRAL:
        return (
            ImpactDirection.UNCERTAIN,
            "The language is factual rather than directional; little price impact is implied.",
        )

    positive = sentiment is TextSentiment.POSITIVE

    if relationship is Relationship.DIRECT:
        return (
            (ImpactDirection.BULLISH if positive else ImpactDirection.BEARISH),
            (
                f"{'Positive' if positive else 'Negative'} company-specific news may "
                f"{'support' if positive else 'weigh on'} {profile.name}'s share price."
            ),
        )

    if relationship is Relationship.COMPETITOR:
        # The inversion. Negative competitor news is potentially bullish.
        return (
            (ImpactDirection.BEARISH if positive else ImpactDirection.BULLISH),
            (
                f"This concerns a competitor rather than {profile.name}. "
                + (
                    "A rival's positive news could indicate competitive pressure, which may "
                    "weigh on the company."
                    if positive else
                    "Difficulty at a rival could ease competitive pressure, which may "
                    "potentially benefit the company."
                )
            ),
        )

    if relationship is Relationship.SECTOR:
        return (
            (ImpactDirection.BULLISH if positive else ImpactDirection.BEARISH),
            (
                f"Sector-wide {'strength' if positive else 'weakness'} in "
                f"{profile.sector or 'this sector'} historically tends to move constituent "
                f"stocks in the same direction, though company-specific factors may dominate."
            ),
        )

    # MACRO — real but diffuse; MIXED is the honest label.
    return (
        ImpactDirection.MIXED,
        (
            f"Macro developments affect equities broadly. The direction for {profile.name} "
            f"specifically depends on its exposure, which this analysis does not measure."
        ),
    )


def score_articles(
    articles: list[Article],
    profile: CompanyProfile,
    *,
    competitor_names: tuple[str, ...] = (),
    min_relevance: float = MIN_RELEVANCE,
) -> list[tuple[Article, EventImpact]]:
    """Assess and filter articles for one company, most relevant first."""
    assessed: list[tuple[Article, EventImpact]] = []

    for article in articles:
        impact = assess_impact(article, profile, competitor_names=competitor_names)
        if impact.relevance_score < min_relevance:
            continue

        article.relevance_score = impact.relevance_score
        article.relevance_reasons = [impact.why_relevant]
        assessed.append((article, impact))

    dropped = len(articles) - len(assessed)
    if dropped:
        logger.info("Dropped %d/%d articles below relevance %.2f",
                    dropped, len(articles), min_relevance)

    assessed.sort(key=lambda pair: (pair[1].relevance_score, pair[0].published_at), reverse=True)
    return assessed

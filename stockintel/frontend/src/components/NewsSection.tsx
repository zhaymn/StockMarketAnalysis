"use client";

import { api } from "@/lib/api";
import { useAsyncData } from "@/lib/useAsyncData";
import { relativeTime } from "@/lib/format";
import { Badge, Card, EmptyState, InfoTip, Section, Skeleton } from "./ui";

interface ArticleImpact {
  relationship: string;
  relevance_score: number;
  expected_impact: "BULLISH" | "BEARISH" | "MIXED" | "UNCERTAIN";
  impact_magnitude: "LOW" | "MEDIUM" | "HIGH";
  time_horizon: string;
  event_type: string | null;
  what_happened: string;
  why_relevant: string;
  reasoning: string;
}

interface NewsArticle {
  title: string;
  url: string;
  source: string;
  published_at: string;
  description: string;
  text_sentiment: string;
  sentiment_confidence: number | null;
  duplicate_count: number;
  impact: ArticleImpact;
}

interface NewsPayload {
  sections: Record<string, NewsArticle[]>;
  aggregate_sentiment: Record<string, { available: boolean; label?: string; net_score?: number; n_articles?: number; reason?: string }>;
  sentiment_model: { name: string; available: boolean; limitations: string };
  note: string;
}

const SECTION_TITLES: Record<string, string> = {
  company: "Company news",
  sector: "Sector news",
  macro: "Macro & current affairs",
};

const IMPACT_TONE = {
  BULLISH: "bullish",
  BEARISH: "bearish",
  MIXED: "warning",
  UNCERTAIN: "neutral",
} as const;

const SENTIMENT_TONE: Record<string, "bullish" | "bearish" | "neutral"> = {
  POSITIVE: "bullish",
  NEGATIVE: "bearish",
  NEUTRAL: "neutral",
  UNAVAILABLE: "neutral",
};

export function NewsSection({ symbol }: { symbol: string }) {
  const { data, error, isLoading } = useAsyncData<NewsPayload>(symbol, (signal) =>
    api.news(symbol, signal).then((payload) => payload as unknown as NewsPayload),
  );

  if (isLoading) {
    return (
      <Section title="News &amp; current affairs">
        <div className="space-y-4">
          <Skeleton className="h-28 w-full" />
          <Skeleton className="h-28 w-full" />
        </div>
      </Section>
    );
  }

  // The not-configured state: explicit, actionable, never filled with fake news.
  if (error?.isNotConfigured) {
    return (
      <Section title="News &amp; current affairs">
        <EmptyState
          tone="warning"
          title="News API not configured"
          message={error.payload.reason ?? error.message}
          action={
            <div className="mt-3 w-full rounded-lg border border-line bg-ink-000 p-4">
              <div className="eyebrow mb-2">To enable this section</div>
              <ol className="space-y-1.5 text-sm text-text-secondary">
                <li>
                  1. Get a free API key at{" "}
                  <a
                    href={error.payload.obtain_at}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-lime underline underline-offset-2 hover:text-lime-dim"
                  >
                    {error.payload.obtain_at}
                  </a>
                </li>
                <li>
                  2. Add{" "}
                  <code className="tnum rounded bg-ink-030 px-1.5 py-0.5 text-xs text-lime">
                    {error.payload.env_var}=your_key
                  </code>{" "}
                  to{" "}
                  <code className="tnum rounded bg-ink-030 px-1.5 py-0.5 text-xs text-text-primary">
                    stockintel/backend/.env
                  </code>
                </li>
                <li>3. Restart the backend</li>
              </ol>
              <p className="mt-3 border-t border-line pt-3 text-xs text-text-faint">
                No placeholder articles are shown in the meantime. This platform does
                not fabricate news.
              </p>
            </div>
          }
        />
      </Section>
    );
  }

  if (error || !data) {
    return (
      <Section title="News &amp; current affairs">
        <EmptyState
          title="News unavailable"
          message={error?.message ?? "News could not be retrieved."}
          tone="warning"
        />
      </Section>
    );
  }

  return (
    <>
      <Section
        title="News &amp; current affairs"
        subtitle={data.note}
        action={
          <div className="flex flex-wrap gap-2">
            {Object.entries(data.aggregate_sentiment).map(([key, aggregate]) =>
              aggregate.available ? (
                <Badge key={key} tone={SENTIMENT_TONE[aggregate.label ?? "NEUTRAL"]}>
                  {SECTION_TITLES[key]?.split(" ")[0]}: {aggregate.label}
                </Badge>
              ) : null,
            )}
          </div>
        }
      >
        <div className="space-y-8">
          {Object.entries(SECTION_TITLES).map(([key, title]) => {
            const articles = data.sections[key] ?? [];
            const aggregate = data.aggregate_sentiment[key];

            return (
              <div key={key}>
                <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                  <h3 className="eyebrow">{title}</h3>
                  {aggregate?.available && (
                    <span className="text-xs text-text-muted">
                      Weighted sentiment{" "}
                      <span className="tnum font-semibold text-text-primary">
                        {aggregate.net_score! > 0 ? "+" : ""}
                        {aggregate.net_score!.toFixed(3)}
                      </span>{" "}
                      across {aggregate.n_articles} deduplicated articles
                      <InfoTip text="Signed FinBERT score weighted by recency, relevance and model confidence — not a count of positive versus negative headlines." />
                    </span>
                  )}
                </div>

                {articles.length === 0 ? (
                  <Card>
                    <p className="text-sm text-text-muted">
                      {aggregate?.reason ??
                        "No relevant articles found in the lookback window."}
                    </p>
                  </Card>
                ) : (
                  <div className="space-y-3">
                    {articles.slice(0, 6).map((article) => (
                      <ArticleCard key={article.url} article={article} />
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </Section>

      <Card>
        <h3 className="eyebrow mb-2">Sentiment model</h3>
        <p className="text-sm text-text-secondary">{data.sentiment_model.name}</p>
        <p className="mt-2 text-xs leading-relaxed text-text-faint">
          {data.sentiment_model.limitations}
        </p>
      </Card>
    </>
  );
}

function ArticleCard({ article }: { article: NewsArticle }) {
  const { impact } = article;

  return (
    <Card className="transition-colors hover:border-line-strong">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <a
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-medium leading-snug text-text-primary hover:text-lime"
          >
            {article.title}
          </a>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-text-muted">
            <span>{article.source}</span>
            <span aria-hidden="true">·</span>
            <span>{relativeTime(article.published_at)}</span>
            {article.duplicate_count > 0 && (
              <>
                <span aria-hidden="true">·</span>
                <span>+{article.duplicate_count} syndicated copies merged</span>
              </>
            )}
          </div>
        </div>

        <div className="flex shrink-0 flex-wrap items-center gap-2">
          <Badge tone={SENTIMENT_TONE[article.text_sentiment]}>
            Text: {article.text_sentiment}
          </Badge>
          <Badge tone={IMPACT_TONE[impact.expected_impact]}>
            Impact: {impact.expected_impact}
          </Badge>
        </div>
      </div>

      <div className="mt-4 grid gap-4 border-t border-line pt-4 sm:grid-cols-[2fr_1fr]">
        <div className="space-y-2 text-xs leading-relaxed">
          <div>
            <span className="font-semibold text-text-muted">Why relevant: </span>
            <span className="text-text-secondary">{impact.why_relevant}</span>
          </div>
          <div>
            <span className="font-semibold text-text-muted">Possible impact: </span>
            <span className="text-text-secondary">{impact.reasoning}</span>
          </div>
        </div>

        <dl className="space-y-1.5 text-xs">
          {[
            ["Event type", impact.event_type ?? "Unclassified"],
            ["Magnitude", impact.impact_magnitude],
            ["Horizon", impact.time_horizon.replace(/_/g, " ")],
            ["Relevance", `${(impact.relevance_score * 100).toFixed(0)}%`],
          ].map(([label, value]) => (
            <div key={label} className="flex justify-between gap-2">
              <dt className="text-text-faint">{label}</dt>
              <dd className="tnum text-right text-text-secondary">{value}</dd>
            </div>
          ))}
        </dl>
      </div>
    </Card>
  );
}

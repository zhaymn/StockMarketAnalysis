import Link from "next/link";
import type { Metadata } from "next";

/**
 * Landing page.
 *
 * A Server Component: it is static content with no interactivity beyond links,
 * so there is no reason to ship it as client JavaScript.
 *
 * Every figure on this page is a real measured result from
 * `backend/.artifacts/*.json`. That is not incidental — a page whose argument
 * is "most stock predictors report numbers they cannot support" would be
 * self-refuting if its own numbers were decorative.
 */

export const metadata: Metadata = {
  title: "StockIntel — Prediction with the evidence attached",
  description:
    "A stock prediction platform built to test whether machine learning can "
    + "forecast equity direction — and to report honestly that, on price data, it does not.",
};

/* ---------------------------------------------------------------- primitives */

function Section({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    // scroll-mt clears the sticky nav: without it, scrolling a section to the
    // top of the viewport parks its heading underneath the header bar.
    <section className={`scroll-mt-20 border-b border-line ${className}`}>
      <div className="mx-auto max-w-6xl px-6 py-20 sm:px-8 lg:py-28">{children}</div>
    </section>
  );
}

function Eyebrow({ children }: { children: React.ReactNode }) {
  return <div className="eyebrow mb-4 text-lime">{children}</div>;
}

function Heading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="max-w-3xl text-3xl font-bold leading-tight tracking-tight text-text-primary sm:text-4xl">
      {children}
    </h2>
  );
}

function Lead({ children }: { children: React.ReactNode }) {
  return (
    <p className="mt-5 max-w-2xl text-base leading-relaxed text-text-secondary sm:text-lg">
      {children}
    </p>
  );
}

/* ---------------------------------------------------------------------- page */

export default function Landing() {
  return (
    <div className="min-h-screen bg-ink-000">
      <Nav />
      <Hero />
      <TheProblem />
      <WhatWeDidDifferently />
      <TheFinding />
      <WhatItActuallyDoes />
      <WhatItWillNotDo />
      <ClosingCta />
      <Footer />
    </div>
  );
}

function Nav() {
  return (
    <nav className="sticky top-0 z-50 border-b border-line bg-ink-000/85 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4 sm:px-8">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-lime">
            <span className="text-sm font-bold text-ink-000">SI</span>
          </div>
          <span className="text-sm font-semibold tracking-tight text-text-primary">
            StockIntel
          </span>
        </div>
        <div className="flex items-center gap-6">
          <a
            href="https://github.com/zhaymn/StockMarketAnalysis"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden text-sm text-text-secondary transition-colors hover:text-text-primary sm:block"
          >
            Source
          </a>
          <Link
            href="/dashboard"
            className="rounded-lg bg-lime px-4 py-2 text-sm font-semibold text-ink-000 transition-opacity hover:opacity-90"
          >
            Open dashboard
          </Link>
        </div>
      </div>
    </nav>
  );
}

function Hero() {
  return (
    <header className="border-b border-line">
      <div className="mx-auto max-w-6xl px-6 py-24 sm:px-8 lg:py-36">
        <div className="inline-flex items-center gap-2 rounded-full border border-line-strong bg-ink-020 px-3 py-1.5 text-xs text-text-secondary">
          <span className="h-1.5 w-1.5 rounded-full bg-lime" aria-hidden="true" />
          Two markets · four model families · 121 tests
        </div>

        <h1 className="mt-8 max-w-4xl text-4xl font-bold leading-[1.05] tracking-tight text-text-primary sm:text-6xl lg:text-7xl">
          Most stock predictors
          <br />
          are <span className="text-lime">lying to you.</span>
        </h1>

        <p className="mt-8 max-w-2xl text-lg leading-relaxed text-text-secondary">
          Not maliciously. Look-ahead bias is easy to introduce and hard to
          notice, and a model that has seen the future scores brilliantly right
          up until it meets a real market.
        </p>

        <p className="mt-5 max-w-2xl text-lg leading-relaxed text-text-secondary">
          StockIntel was built to find out whether machine learning can predict
          short-term equity direction, with the validation done properly. It
          reports what it found — including when the answer is{" "}
          <span className="font-semibold text-text-primary">no</span>.
        </p>

        <div className="mt-10 flex flex-wrap items-center gap-4">
          <Link
            href="/dashboard"
            className="rounded-lg bg-lime px-6 py-3 text-sm font-semibold text-ink-000 transition-opacity hover:opacity-90"
          >
            Open the dashboard
          </Link>
          <a
            href="https://github.com/zhaymn/StockMarketAnalysis#measured-results"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-lg border border-line-strong px-6 py-3 text-sm font-semibold text-text-primary transition-colors hover:border-lime hover:text-lime"
          >
            See the evidence
          </a>
        </div>
      </div>
    </header>
  );
}

function TheProblem() {
  const traps = [
    {
      title: "The scaler saw everything",
      body:
        "Fit a StandardScaler on the full series before splitting and the "
        + "training set now carries the test set's mean and variance. Small "
        + "leak, consistently flattering results.",
    },
    {
      title: "Overlapping labels",
      body:
        "With a 5-day horizon, the label at Monday depends on Friday's price. "
        + "Put Monday in train and Tuesday in test and the two sets share an "
        + "outcome. The usual symptom is a 5-day model that beats its 1-day "
        + "counterpart — backwards, since longer horizons are harder.",
    },
    {
      title: "Accuracy without a baseline",
      body:
        "\"54% directional accuracy\" sounds like signal. If the stock rose on "
        + "54% of days in the sample, it is precisely nothing.",
    },
    {
      title: "Probabilities that aren't",
      body:
        "A softmax output is a number between 0 and 1. That does not make it a "
        + "probability. Unless it is calibrated, \"68% confident\" is a decoration.",
    },
  ];

  return (
    <Section>
      <Eyebrow>The problem</Eyebrow>
      <Heading>Four ways to accidentally cheat</Heading>
      <Lead>
        None of these require dishonesty. Each is a default that looks
        reasonable, produces a better number, and quietly invalidates the
        result.
      </Lead>

      <div className="mt-12 grid gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-2">
        {traps.map((trap) => (
          <div key={trap.title} className="bg-ink-010 p-6 sm:p-8">
            <h3 className="text-base font-semibold text-text-primary">{trap.title}</h3>
            <p className="mt-3 text-sm leading-relaxed text-text-secondary">{trap.body}</p>
          </div>
        ))}
      </div>

      <p className="mt-10 max-w-2xl text-sm leading-relaxed text-text-muted">
        Published daily-direction accuracies of 70–90% are almost always one of
        the above. Real edges in liquid large-caps are small, fragile, and
        mostly gone after costs.
      </p>
    </Section>
  );
}

function WhatWeDidDifferently() {
  const measures = [
    {
      label: "Purged walk-forward",
      body:
        "Both a purge and an embargo, each the width of the label horizon, "
        + "around every test block. No training label can overlap a test-period "
        + "price.",
      test: "test_train_never_overlaps_test_label_window",
    },
    {
      label: "An empirical leak probe",
      body:
        "Features are recomputed from a truncated series and compared against "
        + "the full-history values. If any indicator read the future, the test "
        + "fails. Comments cannot catch this; a test can.",
      test: "test_features_have_no_lookahead",
    },
    {
      label: "A calibration gate",
      body:
        "A probability is displayed only if it beats the base rate on the "
        + "Brier score out-of-sample. Otherwise the interface says "
        + "PROBABILITY NOT CALIBRATED and shows direction alone.",
      test: "test_uncalibrated_probabilities_are_rejected",
    },
    {
      label: "Baselines everywhere",
      body:
        "Accuracy is never displayed without the naive baseline beside it, and "
        + "a skill score that reads zero when a model merely matches guessing "
        + "the majority class.",
      test: "test_skill_score_is_zero_for_majority_guessing",
    },
  ];

  return (
    <Section className="bg-ink-010">
      <Eyebrow>The method</Eyebrow>
      <Heading>Every guarantee is a test, not a claim</Heading>
      <Lead>
        Documentation drifts from code. These properties are enforced by tests
        that fail loudly when a guarantee breaks.
      </Lead>

      <div className="mt-12 space-y-4">
        {measures.map((measure) => (
          <div
            key={measure.label}
            className="rounded-xl border border-line bg-ink-000 p-6 sm:p-8"
          >
            <div className="flex flex-wrap items-baseline justify-between gap-4">
              <h3 className="text-base font-semibold text-lime">{measure.label}</h3>
              <code className="tnum text-xs text-text-faint">{measure.test}</code>
            </div>
            <p className="mt-3 max-w-3xl text-sm leading-relaxed text-text-secondary">
              {measure.body}
            </p>
          </div>
        ))}
      </div>
    </Section>
  );
}

function TheFinding() {
  const rows = [
    { model: "LightGBM", accuracy: "0.3836", skill: "−0.0496", won: "1/10", best: true },
    { model: "Majority baseline", accuracy: "0.3854", skill: "−0.0459", won: "0/10", best: false },
    { model: "Logistic Regression", accuracy: "0.3653", skill: "−0.0802", won: "0/10", best: false },
    { model: "Stacked ensemble", accuracy: "0.3383", skill: "−0.1296", won: "0/10", best: false },
  ];

  return (
    <Section>
      <Eyebrow>The finding</Eyebrow>
      <Heading>Nothing beat the baseline</Heading>
      <Lead>
        Five stocks, two horizons, ten years of history, ten configurations,
        identical protocol for every model. A skill score of zero means no
        better than always guessing the most common outcome.
      </Lead>

      <div className="mt-12 overflow-x-auto rounded-xl border border-line bg-ink-010">
        <table className="w-full min-w-[560px] text-sm">
          <thead>
            <tr className="border-b border-line text-left">
              <th className="px-6 py-4 font-medium text-text-muted">Model</th>
              <th className="px-6 py-4 text-right font-medium text-text-muted">Accuracy</th>
              <th className="px-6 py-4 text-right font-medium text-text-muted">Skill</th>
              <th className="px-6 py-4 text-right font-medium text-text-muted">Beat baseline</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.model} className="border-b border-line/60 last:border-0">
                <td className="px-6 py-4 text-text-primary">
                  {row.model}
                  {row.best && (
                    <span className="ml-2 rounded bg-lime-faint px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-lime">
                      best
                    </span>
                  )}
                </td>
                <td className="tnum px-6 py-4 text-right text-text-secondary">{row.accuracy}</td>
                <td className="tnum px-6 py-4 text-right font-semibold text-bearish">{row.skill}</td>
                <td className="tnum px-6 py-4 text-right text-text-muted">{row.won}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-12 grid gap-6 sm:grid-cols-3">
        <Stat
          value="−0.1086"
          label="LSTM mean skill"
          note="Last place, at 6× LightGBM's training cost. Deep learning is not automatically better."
        />
        <Stat
          value="10 / 10"
          label="Configurations where stacking hurt"
          note="The ensemble degraded ROC-AUC against its own best component every time."
        />
        <Stat
          value="0.52–0.59"
          label="ROC-AUC across models"
          note="Never 0.50. Weak ranking signal exists; the decision threshold destroys it."
        />
      </div>

      <div className="mt-12 rounded-xl border border-lime-dim/40 bg-lime-faint/40 p-6 sm:p-8">
        <p className="max-w-3xl text-base leading-relaxed text-text-primary">
          These unimpressive numbers are the evidence that the leak controls
          work. Had the purge and embargo been broken, this table would read
          65%+ — and it would be fiction.
        </p>
      </div>
    </Section>
  );
}

function Stat({
  value,
  label,
  note,
}: {
  value: string;
  label: string;
  note: string;
}) {
  return (
    <div className="rounded-xl border border-line bg-ink-010 p-6">
      <div className="tnum text-3xl font-bold text-lime">{value}</div>
      <div className="mt-2 text-sm font-medium text-text-primary">{label}</div>
      <p className="mt-2 text-xs leading-relaxed text-text-muted">{note}</p>
    </div>
  );
}

function WhatItActuallyDoes() {
  const features = [
    {
      title: "Two markets, no mixing",
      body:
        "US and India, each with its own currency, benchmark, exchange "
        + "timezone and trading-day count. Switching markets re-validates the "
        + "model on the new series.",
    },
    {
      title: "Predictions that decline to answer",
      body:
        "Where a model fails to beat its baseline, the dashboard issues no "
        + "directional call and shows the comparison that produced that verdict.",
    },
    {
      title: "Evidence beside every claim",
      body:
        "Walk-forward accuracy, calibration, per-regime breakdown and feature "
        + "importance — because a prediction without its error bars is a guess "
        + "in a suit.",
    },
    {
      title: "News that separates text from impact",
      body:
        "FinBERT scores the language; a relevance engine decides what it means "
        + "for the company. A competitor's factory fire is negative text and "
        + "potentially bullish impact.",
    },
    {
      title: "Missing data that looks missing",
      body:
        "Every unavailable metric renders as an em dash, never a zero. A "
        + "fabricated zero is worse than an honest gap.",
    },
    {
      title: "Charts that separate fact from forecast",
      body:
        "Candlesticks, moving averages, volume, RSI, MACD and rolling "
        + "volatility — all actual historical data, never interpolated.",
    },
  ];

  return (
    <Section className="bg-ink-010">
      <Eyebrow>The product</Eyebrow>
      <Heading>What it does with that honesty</Heading>
      <Lead>
        A null result is not a broken product. It is a product that tells you
        the truth, and gives you the analytics to reason for yourself.
      </Lead>

      <div className="mt-12 grid gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-2 lg:grid-cols-3">
        {features.map((feature) => (
          <div key={feature.title} className="bg-ink-000 p-6 sm:p-7">
            <h3 className="text-sm font-semibold text-text-primary">{feature.title}</h3>
            <p className="mt-3 text-sm leading-relaxed text-text-secondary">{feature.body}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}

function WhatItWillNotDo() {
  const limits = [
    ["Tell you what to buy", "It is not financial advice and holds no licence to give it."],
    ["Promise returns", "No model here demonstrated an out-of-sample edge. That is stated in the interface, not buried."],
    ["Claim real-time data", "The development feed is delayed. The UI labels data DELAYED, END OF DAY or CACHED — never REAL-TIME."],
    ["Invent news", "Without an API key the news section says NEWS API NOT CONFIGURED and shows setup steps. No placeholder articles."],
    ["Validate sentiment yet", "The pipeline is built and tested, but the free news tier serves 3 articles per request — too thin to validate. So no sentiment backtest is reported."],
    ["Account for costs", "Metrics are classification metrics, not trading returns. An edge this size would not survive slippage anyway."],
  ];

  return (
    <Section>
      <Eyebrow>The limits</Eyebrow>
      <Heading>What it will not do</Heading>
      <Lead>
        Stated here rather than in a footnote, because a product that hides its
        limitations has already told you something about its numbers.
      </Lead>

      <dl className="mt-12 divide-y divide-line overflow-hidden rounded-xl border border-line bg-ink-010">
        {limits.map(([title, body]) => (
          <div key={title} className="grid gap-2 p-6 sm:grid-cols-[minmax(0,1fr)_2fr] sm:gap-8 sm:p-7">
            <dt className="text-sm font-semibold text-text-primary">{title}</dt>
            <dd className="text-sm leading-relaxed text-text-secondary">{body}</dd>
          </div>
        ))}
      </dl>
    </Section>
  );
}

function ClosingCta() {
  return (
    <Section>
      <div className="mx-auto max-w-2xl text-center">
        <h2 className="text-3xl font-bold tracking-tight text-text-primary sm:text-4xl">
          Look at the evidence yourself
        </h2>
        <p className="mt-5 text-base leading-relaxed text-text-secondary">
          The dashboard runs against live market data. The fold-level results
          behind every figure on this page are committed to the repository, so
          you can check them rather than trust them.
        </p>
        <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
          <Link
            href="/dashboard"
            className="rounded-lg bg-lime px-6 py-3 text-sm font-semibold text-ink-000 transition-opacity hover:opacity-90"
          >
            Open the dashboard
          </Link>
          <a
            href="https://github.com/zhaymn/StockMarketAnalysis"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-lg border border-line-strong px-6 py-3 text-sm font-semibold text-text-primary transition-colors hover:border-lime hover:text-lime"
          >
            Read the source
          </a>
        </div>
      </div>
    </Section>
  );
}

function Footer() {
  return (
    <footer className="bg-ink-000">
      <div className="mx-auto max-w-6xl px-6 py-12 sm:px-8">
        <p className="max-w-3xl text-xs leading-relaxed text-text-faint">
          <span className="font-semibold text-text-muted">Not financial advice.</span>{" "}
          StockIntel is a research and analytics tool. Its predictions are
          statistical estimates from historical data, they are not guarantees,
          and they must never be the sole basis for an investment decision.
          Model performance measured on historical data does not imply future
          results. Market data is delayed.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-between gap-4 border-t border-line pt-8">
          <span className="text-xs text-text-faint">MIT licensed</span>
          <a
            href="https://github.com/zhaymn/StockMarketAnalysis"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-text-muted transition-colors hover:text-lime"
          >
            github.com/zhaymn/StockMarketAnalysis
          </a>
        </div>
      </div>
    </footer>
  );
}

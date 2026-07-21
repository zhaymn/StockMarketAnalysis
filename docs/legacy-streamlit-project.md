# Stock Price Prediction System

A production-structured deep learning system for forecasting stock closing prices from
historical market data, built with stacked LSTM networks (plus GRU, Bidirectional LSTM,
CNN-LSTM, and Attention-LSTM variants). Includes a full data pipeline, five interchangeable
model architectures, a rigorous evaluation suite (with a naive-baseline sanity check), and an
interactive Streamlit dashboard for training, evaluating, and forecasting on any ticker.

> **Disclaimer:** This project is for educational and portfolio purposes only. It is **not**
> financial advice, and its predictions should never be used as the sole basis for a real
> trading or investment decision. Stock prices are influenced by far more than their own
> price history — news, macroeconomics, and market sentiment are not modeled here.

---

## Table of Contents

- [Key Features](#key-features)
- [Markets & Stock Selection](#markets--stock-selection)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Mathematical Background: How LSTM Works](#mathematical-background-how-lstm-works)
- [Technical Indicators](#technical-indicators)
- [Evaluation Metrics](#evaluation-metrics)
- [Design Notes & Known Limitations](#design-notes--known-limitations)
- [Screenshots](#screenshots)
- [Deployment (Streamlit Community Cloud)](#deployment-streamlit-community-cloud)
- [Future Improvements](#future-improvements)

---

## Key Features

- **Two markets, dynamically loaded stock directories** — choose Indian or International
  equities from the sidebar, then search or browse a live-sourced list of real companies
  (not a hardcoded ticker list) — see [Markets & Stock Selection](#markets--stock-selection).
- **Leakage-free data pipeline** — chronological train/test splitting, scalers fit only on the
  training split, sliding-window sequence generation.
- **15 engineered technical-indicator features** — SMA, EMA, RSI, MACD, Bollinger Bands, OBV
  — computed from scratch with documented formulas (see [Technical Indicators](#technical-indicators)).
- **Five interchangeable architectures** — Stacked LSTM, Stacked GRU, Bidirectional LSTM,
  CNN-LSTM hybrid, and LSTM with a custom additive-attention pooling layer.
- **Full training pipeline** — EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard,
  CSV epoch logging, all wired through a Streamlit progress bar during interactive training.
- **Rigorous evaluation** — MAE, MSE, RMSE, MAPE, R², and Directional Accuracy, each with a
  plain-English explanation, plus a comparison against a naive "predict no change" baseline
  (stock prices are close to a random walk — beating that baseline is the real bar to clear).
- **Recursive multi-day forecasting** — a user-defined horizon, with the approximation it
  relies on clearly documented (see [Design Notes](#design-notes--known-limitations)).
- **Interactive Streamlit dashboard** — candlestick/line charts with indicator overlays,
  training curves, actual-vs-predicted plots, residual analysis, error histograms, forecast
  overlays, a 5-architecture comparison tool, and CSV export throughout.

## Markets & Stock Selection

The dashboard's sidebar workflow is: **pick a market → search or browse for a stock → fetch
data for that one stock**. Every downstream step (preprocessing, training, evaluation,
forecasting) then operates on exactly that ticker — a prediction is always for one company,
trained only on its own history, never mixed with any other company's data.

### Indian Market

NSE's own site (`nseindia.com`, `archives.nseindia.com`) blocks automated requests outright —
verified directly during development: HTTP 403 on the homepage and HTTP 503 on the official
equity-list CSV, even with a realistic browser User-Agent and a cookie-priming request first.
This is NSE's WAF/anti-bot protection, a widely reported real-world limitation of scripted NSE
access, not something fixable from application code (and it affects any server making the
request, not just this specific environment).

So the Indian provider (`src/markets/indian_market.py`) is built entirely on **Yahoo Finance's
live search API**, which does correctly index NSE-listed (and BSE-listed) equities:

- **Search** — type a company name or symbol; results come back live, filtered to NSE (`NSI`)
  and BSE exchange codes.
- **Starter directory** — a list of ~35 well-known large-cap NSE stocks shown before you type
  anything, built by querying that same live API for well-known company names (concurrently,
  cached for a day) — not a static ticker list baked into source.

### International Market

Unlike NSE, **NASDAQ Trader's symbol directory files are reliably scriptable** without
authentication (`nasdaqlisted.txt` and `otherlisted.txt`, verified working) — so the
International provider (`src/markets/international_market.py`) fetches the *entire* current
NASDAQ + NYSE/NYSE American/NYSE Arca listing (~7,000+ real companies) live on first load,
cached for a day. Live Yahoo Finance search is layered on top for anything outside that file
(newly listed symbols, etc.).

### Adding a new market

Adding London, Tokyo, Hong Kong, crypto, ETFs, mutual funds, or commodities as a future market
is two steps, and touches nothing else in the app:

1. Implement a `MarketProvider` subclass in `src/markets/` (see `src/markets/base.py`) with a
   `search()` method and, if a reliable bulk source exists, a `list_directory()` override.
2. Add one line to `MARKET_REGISTRY` in `src/markets/registry.py`.

The prediction pipeline (`src.preprocessing` → `src.dataset` → `src.model` → `src.train` →
`src.evaluate` → `src.predict`) never changes — it only ever consumes a plain yfinance-ready
ticker string, regardless of which provider produced it.

## Tech Stack

| Purpose | Library |
|---|---|
| Deep learning | TensorFlow / Keras 3 |
| Data handling | Pandas, NumPy |
| ML utilities (scaling, metrics) | scikit-learn |
| Market data | yfinance, Yahoo Finance search API, NASDAQ Trader symbol directory |
| HTTP requests | requests |
| Charting | Plotly (interactive), Matplotlib |
| Dashboard | Streamlit |
| Model/scaler persistence | Keras native `.keras` format, joblib |
| Language | Python 3.12+ |

## Project Structure

```
stock-price-predictor/
├── data/                    # Cached raw & processed CSVs (gitignored, regenerated on demand)
├── notebooks/                # Scratch space for exploration
├── models/                  # Trained models, scalers, metadata, history (gitignored)
├── logs/                    # App logs + TensorBoard event files (gitignored)
├── src/
│   ├── markets/
│   │   ├── base.py            # StockResult + MarketProvider abstraction
│   │   ├── yahoo_search.py    # Shared live Yahoo Finance search helper
│   │   ├── indian_market.py   # NSE/BSE provider (live search + starter directory)
│   │   ├── international_market.py  # NASDAQ/NYSE provider (bulk directory + search)
│   │   └── registry.py        # MARKET_REGISTRY -- add a market in two lines
│   ├── utils.py              # Paths, logging, exceptions, seeding, date/ticker helpers
│   ├── preprocessing.py      # yfinance download, missing-value handling, technical indicators
│   ├── dataset.py            # Scaling, sliding-window sequences, leakage-free train/test split
│   ├── model.py               # LSTM/GRU/BiLSTM/CNN-LSTM/Attention architectures
│   ├── train.py               # Training loop, callbacks, artifact persistence
│   ├── evaluate.py            # MAE/MSE/RMSE/MAPE/R²/Directional Accuracy + explanations
│   └── predict.py             # Next-day prediction, recursive multi-day forecasting
├── app.py                     # Streamlit dashboard (the main entry point)
├── requirements.txt
├── .gitignore
└── README.md
```

`model.keras`, `model_scalers.joblib`, and `model_metadata.json` appear at the project root
once you train a model through the dashboard — that's the app's "default" model, used
whenever no specific ticker/architecture is selected for loading.

## Installation

**Prerequisites:** Python 3.12+ and `pip`.

```bash
# 1. Clone the repository
git clone <your-repo-url> stock-price-predictor
cd stock-price-predictor

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

This installs TensorFlow, Streamlit, Plotly, scikit-learn, yfinance, and the rest of the
stack. On first run, TensorFlow prints a one-time warning that GPU support isn't available on
native Windows for TensorFlow >= 2.11 — this is expected; the project runs entirely on CPU
without issue for the model sizes involved here (use WSL2 or Linux/macOS if you want GPU
acceleration for larger architectures).

## Usage

### Running the dashboard (recommended)

```bash
streamlit run app.py
```

This opens the dashboard in your browser (default `http://localhost:8501`). Workflow:

1. **Sidebar** — pick a market (Indian or International), search for a company by name or
   symbol (or browse the default starter/bulk list), pick a history length and architecture,
   and hit **Fetch & Prepare Data**. A manual ticker-entry override is also available if you
   already know the exact symbol.
2. **Overview tab** — explore the price chart (candlestick or line), toggle SMA/EMA/Bollinger
   overlays, and inspect the RSI/MACD sub-charts.
3. **Train tab** — click **Train Model**. Watch live epoch progress, then review the
   training-vs-validation loss curve.
4. **Evaluate tab** — see MAE/RMSE/MAPE/R²/Directional Accuracy (hover each for an
   explanation), compared against a naive baseline, plus actual-vs-predicted, residual, and
   error-histogram charts.
5. **Forecast tab** — pick a horizon (1–90 trading days) and generate a recursive forecast
   overlaid on recent history.
6. **Compare Architectures tab** — train all five architectures on the same data and compare
   test-set error side by side.

Every chart's underlying data can be exported as CSV from the dashboard.

### Using the pipeline programmatically

```python
from src.preprocessing import load_and_prepare_data
from src.dataset import prepare_dataset
from src.model import build_lstm_model
from src.train import train_and_save, TrainingConfig
from src.predict import forecast, load_model_bundle

# 1. Fetch + engineer features for a ticker
df = load_and_prepare_data("AAPL", years=5)

# 2. Scale + window into train/test sequences
dataset = prepare_dataset(df, window_size=60, test_size=0.2)

# 3. Build and train a model
model = build_lstm_model((dataset.window_size, dataset.n_features))
result = train_and_save(
    "AAPL", model, dataset, model_name="lstm",
    config=TrainingConfig(epochs=100),
)

# 4. Forecast the next 14 trading days using the saved model
bundle = load_model_bundle("AAPL", "lstm")
forecast_df = forecast(df, bundle, horizon=14)
print(forecast_df)
```

### Looking up a ticker programmatically

```python
from src.markets.registry import MARKET_REGISTRY

indian = MARKET_REGISTRY["indian"]
for result in indian.search("Bharti Airtel", limit=3):
    print(result.label(), "|", result.exchange)
# BHARTI AIRTEL LIMITED (BHARTIARTL.NS) | NSE
# BHARTI AIRTEL LTD. (BHARTIARTL.BO) | Bombay
# ...

international = MARKET_REGISTRY["international"]
apple = international.search("Apple", limit=1)[0]
print(apple.symbol)  # "AAPL" -- feed this straight into load_and_prepare_data()
```

### TensorBoard

Training writes event files to `logs/tensorboard/`. To inspect a run:

```bash
tensorboard --logdir logs/tensorboard
```

## Architecture

**Pipeline:** `preprocessing.py` (download + clean + engineer features) → `dataset.py`
(scale + window into sequences, chronological split) → `model.py` (build architecture) →
`train.py` (fit with callbacks, persist artifacts) → `evaluate.py` (score against the held-out
test set) → `predict.py` (forecast forward using the trained model).

**Primary architecture — Stacked LSTM:**

```
Input (window_size, n_features)
   │
   ├─ LSTM(128, return_sequences=True) → Dropout(0.2)
   ├─ LSTM(64,  return_sequences=True) → Dropout(0.2)
   ├─ LSTM(32,  return_sequences=False) → Dropout(0.2)
   │
   ├─ Dense(25, activation="relu") → Dropout(0.1)
   └─ Dense(1, activation="linear")   # predicted next-day Close
```

Compiled with the **Adam** optimizer and **MSE** loss (MAE tracked as an additional metric).

**Architecture variants** (all built from one configurable factory in `src/model.py`, so they
share a single code path rather than duplicating layer-stacking logic):

| Variant | What changes |
|---|---|
| Stacked GRU | LSTM cells → GRU cells (fewer parameters, often comparable accuracy) |
| Bidirectional LSTM | Each recurrent layer reads the window forwards *and* backwards |
| CNN-LSTM Hybrid | A `Conv1D` + `MaxPooling1D` block extracts local patterns before the LSTM stack |
| LSTM + Attention | A custom Bahdanau-style additive-attention layer replaces "just use the last timestep" with a learned weighted sum over the whole window |

## Mathematical Background: How LSTM Works

A vanilla RNN updates a single hidden state `h_t = tanh(W_h·h_{t-1} + W_x·x_t + b)` at every
timestep. Backpropagating an error signal through many timesteps means repeatedly multiplying
by the same weight matrix, which causes gradients to either vanish toward zero or explode —
in practice, vanilla RNNs struggle to learn dependencies more than a few steps back, which is
a real problem for a 60-day price window.

**LSTM (Long Short-Term Memory)** solves this by adding a separate **cell state** `C_t` that
information can flow along almost unchanged, protected by three learned *gates* that decide
what to forget, what to add, and what to output. At each timestep `t`, given the previous
hidden state `h_{t-1}`, previous cell state `C_{t-1}`, and current input `x_t`:

**1. Forget gate** — how much of the old cell state to keep:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

**2. Input gate** — how much new information to write, and what that new information is:

```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```

**3. Cell state update** — combine the two: forget part of the old state, add part of the new
candidate:

```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
```

**4. Output gate** — how much of the (squashed) cell state becomes this timestep's hidden state:

```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

Where `σ` is the sigmoid function (squashes gate values to `[0, 1]`, acting as a soft
on/off switch), `tanh` squashes candidate values to `[-1, 1]`, `⊙` is elementwise
multiplication, and `[h_{t-1}, x_t]` denotes concatenation. Every `W` and `b` is learned
during training via backpropagation-through-time.

Because the forget/input gates are *additive* updates to `C_t` rather than repeated
matrix multiplications, gradients can flow back through many timesteps largely
unimpeded — this is what lets an LSTM (unlike a vanilla RNN) actually learn from a
60-day window of price history instead of effectively only "seeing" the last few days.

**In this project:** each `LSTM(units, return_sequences=True)` layer applies the above
recurrence at every one of the `window_size` timesteps and passes the full sequence of
`h_t` vectors to the next layer; the final `LSTM(units, return_sequences=False)` layer
(or the attention-pooling layer, in the Attention variant) collapses that sequence down
to a single vector, which the dense head maps to one predicted price. **GRU** (used in
the Stacked GRU variant) is a related, slightly simplified gating scheme that merges the
forget and input gates into a single "update gate" and drops the separate cell state —
fewer parameters, similar intuition.

## Technical Indicators

All computed in `src/preprocessing.py` from OHLCV data only, using strictly past-and-current
values (rolling/exponential windows) so no future information leaks into any row's features.

| Indicator | Formula | What it captures |
|---|---|---|
| **SMA** (Simple Moving Average) | `mean(Close[t-w+1..t])` | Smoothed trend over `w` days |
| **EMA** (Exponential Moving Average) | `α·Close_t + (1-α)·EMA_{t-1}`, `α = 2/(w+1)` | Trend, weighted toward recent prices |
| **RSI** (Relative Strength Index, Wilder's smoothing) | `100 - 100/(1 + avg_gain/avg_loss)` | Momentum; overbought (>70) / oversold (<30) |
| **MACD** | `EMA_12(Close) - EMA_26(Close)`, plus a 9-day EMA "signal" line | Trend-following momentum, and where it's shifting |
| **Bollinger Bands** | `SMA_20 ± 2·rolling_std_20` | Volatility envelope around the trend |
| **OBV** (On-Balance Volume) | Cumulative signed volume (add on up days, subtract on down days) | Whether volume is confirming the price trend |

## Evaluation Metrics

Computed in `src/evaluate.py` on the chronologically held-out test set (never seen during
training), after inverse-transforming predictions back to raw price scale.

| Metric | Meaning |
|---|---|
| **MAE** | Average absolute error, in the stock's own currency — "the model is off by about $X on a typical day." |
| **MSE** | Average squared error — penalizes large misses disproportionately; units are squared currency. |
| **RMSE** | `sqrt(MSE)`, back in raw currency units — RMSE noticeably larger than MAE signals a few large misses rather than uniform small error. |
| **MAPE** | Average error as a percentage of actual price — scale-independent, useful across tickers at very different price levels. |
| **R²** | Fraction of price variance the model explains; 1.0 is perfect, 0.0 matches "always predict the mean," negative is worse than that. |
| **Directional Accuracy** | % of days the model correctly predicted whether price would rise or fall vs. the previous actual close — often matters more than raw error for trading decisions. |

The dashboard also reports a **naive "predict no change" baseline** alongside every model's
metrics. Stock prices are close to a random walk, so a model needs to beat this trivial
baseline's RMSE to be adding real value — not just track yesterday's price. (Note: this
baseline's Directional Accuracy is mathematically always 0%, since "predict no change" never
commits to a direction — that's expected, not a sign the baseline is broken.)

## Design Notes & Known Limitations

- **Recursive forecasting is an approximation.** Beyond day 1, the model only ever predicts
  `Close`. Each recursive step approximates that day's Open/High/Low/Adj Close as equal to
  the predicted Close (a "flat bar") and Volume as the trailing 20-day average, then
  re-derives all technical indicators from that synthetic history before predicting the next
  step. This is a standard simplification in multi-step stock forecasting, but it means
  uncertainty compounds with horizon length — treat day 1 of a forecast as far more reliable
  than day 30.
- **No leakage, by construction.** Scalers are fit only on the training split; the
  chronological split never lets a test-period row influence a train-period prediction; the
  validation split Keras uses during training is the trailing slice of the *training* data
  only, never the test set.
- **This is a single-asset, price-history-only model.** It has no access to news, earnings,
  macroeconomic data, or broader market sentiment — all real drivers of price that this model
  cannot see.
- **The Indian stock directory is not NSE's complete official list.** NSE blocks automated
  access to its own bulk equity list (see [Markets & Stock Selection](#markets--stock-selection)
  for the verified details), so Indian coverage relies on live Yahoo Finance search rather than
  a downloaded complete directory. In practice this covers essentially any NSE/BSE stock you
  search for by name or symbol — it just isn't pre-loaded as one big browsable list the way the
  International market's ~7,000-stock NASDAQ/NYSE directory is.

## Screenshots

Screenshots aren't checked into this repository. To add your own for a portfolio README:

1. Run `streamlit run app.py`, walk through each tab, and use your OS/browser's screenshot
   tool (or Streamlit's own "Screenshot" option in its menu).
2. Save the images under a new `screenshots/` folder.
3. Replace this section with, e.g.:

```markdown
![Overview tab](screenshots/overview.png)
![Training tab](screenshots/train.png)
![Evaluation tab](screenshots/evaluate.png)
![Forecast tab](screenshots/forecast.png)
```

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (public, or private on a plan that supports it).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repository, branch, and set the main file path to `app.py`.
4. Streamlit Cloud installs `requirements.txt` automatically — no extra configuration needed
   for this project (it has no secrets or external services to configure).
5. Deploy. The first load will be slow while TensorFlow installs and the container spins up;
   subsequent loads are fast.

**Notes for cloud deployment:**
- Streamlit Community Cloud containers are ephemeral — anything written to `models/` or
  `data/` during a session will **not** persist across app restarts/redeploys. For a portfolio
  demo this is fine (train live in the session); for persistent model storage you'd want to
  wire in external storage (S3, a mounted volume, etc.), which is out of scope here.
  For serious use, keep long training runs on a proper CI/training pipeline rather than the
  free-tier serverless container.
- Free-tier containers have limited CPU/RAM — keep epoch counts and architecture sizes
  modest when demoing training live in a deployed instance (the Compare Architectures tab's
  "epochs per architecture" control exists partly for this reason).

## Future Improvements

- **More markets** — London Stock Exchange, Tokyo, Hong Kong, cryptocurrency, ETFs, mutual
  funds, commodities. Each is a new `MarketProvider` (see
  [Markets & Stock Selection](#markets--stock-selection)) plus one registry line; the
  prediction pipeline itself needs no changes.
- **A user-supplied NSE bulk CSV** — if you have (or can manually download from a browser) an
  official NSE equity-list export, `IndianMarketProvider.list_directory()` is the one place to
  wire it in as an additional source alongside the live-search-based starter list.
- **Automated hyperparameter search** — `src/model.py`'s `ModelConfig` dataclass is already
  structured as a tunable search space (recurrent units, dropout, learning rate, etc.); wiring
  in Keras Tuner or Optuna would be a natural next step. Not included here to avoid pulling in
  a heavy extra dependency for this iteration.
- **Walk-forward cross-validation** instead of a single chronological train/test split, for a
  more robust estimate of out-of-sample performance across different market regimes.
- **Multivariate forecasting** — jointly predicting Open/High/Low/Close/Volume instead of
  approximating them during recursive forecasting, which would remove the "flat bar"
  simplification entirely.
- **External data sources** — news sentiment, earnings calendars, macroeconomic indicators,
  sector/index correlation features.
- **Ensembling** — combining predictions across the five architectures (already trainable
  side by side via the Compare Architectures tab) rather than picking just one.
- **Model registry / experiment tracking** — MLflow or Weights & Biases integration for
  comparing runs beyond a single session.
- **Persistent storage for deployed models** — S3 or similar, so trained models survive
  container restarts on Streamlit Community Cloud.

"""Streamlit dashboard for the LSTM stock price prediction system.

Ties together every ``src`` module into an interactive workflow: fetch data
for any ticker -> explore it with technical-indicator charts -> train a
chosen architecture -> evaluate it against the held-out test set (and a
naive baseline) -> forecast future prices -> compare architectures head to
head. Run with ``streamlit run app.py``.
"""

from __future__ import annotations

import keras
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.dataset import inverse_transform_target, prepare_dataset
from src.evaluate import (
    METRIC_EXPLANATIONS,
    compute_residuals,
    evaluate_against_naive_baseline,
    evaluate_predictions,
)
from src.markets.base import StockResult
from src.markets.registry import MARKET_REGISTRY
from src.model import MODEL_REGISTRY, build_model_by_name
from src.predict import LoadedModelBundle, forecast, load_model_bundle
from src.preprocessing import load_and_prepare_data
from src.train import TrainingConfig, train_and_save, train_model
from src.utils import (
    DataDownloadError,
    InsufficientDataError,
    ModelNotFoundError,
    PERIOD_TO_YEARS,
    get_logger,
)
from theme import (
    AMBER,
    CYAN,
    ROSE,
    VIOLET,
    apply_chart_theme,
    glow_line,
    inject_theme,
    metric_card,
    render_app_header,
    render_metric_grid,
)

logger = get_logger(__name__)

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
inject_theme()

MODEL_LABELS = {
    "lstm": "Stacked LSTM",
    "gru": "Stacked GRU",
    "bidirectional_lstm": "Bidirectional LSTM",
    "cnn_lstm": "CNN-LSTM Hybrid",
    "attention_lstm": "LSTM + Attention",
}


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #


def init_session_state() -> None:
    """Register every session-state key this app reads, with its default value."""
    defaults = {
        "df_featured": None,
        "data_ticker": None,
        "dataset": None,
        "dataset_ticker": None,
        "training_result": None,
        "bundle": None,
        "bundle_label": None,
        "forecast_df": None,
        "comparison_results": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# --------------------------------------------------------------------------- #
# Streamlit-aware Keras callback (epoch progress in the UI, not just console)
# --------------------------------------------------------------------------- #


class StreamlitProgressCallback(keras.callbacks.Callback):
    """Reports epoch progress to a Streamlit progress bar and status line."""

    def __init__(self, progress_bar, status_text, total_epochs: int) -> None:
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        fraction = min((epoch + 1) / self.total_epochs, 1.0)
        self.progress_bar.progress(fraction)
        self.status_text.text(
            f"Epoch {epoch + 1}/{self.total_epochs}  |  "
            f"loss: {logs.get('loss', float('nan')):.4f}  |  "
            f"val_loss: {logs.get('val_loss', float('nan')):.4f}"
        )


# --------------------------------------------------------------------------- #
# Cached data loading (avoids re-downloading on every widget interaction)
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False, ttl=3600)
def cached_load_and_prepare_data(ticker: str, years: int) -> pd.DataFrame:
    return load_and_prepare_data(ticker, years=years, cache=True)


@st.cache_data(ttl=86400, show_spinner="Loading stock directory...")
def cached_market_directory(market_id: str) -> list[StockResult]:
    """Bulk/starter stock list for a market, cached for a day (this can involve
    several HTTP round-trips, e.g. NASDAQ's full symbol directory or the Indian
    provider's concurrent seed queries -- too slow to redo on every rerun)."""
    return MARKET_REGISTRY[market_id].list_directory()


@st.cache_data(ttl=300, show_spinner=False)
def cached_market_search(market_id: str, query: str) -> list[StockResult]:
    """Live search results for a query, cached briefly so retyping/backspacing
    doesn't re-hit the network for a query already seen in the last 5 minutes."""
    return MARKET_REGISTRY[market_id].search(query, limit=25)


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #


def bundle_from_training(dataset, model: keras.Model, ticker: str, model_name: str) -> LoadedModelBundle:
    """Build a ``LoadedModelBundle`` from an in-memory training run (no disk reload needed)."""
    return LoadedModelBundle(
        model=model,
        feature_scaler=dataset.feature_scaler,
        target_scaler=dataset.target_scaler,
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        window_size=dataset.window_size,
        ticker=ticker,
        model_name=model_name,
    )


def currency_symbol(ticker: str) -> str:
    return "₹" if ticker.endswith((".NS", ".BO")) else "$"


def download_button_for_df(df: pd.DataFrame, label: str, filename: str, key: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv().encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


# --------------------------------------------------------------------------- #
# Chart builders (all Plotly, all interactive)
# --------------------------------------------------------------------------- #


def build_price_chart(
    df: pd.DataFrame,
    ticker: str,
    show_candlestick: bool,
    show_sma: bool,
    show_ema: bool,
    show_bollinger: bool,
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
        vertical_spacing=0.03, subplot_titles=(f"{ticker} Closing Price", "Volume"),
    )

    if show_candlestick:
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                name="OHLC", increasing_line_color=CYAN, decreasing_line_color=ROSE,
            ),
            row=1, col=1,
        )
    else:
        for trace in glow_line(df.index, df["Close"], CYAN, "Close"):
            fig.add_trace(trace, row=1, col=1)

    if show_sma:
        for window, color in (("SMA_20", AMBER), ("SMA_50", VIOLET)):
            if window in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[window], mode="lines", name=window.replace("_", " "),
                               line=dict(width=1.3, color=color)),
                    row=1, col=1,
                )

    if show_ema and "EMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA_20"], mode="lines", name="EMA 20",
                       line=dict(width=1.3, dash="dot", color=AMBER)),
            row=1, col=1,
        )

    if show_bollinger and "BB_Upper" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_Upper"], mode="lines", name="Bollinger Upper",
                       line=dict(width=1, color="rgba(143,166,171,0.5)")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_Lower"], mode="lines", name="Bollinger Lower",
                       line=dict(width=1, color="rgba(143,166,171,0.5)"), fill="tonexty",
                       fillcolor="rgba(143,166,171,0.08)"),
            row=1, col=1,
        )

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="rgba(45,212,238,0.35)"), row=2, col=1)
    fig.update_layout(height=620, xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.08))
    return apply_chart_theme(fig)


def build_rsi_macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
        vertical_spacing=0.06, subplot_titles=("RSI (14)", "MACD"),
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], mode="lines", name="RSI 14", line=dict(color=VIOLET)), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=ROSE, opacity=0.5, row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=CYAN, opacity=0.5, row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD", line=dict(color=CYAN)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], mode="lines", name="Signal", line=dict(color=AMBER)), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram", marker_color="rgba(143,166,171,0.4)"), row=2, col=1)

    fig.update_layout(height=480, legend=dict(orientation="h", y=1.1))
    return apply_chart_theme(fig)


def build_loss_chart(history: dict[str, list[float]]) -> go.Figure:
    epochs = list(range(1, len(history["loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["loss"], mode="lines+markers", name="Training Loss", line=dict(color=CYAN)))
    if "val_loss" in history:
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines+markers", name="Validation Loss", line=dict(color=AMBER)))
    fig.update_layout(
        title="Training vs. Validation Loss (MSE)", xaxis_title="Epoch", yaxis_title="Loss",
        height=420, legend=dict(orientation="h", y=1.1),
    )
    return apply_chart_theme(fig)


def build_actual_vs_predicted_chart(dates, actual, predicted, ticker: str) -> go.Figure:
    fig = go.Figure()
    for trace in glow_line(dates, actual, CYAN, "Actual"):
        fig.add_trace(trace)
    fig.add_trace(go.Scatter(x=dates, y=predicted, mode="lines", name="Predicted", line=dict(color=AMBER, dash="dash", width=2)))
    fig.update_layout(
        title=f"{ticker}: Actual vs. Predicted Close (Test Set)", xaxis_title="Date", yaxis_title="Price",
        height=450, legend=dict(orientation="h", y=1.08),
    )
    return apply_chart_theme(fig)


def build_residual_chart(dates, residuals) -> go.Figure:
    fig = go.Figure()
    colors = [ROSE if r < 0 else CYAN for r in residuals]
    fig.add_trace(go.Bar(x=dates, y=residuals, marker_color=colors, name="Residual"))
    fig.add_hline(y=0, line_color="rgba(234,243,244,0.4)", opacity=0.6)
    fig.update_layout(
        title="Residuals (Actual - Predicted)", xaxis_title="Date", yaxis_title="Residual",
        height=380,
    )
    return apply_chart_theme(fig)


def build_error_histogram(residuals) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=30, marker_color=VIOLET))
    fig.update_layout(
        title="Distribution of Prediction Errors", xaxis_title="Residual", yaxis_title="Frequency",
        height=380,
    )
    return apply_chart_theme(fig)


def build_forecast_chart(history_df: pd.DataFrame, forecast_df: pd.DataFrame, ticker: str) -> go.Figure:
    recent_history = history_df.tail(90)
    fig = go.Figure()
    for trace in glow_line(recent_history.index, recent_history["Close"], CYAN, "Historical Close"):
        fig.add_trace(trace)

    bridge_x = [recent_history.index[-1], *forecast_df.index]
    bridge_y = [recent_history["Close"].iloc[-1], *forecast_df["Predicted_Close"]]
    for trace in glow_line(bridge_x, bridge_y, AMBER, "Forecast", dash="dash"):
        fig.add_trace(trace)
    fig.data[-1].mode = "lines+markers"

    fig.update_layout(
        title=f"{ticker}: {len(forecast_df)}-Day Forecast", xaxis_title="Date", yaxis_title="Price",
        height=480, legend=dict(orientation="h", y=1.08),
    )
    return apply_chart_theme(fig)


def build_comparison_chart(results: list[dict]) -> go.Figure:
    names = [MODEL_LABELS.get(r["architecture"], r["architecture"]) for r in results]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[r["rmse"] for r in results], name="RMSE", marker_color=CYAN))
    fig.add_trace(go.Bar(x=names, y=[r["mae"] for r in results], name="MAE", marker_color=AMBER))
    fig.update_layout(
        title="Model Comparison: Test-Set Error (lower is better)", barmode="group",
        yaxis_title="Price Error", height=420, legend=dict(orientation="h", y=1.1),
    )
    return apply_chart_theme(fig)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #


def render_sidebar() -> dict:
    st.sidebar.header("Configuration")

    st.sidebar.subheader("Market")
    market_ids = list(MARKET_REGISTRY.keys())
    market_id = st.sidebar.selectbox(
        "Market", market_ids, format_func=lambda mid: MARKET_REGISTRY[mid].market_label,
    )
    provider = MARKET_REGISTRY[market_id]

    st.sidebar.subheader("Stock")
    directory = cached_market_directory(market_id)

    search_query = st.sidebar.text_input(
        "Search by company name or symbol",
        placeholder="e.g. Bharti Airtel, or BHARTIARTL",
        key=f"search_{market_id}",
    ).strip()

    if search_query:
        candidates = cached_market_search(market_id, search_query)
        if not candidates:
            st.sidebar.caption("No matches. Try a different spelling, or enter the ticker manually below.")
    else:
        candidates = directory
        if not candidates:
            st.sidebar.caption(
                f"Could not load the {provider.market_label} stock directory right now "
                "(the data source may be temporarily unreachable) -- try searching above, "
                "or enter a ticker manually below."
            )

    label_to_result = {result.label(): result for result in candidates}
    selected_label = st.sidebar.selectbox(
        "Select stock",
        list(label_to_result.keys()) or ["(no results)"],
        key=f"select_{market_id}",
    )
    selected_result = label_to_result.get(selected_label)

    with st.sidebar.expander("Or enter a ticker manually"):
        manual_ticker = st.text_input(
            "Ticker symbol", value="", placeholder="e.g. AAPL or RELIANCE.NS", key=f"manual_{market_id}"
        ).strip().upper()

    ticker = manual_ticker or (selected_result.symbol if selected_result else "")

    period_label = st.sidebar.selectbox("Historical data range", list(PERIOD_TO_YEARS.keys()), index=2)
    years = PERIOD_TO_YEARS[period_label]

    if st.sidebar.button("Fetch & Prepare Data", width="stretch"):
        if not ticker:
            st.sidebar.error("Select a stock above (or enter a ticker manually) first.")
        else:
            with st.spinner(f"Downloading {ticker} ({period_label})..."):
                try:
                    df = cached_load_and_prepare_data(ticker, years)
                    st.session_state.df_featured = df
                    st.session_state.data_ticker = ticker
                    st.sidebar.success(f"Loaded {len(df)} rows for {ticker}.")
                except (DataDownloadError, InsufficientDataError) as exc:
                    st.sidebar.error(str(exc))
                except ValueError as exc:
                    st.sidebar.error(f"Invalid input: {exc}")

    st.sidebar.divider()
    st.sidebar.subheader("Model")
    model_name = st.sidebar.selectbox(
        "Architecture", list(MODEL_REGISTRY.keys()), format_func=lambda k: MODEL_LABELS.get(k, k),
    )
    window_size = st.sidebar.slider("Window size (days)", min_value=20, max_value=120, value=60, step=5)
    test_size = st.sidebar.slider("Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    with st.sidebar.expander("Advanced: training hyperparameters"):
        epochs = st.number_input("Max epochs", min_value=10, max_value=300, value=100, step=10)
        batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
        learning_rate = st.number_input("Learning rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
        dropout_rate = st.slider("Dropout rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)

    with st.sidebar.expander("Load a previously trained model"):
        load_ticker = st.text_input("Ticker", value=ticker, key="load_ticker_input").strip().upper()
        load_model_name = st.selectbox(
            "Architecture", list(MODEL_REGISTRY.keys()), format_func=lambda k: MODEL_LABELS.get(k, k),
            key="load_model_name_input",
        )
        if st.button("Load saved model", width="stretch"):
            try:
                st.session_state.bundle = load_model_bundle(load_ticker, load_model_name)
                st.session_state.bundle_label = f"{load_ticker} ({MODEL_LABELS.get(load_model_name, load_model_name)}) - loaded from disk"
                st.success(f"Loaded saved model for {load_ticker}.")
            except ModelNotFoundError as exc:
                st.error(str(exc))

    return {
        "ticker": ticker,
        "years": years,
        "model_name": model_name,
        "window_size": window_size,
        "test_size": test_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
    }


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #


def render_overview_tab(config: dict) -> None:
    df = st.session_state.df_featured
    if df is None:
        st.info("Use the sidebar to fetch data for a ticker before exploring charts.")
        return

    ticker = st.session_state.data_ticker
    st.caption(f"Showing {len(df)} trading days for **{ticker}**.")

    currency = currency_symbol(ticker)
    current_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    day_change_pct = (current_close - prev_close) / prev_close * 100
    render_metric_grid([
        metric_card("CURRENT CLOSE", f"{currency}{current_close:,.2f}",
                    f"{day_change_pct:+.2f}% vs prior close",
                    "positive" if day_change_pct >= 0 else "negative"),
        metric_card("PERIOD HIGH", f"{currency}{df['High'].max():,.2f}"),
        metric_card("PERIOD LOW", f"{currency}{df['Low'].min():,.2f}"),
        metric_card("LATEST VOLUME", f"{int(df['Volume'].iloc[-1]):,}"),
    ])

    cols = st.columns(4)
    show_candlestick = cols[0].checkbox("Candlestick", value=False)
    show_sma = cols[1].checkbox("SMA (20 / 50)", value=True)
    show_ema = cols[2].checkbox("EMA (20)", value=False)
    show_bollinger = cols[3].checkbox("Bollinger Bands", value=False)

    st.plotly_chart(
        build_price_chart(df, ticker, show_candlestick, show_sma, show_ema, show_bollinger),
        width="stretch",
    )
    st.plotly_chart(build_rsi_macd_chart(df), width="stretch")

    with st.expander("Raw feature-engineered data (tail)"):
        st.dataframe(df.tail(20), width="stretch")
    download_button_for_df(df, "Download full dataset (CSV)", f"{ticker}_processed.csv", key="download_overview_csv")


def render_train_tab(config: dict) -> None:
    df = st.session_state.df_featured
    if df is None:
        st.info("Fetch data first from the sidebar.")
        return

    ticker = st.session_state.data_ticker
    st.write(
        f"Train a **{MODEL_LABELS.get(config['model_name'], config['model_name'])}** model on "
        f"**{ticker}** with a {config['window_size']}-day window and "
        f"{config['test_size']:.0%} held out for testing."
    )

    if st.button("Train Model", type="primary"):
        try:
            dataset = prepare_dataset(df, window_size=config["window_size"], test_size=config["test_size"])
        except InsufficientDataError as exc:
            st.error(str(exc))
            return

        model = build_model_by_name(
            config["model_name"],
            (dataset.window_size, dataset.n_features),
            learning_rate=config["learning_rate"],
            dropout_rate=config["dropout_rate"],
        )

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        callback = StreamlitProgressCallback(progress_bar, status_text, config["epochs"])

        training_config = TrainingConfig(
            epochs=config["epochs"], batch_size=config["batch_size"], verbose=0,
        )

        with st.spinner("Training..."):
            result = train_and_save(
                ticker, model, dataset, model_name=config["model_name"],
                config=training_config, extra_callbacks=[callback],
            )

        status_text.text(f"Finished after {result.trained_epochs} epochs. Best val_loss: {result.best_val_loss:.6f}")

        st.session_state.dataset = dataset
        st.session_state.dataset_ticker = ticker
        st.session_state.training_result = result
        st.session_state.bundle = bundle_from_training(dataset, result.model, ticker, config["model_name"])
        st.session_state.bundle_label = f"{ticker} ({MODEL_LABELS.get(config['model_name'], config['model_name'])}) - just trained"
        st.success(f"Model trained and saved to {result.model_path}")

    result = st.session_state.training_result
    if result is not None:
        st.plotly_chart(build_loss_chart(result.history), width="stretch")
        history_df = pd.DataFrame(result.history)
        download_button_for_df(history_df, "Download training history (CSV)", f"{ticker}_history.csv", key="download_history_csv")


def render_evaluate_tab() -> None:
    result = st.session_state.training_result
    dataset = st.session_state.dataset
    if result is None or dataset is None:
        st.info("Train a model in the Train tab first (evaluation needs the held-out test set from that run).")
        return

    st.caption(f"Evaluating **{st.session_state.bundle_label}** on its held-out test set.")

    y_pred_scaled = result.model.predict(dataset.X_test, verbose=0).reshape(-1)
    y_pred = inverse_transform_target(dataset.target_scaler, y_pred_scaled)
    y_true = inverse_transform_target(dataset.target_scaler, dataset.y_test)
    dates = dataset.test_dates

    metrics = evaluate_predictions(y_true, y_pred)
    _, naive_metrics = evaluate_against_naive_baseline(y_true, y_pred)

    metric_specs = [
        ("MAE", metrics.mae, "mae", ""),
        ("RMSE", metrics.rmse, "rmse", ""),
        ("MAPE", metrics.mape, "mape", "%"),
        ("R2", metrics.r2, "r2", ""),
        ("DIRECTIONAL ACCURACY", metrics.directional_accuracy, "directional_accuracy", "%"),
    ]
    cards = [
        metric_card(label, f"{value:.4f}{unit}", help_text=METRIC_EXPLANATIONS[key])
        for label, value, key, unit in metric_specs
    ]
    render_metric_grid(cards)

    st.caption(
        f"Naive ('no change') baseline for comparison — RMSE: {naive_metrics.rmse:.4f}, "
        f"Directional Accuracy: {naive_metrics.directional_accuracy:.2f}%. "
        "Stock prices are close to a random walk, so beating this baseline on RMSE is the real bar to "
        "clear. Its Directional Accuracy is mathematically always 0% — 'predict no change' never commits "
        "to a direction, so it 'misses' on every single day by definition, not because it's a weak model."
    )

    st.plotly_chart(build_actual_vs_predicted_chart(dates, y_true, y_pred, st.session_state.dataset_ticker), width="stretch")

    residuals = compute_residuals(y_true, y_pred)
    res_col, hist_col = st.columns(2)
    with res_col:
        st.plotly_chart(build_residual_chart(dates, residuals), width="stretch")
    with hist_col:
        st.plotly_chart(build_error_histogram(residuals), width="stretch")

    results_df = pd.DataFrame(
        {"Date": dates, "Actual": y_true, "Predicted": y_pred, "Residual": residuals}
    ).set_index("Date")
    download_button_for_df(results_df, "Download test predictions (CSV)", "test_predictions.csv", key="download_eval_csv")


def render_forecast_tab() -> None:
    bundle = st.session_state.bundle
    df = st.session_state.df_featured
    if bundle is None or df is None:
        st.info("Train a model (or load a saved one) and fetch data first.")
        return

    st.caption(f"Forecasting with **{st.session_state.bundle_label}**.")

    horizon = st.slider("Forecast horizon (trading days)", min_value=1, max_value=90, value=7)
    if st.button("Generate Forecast", type="primary"):
        with st.spinner(f"Forecasting {horizon} trading day(s) ahead..."):
            try:
                forecast_df = forecast(df, bundle, horizon)
                st.session_state.forecast_df = forecast_df
            except ValueError as exc:
                st.error(str(exc))
                return

    forecast_df = st.session_state.forecast_df
    if forecast_df is not None:
        currency = currency_symbol(bundle.ticker)
        last_actual = float(df["Close"].iloc[-1])
        final_forecast = float(forecast_df["Predicted_Close"].iloc[-1])
        horizon_change_pct = (final_forecast - last_actual) / last_actual * 100
        render_metric_grid([
            metric_card("LAST ACTUAL CLOSE", f"{currency}{last_actual:,.2f}"),
            metric_card(f"FORECAST / {len(forecast_df)}D", f"{currency}{final_forecast:,.2f}",
                        f"{horizon_change_pct:+.2f}% vs last actual",
                        "positive" if horizon_change_pct >= 0 else "negative"),
            metric_card("MODEL", MODEL_LABELS.get(bundle.model_name, bundle.model_name)),
        ])
        st.plotly_chart(build_forecast_chart(df, forecast_df, bundle.ticker), width="stretch")
        st.dataframe(forecast_df, width="stretch")
        download_button_for_df(forecast_df, "Download forecast (CSV)", f"{bundle.ticker}_forecast.csv", key="download_forecast_csv")


def render_compare_tab(config: dict) -> None:
    df = st.session_state.df_featured
    if df is None:
        st.info("Fetch data first from the sidebar.")
        return

    st.write(
        "Train every registered architecture on the same data and compare test-set error. "
        "Runs in-memory only (not saved to disk) with a reduced epoch budget, so it stays fast; "
        "use the Train tab to persist your chosen architecture afterward."
    )
    comparison_epochs = st.number_input("Epochs per architecture", min_value=5, max_value=100, value=25, step=5)

    if st.button("Compare All Architectures", type="primary"):
        try:
            dataset = prepare_dataset(df, window_size=config["window_size"], test_size=config["test_size"])
        except InsufficientDataError as exc:
            st.error(str(exc))
            return

        results = []
        status_text = st.empty()
        overall_progress = st.progress(0.0)
        training_config = TrainingConfig(epochs=comparison_epochs, batch_size=config["batch_size"], verbose=0)

        for i, name in enumerate(MODEL_REGISTRY):
            status_text.text(f"Training {i + 1}/{len(MODEL_REGISTRY)}: {MODEL_LABELS.get(name, name)}...")
            model = build_model_by_name(name, (dataset.window_size, dataset.n_features))
            history = train_model(model, dataset, config=training_config)

            y_pred = inverse_transform_target(
                dataset.target_scaler, model.predict(dataset.X_test, verbose=0).reshape(-1)
            )
            y_true = inverse_transform_target(dataset.target_scaler, dataset.y_test)
            metrics = evaluate_predictions(y_true, y_pred)

            results.append({
                "architecture": name,
                "params": model.count_params(),
                "epochs_run": len(history.history["loss"]),
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "r2": metrics.r2,
                "directional_accuracy": metrics.directional_accuracy,
            })
            overall_progress.progress((i + 1) / len(MODEL_REGISTRY))

        status_text.text("Comparison complete.")
        st.session_state.comparison_results = results

    results = st.session_state.comparison_results
    if results:
        results_df = pd.DataFrame(results).rename(columns={
            "architecture": "Architecture", "params": "Parameters", "epochs_run": "Epochs Run",
            "rmse": "RMSE", "mae": "MAE", "r2": "R2", "directional_accuracy": "Directional Accuracy (%)",
        })
        results_df["Architecture"] = results_df["Architecture"].map(lambda k: MODEL_LABELS.get(k, k))
        st.dataframe(results_df.set_index("Architecture"), width="stretch")
        st.plotly_chart(build_comparison_chart(results), width="stretch")
        download_button_for_df(results_df.set_index("Architecture"), "Download comparison (CSV)", "model_comparison.csv", key="download_compare_csv")


# --------------------------------------------------------------------------- #
# Page layout
# --------------------------------------------------------------------------- #


init_session_state()

render_app_header(
    "Stock Price Prediction System",
    "LSTM-based deep learning for stock closing-price forecasting. Fetch real market data, "
    "train a recurrent architecture, evaluate it against a naive baseline, and forecast future prices.",
)

sidebar_config = render_sidebar()

tab_overview, tab_train, tab_evaluate, tab_forecast, tab_compare = st.tabs(
    ["Overview", "Train", "Evaluate", "Forecast", "Compare Architectures"]
)

with tab_overview:
    render_overview_tab(sidebar_config)

with tab_train:
    render_train_tab(sidebar_config)

with tab_evaluate:
    render_evaluate_tab()

with tab_forecast:
    render_forecast_tab()

with tab_compare:
    render_compare_tab(sidebar_config)

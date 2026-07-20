"""Cyan-black terminal theme: design tokens, injected CSS, and small UI helpers.

Kept separate from ``app.py`` so the visual system has one home. Nothing in
here touches business logic -- it only styles the existing Streamlit widgets
and Plotly figures that ``app.py`` already builds.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #

BG_PRIMARY = "#070B0C"
BG_SURFACE = "#0D1416"
BG_SURFACE_ALT = "#111B1E"
BORDER = "rgba(255,255,255,0.08)"
BORDER_STRONG = "rgba(45,212,238,0.35)"

CYAN = "#2DD4EE"
CYAN_DIM = "rgba(45,212,238,0.14)"
AMBER = "#F2A93B"
ROSE = "#F2495C"
VIOLET = "#8B7CF6"

TEXT_PRIMARY = "#EAF3F4"
TEXT_SECONDARY = "#8FA6AB"
TEXT_TERTIARY = "#57696D"

FONT_SANS = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
FONT_MONO = "'JetBrains Mono', 'IBM Plex Mono', 'SF Mono', monospace"

CHART_GRID = "rgba(255,255,255,0.05)"
CHART_LINE = "rgba(255,255,255,0.08)"


# --------------------------------------------------------------------------- #
# Global CSS injection
# --------------------------------------------------------------------------- #

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: {FONT_SANS};
}}

.stApp {{
    background:
        radial-gradient(ellipse 900px 500px at 0% 0%, rgba(45,212,238,0.07), transparent 60%),
        {BG_PRIMARY};
}}

/* ---- Typography ---- */
h1, h2, h3 {{
    font-family: {FONT_SANS};
    font-weight: 700;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.01em;
}}
p, span, label, .stMarkdown {{
    color: {TEXT_SECONDARY};
}}
code, .stCodeBlock, .stCaption {{
    font-family: {FONT_MONO} !important;
}}

/* ---- App header ---- */
.app-header {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 16px;
    margin-bottom: 4px;
    flex-wrap: wrap;
    gap: 10px;
}}
.app-header__eyebrow {{
    font-family: {FONT_MONO};
    font-size: 11px;
    letter-spacing: 0.16em;
    color: {CYAN};
    text-transform: uppercase;
    display: block;
    margin-bottom: 4px;
}}
.app-header__title {{
    font-size: 30px;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    margin: 0;
    line-height: 1.1;
}}
.app-header__subtitle {{
    font-size: 13px;
    color: {TEXT_SECONDARY};
    margin-top: 6px;
    max-width: 640px;
}}
.status-pill {{
    font-family: {FONT_MONO};
    font-size: 11px;
    letter-spacing: 0.1em;
    color: {CYAN};
    background: {CYAN_DIM};
    border: 1px solid {BORDER_STRONG};
    border-radius: 999px;
    padding: 6px 14px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
}}
.status-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: {CYAN};
    box-shadow: 0 0 0 0 rgba(45,212,238,0.6);
    animation: pulse-dot 2s infinite;
}}
@keyframes pulse-dot {{
    0% {{ box-shadow: 0 0 0 0 rgba(45,212,238,0.5); }}
    70% {{ box-shadow: 0 0 0 6px rgba(45,212,238,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(45,212,238,0); }}
}}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {{
    background: {BG_SURFACE};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] h3 {{
    font-family: {FONT_MONO};
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {TEXT_TERTIARY};
    border-bottom: 1px solid {BORDER};
    padding-bottom: 8px;
    margin-top: 18px;
    margin-bottom: 10px;
}}

/* ---- Tabs ---- */
[data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {BORDER};
}}
[data-baseweb="tab"] {{
    font-family: {FONT_MONO};
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_SECONDARY};
    height: 40px;
}}
[data-baseweb="tab"] p {{
    font-family: {FONT_MONO};
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
[aria-selected="true"] {{
    color: {CYAN} !important;
}}
[data-baseweb="tab-highlight"] {{
    background-color: {CYAN} !important;
}}

/* ---- Buttons ---- */
.stButton > button {{
    font-family: {FONT_SANS};
    font-weight: 600;
    border-radius: 8px;
    border: 1px solid {BORDER};
    background: {BG_SURFACE_ALT};
    color: {TEXT_PRIMARY};
    transition: border-color 0.15s ease, color 0.15s ease;
}}
.stButton > button:hover {{
    border-color: {BORDER_STRONG};
    color: {CYAN};
}}
.stButton > button[kind="primary"] {{
    background: {CYAN};
    color: #04191C;
    border: none;
}}
.stButton > button[kind="primary"]:hover {{
    background: #4FE0F5;
    color: #04191C;
}}

/* ---- Inputs ---- */
.stTextInput input, .stNumberInput input, [data-baseweb="select"] > div {{
    font-family: {FONT_MONO} !important;
    background: {BG_SURFACE_ALT} !important;
    border-color: {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT_PRIMARY} !important;
}}
.stTextInput input:focus, .stNumberInput input:focus {{
    border-color: {CYAN} !important;
    box-shadow: 0 0 0 1px {CYAN} !important;
}}

/* ---- Alerts (info/success/error) ---- */
[data-testid="stAlert"] {{
    background: {BG_SURFACE_ALT};
    border: 1px solid {BORDER};
    border-left: 3px solid {CYAN};
    border-radius: 8px;
}}

/* ---- Expanders ---- */
[data-testid="stExpander"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    background: {BG_SURFACE_ALT};
}}

/* ---- Dividers ---- */
hr {{
    border-color: {BORDER} !important;
}}

/* ---- Metric cards (custom component, not st.metric) ---- */
.metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 10px;
    margin: 14px 0;
}}
.metric-card {{
    background: {BG_SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 18px;
    transition: border-color 0.2s ease;
}}
.metric-card:hover {{
    border-color: {BORDER_STRONG};
}}
.metric-card__label {{
    font-family: {FONT_MONO};
    font-size: 10.5px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {TEXT_TERTIARY};
    margin-bottom: 8px;
}}
.metric-card__value {{
    font-family: {FONT_MONO};
    font-size: 26px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    line-height: 1.1;
}}
.metric-card__delta {{
    font-family: {FONT_MONO};
    font-size: 12px;
    margin-top: 6px;
}}
.metric-card__delta--positive {{ color: {CYAN}; }}
.metric-card__delta--negative {{ color: {ROSE}; }}
.metric-card__delta--neutral {{ color: {TEXT_SECONDARY}; }}
</style>
"""


def inject_theme() -> None:
    """Inject the global CSS. Call once, right after ``st.set_page_config``."""
    st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Header block
# --------------------------------------------------------------------------- #


def render_app_header(title: str, subtitle: str, eyebrow: str = "MARKET INTELLIGENCE TERMINAL") -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <span class="app-header__eyebrow">{eyebrow}</span>
                <div class="app-header__title">{title}</div>
                <div class="app-header__subtitle">{subtitle}</div>
            </div>
            <div class="status-pill"><span class="status-dot"></span>SYSTEM ONLINE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Metric cards
# --------------------------------------------------------------------------- #


def metric_card(label: str, value: str, delta: str | None = None, sentiment: str = "neutral", help_text: str | None = None) -> str:
    """Return the HTML for one metric card (caller composes these into a ``metric-grid`` div)."""
    delta_html = ""
    if delta:
        delta_html = f'<div class="metric-card__delta metric-card__delta--{sentiment}">{delta}</div>'
    title_attr = f' title="{help_text}"' if help_text else ""
    return (
        f'<div class="metric-card"{title_attr}>'
        f'<div class="metric-card__label">{label}</div>'
        f'<div class="metric-card__value">{value}</div>'
        f"{delta_html}"
        f"</div>"
    )


def render_metric_grid(cards_html: list[str]) -> None:
    st.markdown(f'<div class="metric-grid">{"".join(cards_html)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Plotly chart theming
# --------------------------------------------------------------------------- #


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply the dark cyan-black terminal look to any Plotly figure (incl. subplots)."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_MONO, color=TEXT_SECONDARY, size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SECONDARY, size=11)),
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(bgcolor=BG_SURFACE_ALT, font=dict(family=FONT_MONO, color=TEXT_PRIMARY), bordercolor=BORDER_STRONG),
    )
    fig.update_xaxes(gridcolor=CHART_GRID, zeroline=False, showline=True, linecolor=CHART_LINE)
    fig.update_yaxes(gridcolor=CHART_GRID, zeroline=False, showline=True, linecolor=CHART_LINE)
    for annotation in fig.layout.annotations or []:
        annotation.font = dict(family=FONT_SANS, color=TEXT_PRIMARY, size=13)
    return fig


def glow_line(x, y, color: str, name: str, dash: str | None = None, width: float = 2.2) -> tuple[go.Scatter, go.Scatter]:
    """A soft glow layer plus a crisp line on top, both the same trace data.

    Used sparingly (the one or two "hero" lines per chart) -- most series
    stay plain to keep the terminal readable rather than gaudy.
    """
    glow = go.Scatter(
        x=x, y=y, mode="lines", line=dict(color=color, width=width * 4),
        opacity=0.12, hoverinfo="skip", showlegend=False,
    )
    crisp = go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=width, dash=dash))
    return glow, crisp

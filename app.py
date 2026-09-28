import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Marketing Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# DESIGN TOKENS
# =========================================================
PALETTE = ["#60A5FA", "#FBBF24", "#34D399", "#F472B6", "#FB923C", "#A78BFA", "#22D3EE", "#94A3B8"]
PRIMARY = PALETTE[0]
RESP_COLORS = {"Responded": "#FBBF24", "No response": "#60A5FA"}

LABELS = {
    "Total_Spending": "Total spending ($)",
    "Income": "Income ($)",
    "Recency": "Recency (days)",
    "NumWebVisitsMonth": "Web visits per month",
    "NumWebPurchases": "Web purchases",
    "Response Rate": "Response rate (%)",
    "Acceptance Rate": "Acceptance rate (%)",
    "Accepted Customers": "Customers who accepted",
    "Spending": "Spending ($)",
}

# =========================================================
# STYLE  (glass cards, aurora glow, gradient accents, pill tabs)
# Icon fonts are never overridden, so Streamlit's own icons keep working.
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --accent: #60A5FA;
        --accent2: #A78BFA;
        --accent3: #F472B6;
        --surface: rgba(128,128,128,.09);
        --border: rgba(148,163,184,.26);
        --grad: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 60%, #EC4899 100%);
    }

    /* Font: set on containers only, never on icon elements */
    .stApp, .stMarkdown, [data-testid="stMarkdownContainer"], h1, h2, h3, button, input, label {
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    }
    [data-testid="stIconMaterial"], span[class*="material"] {
        font-family: "Material Symbols Rounded", "Material Icons" !important;
    }

    /* Aurora glow behind everything (transparent layers, works on any base colour) */
    .stApp {
        background-image:
            radial-gradient(900px 420px at 6% -8%, rgba(59,130,246,.22), transparent 60%),
            radial-gradient(800px 380px at 96% -6%, rgba(168,85,247,.20), transparent 60%),
            radial-gradient(700px 320px at 60% 105%, rgba(236,72,153,.10), transparent 60%);
        background-attachment: fixed;
    }
    .block-container {max-width: 1500px; padding-top: 1.6rem; padding-bottom: 2.5rem;}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {min-width: 290px; max-width: 320px;
        background-image: linear-gradient(180deg, rgba(59,130,246,.10), rgba(139,92,246,.06));
        border-right: 1px solid var(--border);}
    section[data-testid="stSidebar"] > div {overflow-x: hidden;}
    section[data-testid="stSidebar"] * {box-sizing: border-box;}

    /* Slider readability: neutral min/max numbers, accent thumb value */
    [data-testid="stTickBarMin"], [data-testid="stTickBarMax"],
    [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {color: inherit !important; opacity: .65;}
    [data-testid="stSliderThumbValue"] {color: var(--accent) !important; font-weight: 700;}
    div[data-baseweb="slider"] [role="slider"] {background-color: var(--accent) !important; box-shadow: 0 0 0 4px rgba(96,165,250,.25);}

    .side-summary {padding: 12px 14px; border: 1px solid var(--border); border-radius: 14px;
        background: var(--surface); margin-bottom: 8px;}
    .side-summary .ss-num {font-size: 1.5rem; font-weight: 800; letter-spacing: -.03em;}
    .side-summary .ss-of {opacity: .75; font-size: .85rem; margin-left: 6px;}
    .side-summary .ss-bar {height: 8px; border-radius: 999px; background: rgba(148,163,184,.25);
        margin: 8px 0 6px 0; overflow: hidden;}
    .side-summary .ss-fill {height: 100%; border-radius: 999px; background: var(--grad);}
    .side-summary .ss-sub {opacity: .75; font-size: .78rem;}

    details.about {border: 1px solid var(--border); border-radius: 12px; padding: 8px 12px;
        background: var(--surface); font-size: .85rem; margin-top: 14px;}
    details.about summary {cursor: pointer; font-weight: 600;}
    details.about ul {padding-left: 1.1rem; margin: .5rem 0 0 0;}
    details.about li {margin-bottom: 4px; opacity: .85;}

    /* ---------- Hero header ---------- */
    .hero {position: relative; padding: 26px 30px; border-radius: 22px; margin-bottom: 14px;
        border: 1px solid var(--border);
        background: linear-gradient(135deg, rgba(59,130,246,.16), rgba(139,92,246,.12) 55%, rgba(236,72,153,.10));
        overflow: hidden;}
    .hero::after {content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
        border-radius: 50%; background: radial-gradient(circle, rgba(167,139,250,.35), transparent 70%);}
    .hero-badge {display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .12em;
        text-transform: uppercase; padding: 4px 12px; border-radius: 999px; margin-bottom: 10px;
        border: 1px solid var(--border); background: var(--surface);}
    h1.dashboard-title {font-size: clamp(1.9rem, 3.6vw, 2.9rem); font-weight: 800; line-height: 1.1;
        letter-spacing: -.04em; margin: 0 0 .4rem 0 !important; padding: 0 !important;
        background: linear-gradient(90deg, #93C5FD, #C4B5FD 55%, #F9A8D4);
        -webkit-background-clip: text; background-clip: text; color: transparent;}
    .dashboard-subtitle {font-size: 1rem; opacity: .82; max-width: 900px; line-height: 1.6; margin: 0;}

    .section-label {font-size: .78rem; text-transform: uppercase; letter-spacing: .1em;
        font-weight: 700; opacity: .7; margin: .8rem 0 .5rem 0;}

    .chip {display: inline-block; padding: 4px 12px; margin: 0 6px 6px 0; border-radius: 999px;
        border: 1px solid var(--border); background: var(--surface); font-size: .82rem;}
    .chip.active {border-color: var(--accent); background: rgba(96,165,250,.14);}

    /* ---------- KPI cards ---------- */
    .kpi-card {position: relative; overflow: hidden; min-height: 104px; padding: 14px 16px;
        border: 1px solid var(--border); border-radius: 18px; background: var(--surface);
        margin-bottom: 10px; transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;}
    .kpi-card::before {content: ""; position: absolute; left: 0; right: 0; top: 0; height: 3px; background: var(--kpi);}
    .kpi-card:hover {transform: translateY(-3px); border-color: var(--kpi); box-shadow: 0 14px 30px rgba(0,0,0,.28);}
    .kpi-card.primary {min-height: 142px; padding: 18px 20px;}
    .kpi-top {display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px;}
    .kpi-label {font-size: .85rem; font-weight: 600; opacity: .85;}
    .kpi-value {font-size: clamp(1.2rem, 1.7vw, 1.55rem); line-height: 1.1; font-weight: 800;
        letter-spacing: -.03em; overflow-wrap: anywhere;}
    .kpi-card.primary .kpi-value {font-size: clamp(1.9rem, 2.8vw, 2.6rem);}
    .kpi-note {margin-top: 8px; font-size: .78rem; opacity: .75;}

    /* ---------- Insight cards ---------- */
    .insight-card {border: 1px solid var(--border); border-radius: 18px; padding: 18px 20px;
        margin-bottom: 14px; background: var(--surface); transition: transform .18s ease, border-color .18s ease;}
    .insight-card:hover {transform: translateY(-2px); border-color: var(--accent2);}
    .insight-head {display: flex; align-items: center; gap: 10px; margin-bottom: 10px;}
    .insight-tag {font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; opacity: .8;}
    .insight-finding {font-size: 1.02rem; font-weight: 650; line-height: 1.45; margin: 0 0 10px 0;}
    .insight-next {font-size: .9rem; opacity: .85; line-height: 1.55; margin: 0; padding-top: 10px;
        border-top: 1px dashed var(--border);}

    /* ---------- Empty state ---------- */
    .empty-state {text-align: center; border: 1px dashed var(--border); border-radius: 18px;
        padding: 44px 20px; margin: 1rem 0 .5rem 0; background: var(--surface);}
    .empty-state h3 {margin: 0 0 6px 0; padding: 0;}
    .empty-state p {opacity: .8; margin: 0;}

    /* ---------- Pill tabs ---------- */
    div[data-baseweb="tab-list"] {gap: 6px; padding: 6px; width: fit-content; max-width: 100%;
        overflow-x: auto; border-radius: 999px; border: 1px solid var(--border); background: var(--surface);}
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {display: none !important;}
    button[data-baseweb="tab"] {height: auto; min-height: 42px; padding: 8px 18px; border-radius: 999px;
        background: transparent; transition: background .18s ease;}
    button[data-baseweb="tab"]:hover {background: rgba(148,163,184,.16);}
    button[data-baseweb="tab"][aria-selected="true"] {background: var(--grad);}
    button[data-baseweb="tab"] p {font-weight: 600; margin: 0;}
    button[data-baseweb="tab"][aria-selected="true"] p {color: #fff !important;}

    /* ---------- Charts, tables, inputs as cards ---------- */
    div[data-testid="stPlotlyChart"] {background: var(--surface); border: 1px solid var(--border);
        border-radius: 18px; padding: 8px 8px 2px 8px;}
    div[data-testid="stDataFrame"] {border: 1px solid var(--border); border-radius: 14px; overflow: hidden;}
    div[data-testid="stHeading"] h2 {font-weight: 800; letter-spacing: -.025em;}
    span[data-baseweb="tag"] {background: rgba(96,165,250,.22) !important; border: 1px solid rgba(96,165,250,.45);}
    span[data-baseweb="tag"] span {color: inherit !important;}
    div[data-testid="stAlert"] {border-radius: 14px;}
    .stButton > button, .stDownloadButton > button {border-radius: 12px; font-weight: 600;
        transition: transform .15s ease, border-color .15s ease;}
    .stButton > button:hover:not(:disabled), .stDownloadButton > button:hover {transform: translateY(-1px);
        border-color: var(--accent);}

    .footer {text-align: center; opacity: .65; font-size: .8rem; padding: 10px 0 0 0;}

    /* Accessibility: visible focus + reduced motion */
    button:focus-visible, [role="tab"]:focus-visible, input:focus-visible, summary:focus-visible {
        outline: 2px solid var(--accent) !important; outline-offset: 2px;}
    @media (prefers-reduced-motion: reduce) {
        * {transition: none !important; animation: none !important;}
        .kpi-card:hover, .insight-card:hover {transform: none;}
    }
    @media (max-width: 900px) {
        .block-container {padding-left: 1rem; padding-right: 1rem;}
        .hero {padding: 20px;}
        .kpi-card.primary {min-height: 118px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DATA LOADING / CLEANING
# Keep definitions consistent with market_fixed_final.ipynb.
# =========================================================
@st.cache_data(show_spinner="Loading data and building customer segments…")
def load_data():
    candidates = [
        Path("data/marketing_campaign.csv"),
        Path("marketing_campaign.csv"),
        Path("marketing_campaign(1).csv"),
    ]
    data_path = next((p for p in candidates if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            "marketing_campaign.csv not found. Put it inside the data folder "
            "or beside app.py."
        )

    df = pd.read_csv(data_path, sep=";")

    # Same cleaning logic as the final notebook.
    df = df.drop_duplicates().copy()
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")

    median_income = df["Income"].median()
    df["Income"] = df["Income"].fillna(median_income)

    df = df[df["Year_Birth"] >= 1920].copy()
    df = df[df["Income"] <= 200000].copy()

    # Notebook uses the maximum registration year as the reference year.
    reference_year = int(df["Dt_Customer"].dt.year.max())
    df["Age"] = reference_year - df["Year_Birth"]

    spending_columns = [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]
    purchase_columns = [
        "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"
    ]
    campaign_columns = [
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
        "AcceptedCmp4", "AcceptedCmp5"
    ]

    df["Total_Spending"] = df[spending_columns].sum(axis=1)
    df["Total_Purchases"] = df[purchase_columns].sum(axis=1)
    df["Total_Campaign_Accepted"] = df[campaign_columns].sum(axis=1)
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    # K-Means: exactly the feature set documented in the notebook.
    segmentation_columns = [
        "Income", "Recency", "Total_Spending",
        "NumWebPurchases", "NumCatalogPurchases",
        "NumStorePurchases", "Total_Campaign_Accepted"
    ]
    X = df[segmentation_columns].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores = {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        silhouette_scores[k] = silhouette_score(X_scaled, labels)

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["Customer_Segment"] = final_model.fit_predict(X_scaled)

    return df, reference_year, best_k, silhouette_scores


try:
    df_all, reference_year, best_k, silhouette_scores = load_data()
except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.stop()

# One fixed colour per segment, so Segment 1 is the same colour in every chart.
SEG_COLORS = {
    f"Segment {s}": PALETTE[i % len(PALETTE)]
    for i, s in enumerate(sorted(df_all["Customer_Segment"].unique()))
}

# =========================================================
# HELPERS
# =========================================================
def pct(series):
    return float(series.mean() * 100) if len(series) else 0.0


def money(x):
    return f"${x:,.0f}"


def safe_mean(series):
    return float(series.mean()) if len(series) else 0.0


def empty_message(label):
    st.info(f"No data for {label} with the current filters. Try widening the filters.")


def seg_label(x):
    return f"Segment {x}"


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def with_response_label(df):
    """Readable categories instead of a 0-1 colour scale."""
    out = df.copy()
    out["Latest campaign"] = out["Response"].map({1: "Responded", 0: "No response"})
    return out


def show(fig, height=400, single_color=True):
    """Apply one consistent chart style, then render."""
    is_pie = any(t.type == "pie" for t in fig.data)
    legend_on = len(fig.data) > 1 and fig.layout.showlegend is not False

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=110 if is_pie else 10, t=56, b=80 if (legend_on and not is_pie) else 10),
        title=dict(x=0, xanchor="left", font=dict(size=16)),
    )
    if is_pie:
        fig.update_layout(legend=dict(orientation="v", yanchor="middle", y=0.5,
                                      xanchor="left", x=1.02, title_text=""))
    else:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.28,
                                      xanchor="center", x=0.5, title_text=""))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.22)")
    if single_color:
        fig.update_traces(marker_color=PRIMARY, selector=dict(type="bar"))
        fig.update_traces(marker_color=PRIMARY, selector=dict(type="histogram"))
        fig.update_traces(marker_color=PRIMARY, line_color=PRIMARY,
                          selector=dict(type="scatter", mode="lines+markers"))
    fig.update_traces(textposition="outside", cliponaxis=False, selector=dict(type="bar"))
    try:  # rounded bars (needs a recent Plotly; silently skipped otherwise)
        fig.update_traces(marker_cornerradius=8, selector=dict(type="bar"))
    except Exception:
        pass
    fig.update_traces(textinfo="percent", textfont_color="#0F172A", selector=dict(type="pie"))
    fig.update_traces(marker_opacity=0.75, selector=dict(type="scatter", mode="markers"))
    for ax in (fig.layout.xaxis, fig.layout.yaxis):
        if ax.title.text in LABELS:
            ax.title.text = LABELS[ax.title.text]
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.title("Filters")
summary_box = st.sidebar.container()   # filled after filtering

education_options = ["All"] + sorted(df_all["Education"].dropna().unique().tolist())
marital_options = ["All"] + sorted(df_all["Marital_Status"].dropna().unique().tolist())
segment_options = ["All"] + [seg_label(x) for x in sorted(df_all["Customer_Segment"].unique())]

age_min = int(max(18, np.floor(df_all["Age"].min())))
age_max = int(min(100, np.ceil(df_all["Age"].max())))
income_min = float(np.floor(df_all["Income"].min()))
income_max = float(np.ceil(df_all["Income"].max()))


def reset_filters():
    st.session_state["f_edu"] = "All"
    st.session_state["f_marital"] = "All"
    st.session_state["f_segment"] = "All"
    st.session_state["f_age"] = (age_min, age_max)
    st.session_state["f_income"] = (income_min, income_max)


st.sidebar.markdown('<div class="section-label">Customer profile</div>', unsafe_allow_html=True)
education_filter = st.sidebar.selectbox("Education", education_options, key="f_edu")
marital_filter = st.sidebar.selectbox("Marital status", marital_options, key="f_marital")
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max), key="f_age")

st.sidebar.markdown('<div class="section-label">Customer value</div>', unsafe_allow_html=True)
income_range = st.sidebar.slider(
    "Income range",
    min_value=income_min,
    max_value=income_max,
    value=(income_min, income_max),
    step=1000.0,
    format="$%.0f",
    key="f_income",
)

st.sidebar.markdown('<div class="section-label">Segment</div>', unsafe_allow_html=True)
segment_filter = st.sidebar.selectbox("Customer segment", segment_options, key="f_segment")

filtered = df_all.copy()
if education_filter != "All":
    filtered = filtered[filtered["Education"] == education_filter]
if marital_filter != "All":
    filtered = filtered[filtered["Marital_Status"] == marital_filter]
if segment_filter != "All":
    selected_segment = int(segment_filter.split()[-1])
    filtered = filtered[filtered["Customer_Segment"] == selected_segment]
filtered = filtered[filtered["Age"].between(age_range[0], age_range[1])]
filtered = filtered[filtered["Income"].between(income_range[0], income_range[1])]

active_filters = []
if education_filter != "All":
    active_filters.append(f"Education: {education_filter}")
if marital_filter != "All":
    active_filters.append(f"Marital status: {marital_filter}")
if segment_filter != "All":
    active_filters.append(f"{segment_filter}")
if age_range != (age_min, age_max):
    active_filters.append(f"Age: {age_range[0]}–{age_range[1]}")
if income_range != (income_min, income_max):
    active_filters.append(f"Income: {money(income_range[0])}–{money(income_range[1])}")

with summary_box:
    share = len(filtered) / len(df_all) if len(df_all) else 0.0
    st.markdown(
        f'<div class="side-summary">'
        f'<div><span class="ss-num">{len(filtered):,}</span><span class="ss-of">of {len(df_all):,} customers</span></div>'
        f'<div class="ss-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{share * 100:.0f}">'
        f'<div class="ss-fill" style="width:{share * 100:.1f}%"></div></div>'
        f'<div class="ss-sub">{share:.0%} of the dataset · updates every chart</div></div>',
        unsafe_allow_html=True,
    )
    st.button(
        "Reset filters",
        on_click=reset_filters,
        disabled=not active_filters,
        key="reset_sidebar",
        use_container_width=True,
    )

# Plain HTML <details>: no icon font needed (the old expander showed garbled text).
st.sidebar.markdown(
    f'<details class="about"><summary>About the data</summary><ul>'
    f'<li><b>Age</b> = {reference_year} minus birth year.</li>'
    f'<li><b>Segments</b> come from K-Means (K={best_k}, highest silhouette score).</li>'
    f'<li><b>Response</b> is the latest campaign response only.</li>'
    f'<li>Segment numbers are model labels, not named personas.</li>'
    f'</ul></details>',
    unsafe_allow_html=True,
)

# =========================================================
# HERO HEADER
# =========================================================
st.markdown(
    '<div class="hero">'
    '<div class="hero-badge">Live analytics</div>'
    '<h1 class="dashboard-title">Marketing Performance Dashboard</h1>'
    '<p class="dashboard-subtitle">Interactive analysis of customer demographics, spending behaviour, '
    'purchasing channels, campaign responses, engagement and customer segments.</p>'
    '</div>',
    unsafe_allow_html=True,
)

if active_filters:
    chips = "".join(f'<span class="chip active">{f}</span>' for f in active_filters)
    st.markdown(
        f'<div><span class="chip">Showing {len(filtered):,} of {len(df_all):,} customers</span>{chips}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div><span class="chip">All {len(df_all):,} customers · no filters applied</span></div>',
        unsafe_allow_html=True,
    )

# =========================================================
# EMPTY STATE (nothing matches the filters)
# =========================================================
if filtered.empty:
    st.markdown(
        '<div class="empty-state"><h3>No customers match these filters</h3>'
        '<p>This combination is too narrow. Widen the age or income range, '
        'or reset to see all customers.</p></div>',
        unsafe_allow_html=True,
    )
    left, mid, right = st.columns([2, 1, 2])
    with mid:
        st.button("Reset all filters", on_click=reset_filters, type="primary",
                  key="reset_main", use_container_width=True)
    st.stop()

# =========================================================
# KPI CARDS
# =========================================================
def render_kpi(label, value, note="", tip="", level="secondary", color="#60A5FA"):
    note_html = f'<div class="kpi-note">{note}</div>' if note else ''
    st.markdown(
        f'<div class="kpi-card {level}" style="--kpi:{color}; --kpi-soft:{rgba(color, .20)}" '
        f'title="{tip}" role="group" aria-label="{label}">'
        f'<div class="kpi-top"><div class="kpi-label">{label}</div></div>'
        f'<div class="kpi-value">{value}</div>{note_html}</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-label">Headline metrics</div>', unsafe_allow_html=True)
p1, p2, p3, p4 = st.columns(4)
total_spending = float(filtered["Total_Spending"].sum())
spending_display = f"${total_spending / 1_000_000:.2f}M" if total_spending >= 1_000_000 else money(total_spending)
with p1:
    render_kpi("Total Customers", f"{len(filtered):,}", "customers in current filter",
               "Number of customers after applying the filters.", "primary", "#60A5FA")
with p2:
    render_kpi("Total Spending", spending_display, f"Exact: {money(total_spending)}",
               "Sum of spending across all six product categories.", "primary", "#A78BFA")
with p3:
    render_kpi("Avg. Spending", money(safe_mean(filtered["Total_Spending"])), "per customer",
               "Total spending divided by number of customers.", "primary", "#F472B6")
with p4:
    render_kpi("Campaign Response", f"{pct(filtered['Response']):.2f}%", "latest campaign only",
               "Share of customers who responded to the latest campaign. Not a sales-conversion rate.",
               "primary", "#34D399")

st.markdown('<div class="section-label">Supporting metrics</div>', unsafe_allow_html=True)
s1, s2, s3, s4, s5 = st.columns(5)
acceptance_rate = (
    filtered["Total_Campaign_Accepted"].sum() / (len(filtered) * 5) * 100
    if len(filtered) else 0
)
with s1:
    render_kpi("Avg. Purchases", f"{safe_mean(filtered['Total_Purchases']):.1f}", "purchases per customer",
               "Web + catalog + store purchases per customer.", color="#FBBF24")
with s2:
    render_kpi("Avg. Income", money(safe_mean(filtered["Income"])), "average customer income",
               "Average annual income of customers in the filter.", color="#22D3EE")
with s3:
    render_kpi("Campaign Acceptance", f"{acceptance_rate:.2f}%", "across 5 campaigns",
               "Accepted offers divided by (customers x 5 campaigns).", color="#34D399")
with s4:
    render_kpi("Complaint Rate", f"{pct(filtered['Complain']):.2f}%", "customers reporting complaints",
               "Share of customers who filed a complaint.", color="#FB923C")
with s5:
    render_kpi("Avg. Recency", f"{safe_mean(filtered['Recency']):.1f} days", "since last purchase",
               "Average days since the customer's last purchase. Lower means more recent.",
               color="#A78BFA")

st.write("")

# =========================================================
# TABS
# =========================================================
tabs = st.tabs([
    "Overview",
    "Customer Analysis",
    "Spending & Channels",
    "Campaign Performance",
    "Segmentation",
    "Business Insights",
    "Data Explorer",
])

# =========================================================
# OVERVIEW
# =========================================================
with tabs[0]:
    st.header("Overview")
    st.caption("Who your customers are: education, marital status, age and income.")
    c1, c2 = st.columns(2)

    with c1:
        edu = filtered["Education"].value_counts().rename_axis("Education").reset_index(name="Customers")
        if len(edu):
            fig = px.bar(edu, x="Education", y="Customers", title="Customers by Education", text="Customers")
            fig.update_traces(texttemplate="%{y:,}")
            show(fig)
        else:
            empty_message("education")

    with c2:
        marital = filtered["Marital_Status"].value_counts().rename_axis("Marital Status").reset_index(name="Customers")
        if len(marital):
            fig = px.bar(marital, x="Marital Status", y="Customers", title="Customers by Marital Status", text="Customers")
            fig.update_traces(texttemplate="%{y:,}")
            show(fig)
        else:
            empty_message("marital status")

    c3, c4 = st.columns(2)
    with c3:
        age_counts = filtered[filtered["Age"].between(18, 100)]["Age"].value_counts().sort_index().reset_index()
        age_counts.columns = ["Age", "Customers"]
        if len(age_counts):
            fig = px.line(age_counts, x="Age", y="Customers", markers=True, title="Customer Age Distribution")
            show(fig)
    with c4:
        income_plot = filtered[["Income"]].copy()
        if len(income_plot):
            fig = px.histogram(income_plot, x="Income", nbins=30, title="Income Distribution")
            fig.update_yaxes(title="Customers")
            show(fig)

# =========================================================
# CUSTOMER ANALYSIS
# =========================================================
with tabs[1]:
    st.header("Customer Analysis")
    st.caption("How income, engagement and recency relate to spending. Colour shows latest campaign response.")

    c1, c2 = st.columns(2)
    with c1:
        sample = with_response_label(filtered.sample(min(len(filtered), 1500), random_state=42))
        fig = px.scatter(
            sample,
            x="Income",
            y="Total_Spending",
            color="Latest campaign",
            color_discrete_map=RESP_COLORS,
            hover_data=["Age", "Education", "Marital_Status", "Customer_Segment"],
            title="Income and Spending Relationship",
        )
        show(fig, height=460, single_color=False)

    with c2:
        spend_edu = filtered.groupby("Education", as_index=False)["Total_Spending"].mean().sort_values("Total_Spending", ascending=False)
        if len(spend_edu):
            fig = px.bar(spend_edu, x="Education", y="Total_Spending", title="Average Spending by Education", text_auto=",.0f")
            fig.update_yaxes(title="Average spending ($)")
            show(fig, height=460)

    st.subheader("Engagement Analysis")
    c3, c4 = st.columns(2)
    with c3:
        sample = with_response_label(filtered.sample(min(len(filtered), 1500), random_state=42))
        fig = px.scatter(
            sample,
            x="NumWebVisitsMonth",
            y="NumWebPurchases",
            color="Latest campaign",
            color_discrete_map=RESP_COLORS,
            hover_data=["Total_Spending", "Recency"],
            title="Web Visits vs Web Purchases",
        )
        show(fig, height=450, single_color=False)
    with c4:
        sample = with_response_label(filtered.sample(min(len(filtered), 1500), random_state=42))
        fig = px.scatter(
            sample,
            x="Recency",
            y="Total_Spending",
            color="Latest campaign",
            color_discrete_map=RESP_COLORS,
            hover_data=["NumWebVisitsMonth", "Total_Purchases"],
            title="Recency vs Total Spending",
        )
        show(fig, height=450, single_color=False)

    st.caption("These charts show observed relationships in the dataset; correlation does not imply causation. Charts plot a random sample of up to 1,500 customers.")

# =========================================================
# SPENDING & CHANNELS
# =========================================================
with tabs[2]:
    st.header("Spending & Purchase Channels")
    st.caption("Where customers spend and how they buy.")

    product_data = pd.DataFrame({
        "Product": ["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold"],
        "Spending": [
            filtered["MntWines"].sum(),
            filtered["MntFruits"].sum(),
            filtered["MntMeatProducts"].sum(),
            filtered["MntFishProducts"].sum(),
            filtered["MntSweetProducts"].sum(),
            filtered["MntGoldProds"].sum(),
        ],
    }).sort_values("Spending", ascending=False)

    channel_data = pd.DataFrame({
        "Channel": ["Web", "Catalog", "Store"],
        "Purchases": [
            filtered["NumWebPurchases"].sum(),
            filtered["NumCatalogPurchases"].sum(),
            filtered["NumStorePurchases"].sum(),
        ],
    }).sort_values("Purchases", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(product_data, x="Product", y="Spending", title="Total Spending by Product Category", text_auto=",.0f")
        show(fig, height=450)
    with c2:
        fig = px.bar(channel_data, x="Channel", y="Purchases", title="Purchases by Available Channel", text_auto=",.0f")
        show(fig, height=450)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.pie(product_data, names="Product", values="Spending", hole=0.45,
                     title="Share of Total Product Spending", color_discrete_sequence=PALETTE)
        show(fig, height=420, single_color=False)
    with c4:
        deal_counts = filtered["NumDealsPurchases"].value_counts().sort_index().reset_index()
        deal_counts.columns = ["Deal Purchases", "Customers"]
        fig = px.bar(deal_counts, x="Deal Purchases", y="Customers", title="Customers by Number of Deal Purchases")
        show(fig, height=420)

    st.info("Web, Catalog and Store are the purchase channels available in this dataset. They are not digital advertising channels such as Google Ads or Instagram.")

# =========================================================
# CAMPAIGN PERFORMANCE
# =========================================================
with tabs[3]:
    st.header("Marketing Campaign Performance")
    st.caption("Acceptance of the five previous campaigns, and response to the latest one.")

    campaign_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
    campaign_data = pd.DataFrame({
        "Campaign": [f"Campaign {i}" for i in range(1, 6)],
        "Accepted Customers": [int(filtered[c].sum()) for c in campaign_cols],
    })
    campaign_data["Acceptance Rate"] = campaign_data["Accepted Customers"] / len(filtered) * 100 if len(filtered) else 0

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(campaign_data, x="Campaign", y="Accepted Customers", text="Accepted Customers", title="Previous Campaign Acceptance")
        show(fig, height=430)
    with c2:
        fig = px.bar(campaign_data, x="Campaign", y="Acceptance Rate", text_auto=".2f", title="Campaign Acceptance Rate (%)")
        show(fig, height=430)

    response_by_edu = filtered.groupby("Education", as_index=False)["Response"].mean()
    response_by_edu["Response Rate"] = response_by_edu["Response"] * 100
    response_by_edu = response_by_edu.sort_values("Response Rate", ascending=False)

    c3, c4 = st.columns(2)
    with c3:
        response_counts = filtered["Response"].value_counts().rename(index={0: "No Response", 1: "Response"}).reset_index()
        response_counts.columns = ["Response", "Customers"]
        fig = px.bar(response_counts, x="Response", y="Customers", text="Customers",
                     title="Latest Campaign Response", color="Response",
                     color_discrete_map={"Response": RESP_COLORS["Responded"], "No Response": RESP_COLORS["No response"]})
        fig.update_traces(texttemplate="%{y:,}")
        fig.update_layout(showlegend=False)
        show(fig, height=410, single_color=False)
    with c4:
        if len(response_by_edu):
            fig = px.bar(response_by_edu, x="Education", y="Response Rate", text_auto=".2f", title="Latest Campaign Response Rate by Education")
            show(fig, height=410)

    st.info("The Response field represents the latest campaign response available in this dataset. It is not treated as a true advertising sales-conversion rate.")

# =========================================================
# SEGMENTATION
# =========================================================
with tabs[4]:
    st.header("Customer Segmentation (K-Means)")
    st.caption(
        "Segments group customers by Income, Recency, Total Spending, Web / Catalog / Store Purchases and "
        "Total Campaign Acceptance, standardized with StandardScaler. Each segment keeps the same colour in every chart."
    )

    c1, c2 = st.columns(2)
    with c1:
        seg_counts = filtered["Customer_Segment"].value_counts().sort_index().reset_index()
        seg_counts.columns = ["Segment", "Customers"]
        seg_counts["Segment"] = seg_counts["Segment"].apply(seg_label)
        fig = px.bar(seg_counts, x="Segment", y="Customers", text="Customers", title="Customer Segment Distribution",
                     color="Segment", color_discrete_map=SEG_COLORS)
        fig.update_traces(texttemplate="%{y:,}")
        fig.update_layout(showlegend=False)
        show(fig, single_color=False)
    with c2:
        seg_spend = filtered.groupby("Customer_Segment", as_index=False)["Total_Spending"].mean().sort_values("Total_Spending", ascending=False)
        seg_spend["Segment"] = seg_spend["Customer_Segment"].apply(seg_label)
        fig = px.bar(seg_spend, x="Segment", y="Total_Spending", text_auto=",.0f", title="Average Spending by Segment",
                     color="Segment", color_discrete_map=SEG_COLORS)
        fig.update_yaxes(title="Average spending ($)")
        fig.update_layout(showlegend=False)
        show(fig, single_color=False)

    profile = filtered.groupby("Customer_Segment").agg(
        Customers=("ID", "count"),
        Avg_Income=("Income", "mean"),
        Avg_Recency=("Recency", "mean"),
        Avg_Spending=("Total_Spending", "mean"),
        Avg_Purchases=("Total_Purchases", "mean"),
        Campaign_Acceptance_Rate=("Total_Campaign_Accepted", lambda s: s.mean() / 5 * 100),
        Response_Rate=("Response", "mean"),
    ).reset_index()
    profile["Response_Rate"] *= 100
    profile["Segment"] = profile["Customer_Segment"].apply(seg_label)
    profile = profile[["Segment", "Customers", "Avg_Income", "Avg_Recency", "Avg_Spending", "Avg_Purchases", "Campaign_Acceptance_Rate", "Response_Rate"]]
    profile = profile.rename(columns={
        "Avg_Income": "Avg Income",
        "Avg_Recency": "Avg Recency (days)",
        "Avg_Spending": "Avg Spending",
        "Avg_Purchases": "Avg Purchases",
        "Campaign_Acceptance_Rate": "Campaign Acceptance",
        "Response_Rate": "Latest Response",
    })

    st.subheader("Segment Profile")
    st.dataframe(
        profile.style.format({
            "Avg Income": "${:,.0f}",
            "Avg Recency (days)": "{:.1f}",
            "Avg Spending": "${:,.0f}",
            "Avg Purchases": "{:.1f}",
            "Campaign Acceptance": "{:.2f}%",
            "Latest Response": "{:.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    c3, c4 = st.columns(2)
    with c3:
        seg_response = filtered.groupby("Customer_Segment", as_index=False)["Response"].mean()
        seg_response["Response Rate"] = seg_response["Response"] * 100
        seg_response["Segment"] = seg_response["Customer_Segment"].apply(seg_label)
        fig = px.bar(seg_response, x="Segment", y="Response Rate", text_auto=".2f", title="Latest Campaign Response by Segment",
                     color="Segment", color_discrete_map=SEG_COLORS)
        fig.update_layout(showlegend=False)
        show(fig, single_color=False)
    with c4:
        seg_purchases = filtered.groupby("Customer_Segment", as_index=False)["Total_Purchases"].mean()
        seg_purchases["Segment"] = seg_purchases["Customer_Segment"].apply(seg_label)
        fig = px.bar(seg_purchases, x="Segment", y="Total_Purchases", text_auto=".1f", title="Average Purchases by Segment",
                     color="Segment", color_discrete_map=SEG_COLORS)
        fig.update_yaxes(title="Average purchases")
        fig.update_layout(showlegend=False)
        show(fig, single_color=False)

    st.caption(f"The tested K range was 2–8. The dashboard selects K={best_k} because it has the highest silhouette score, matching the notebook's selection rule.")

# =========================================================
# BUSINESS INSIGHTS
# =========================================================
with tabs[5]:
    st.header("Business Insights & Recommendations")
    st.caption("Calculated live from the filtered data; nothing here is hardcoded.")

    product_top = product_data.iloc[0]
    channel_top = channel_data.iloc[0]
    campaign_top = campaign_data.sort_values("Accepted Customers", ascending=False).iloc[0]
    segment_spend = filtered.groupby("Customer_Segment")["Total_Spending"].mean()
    top_segment = int(segment_spend.idxmax())
    top_segment_value = float(segment_spend.max())
    segment_response = filtered.groupby("Customer_Segment")["Response"].mean()
    response_segment = int(segment_response.idxmax())
    response_segment_rate = float(segment_response.max() * 100)
    web_corr = filtered[["NumWebVisitsMonth", "NumWebPurchases"]].corr().iloc[0, 1] if len(filtered) > 1 else np.nan

    insights = [
        (
            "Product Spending",
            f"{product_top['Product']} has the highest total spending at {money(product_top['Spending'])}.",
            "Use the strongest observed product category as an evidence-based focus area when discussing product-level marketing opportunities."
        ),
        (
            "Purchase Channel",
            f"{channel_top['Channel']} has the highest purchase volume with {channel_top['Purchases']:,.0f} purchases.",
            "Prioritize analysis of the highest-volume purchase channel and compare its customer behaviour with the other available channels."
        ),
        (
            "Campaign Performance",
            f"Campaign {campaign_top['Campaign'].split()[-1]} has the highest previous-campaign acceptance with {campaign_top['Accepted Customers']:,.0f} accepted customers ({campaign_top['Acceptance Rate']:.2f}%).",
            "Use the observed campaign acceptance differences to identify which previous campaigns deserve closer analysis."
        ),
        (
            "Customer Segment",
            f"Segment {top_segment} has the highest average spending at {money(top_segment_value)} per customer.",
            "Profile this segment's purchase and campaign behaviour before designing targeted customer strategies."
        ),
        (
            "Campaign Response Segment",
            f"Segment {response_segment} has the highest latest-campaign response rate at {response_segment_rate:.2f}%.",
            "Compare this segment's characteristics with other segments to understand patterns associated with campaign response."
        ),
    ]

    if pd.notna(web_corr):
        direction = "positive" if web_corr > 0 else "negative" if web_corr < 0 else "very weak"
        insights.append((
            "Engagement Relationship",
            f"Web visits and web purchases have a {direction} correlation of {web_corr:.2f} in the filtered data.",
            "Treat this as an observed association, not proof that web visits cause purchases."
        ))

    for i in range(0, len(insights), 2):
        cols = st.columns(2)
        for col, (title, finding, recommendation) in zip(cols, insights[i:i + 2]):
            with col:
                st.markdown(
                    f"<div class='insight-card'>"
                    f"<div class='insight-head'><div class='insight-tag'>{title}</div></div>"
                    f"<p class='insight-finding'>{finding}</p>"
                    f"<p class='insight-next'><b>Recommendation:</b> {recommendation}</p></div>",
                    unsafe_allow_html=True,
                )

    st.subheader("Dataset Limitations")
    st.info(
        "This dataset does not contain advertising impressions, ad clicks, advertising cost, "
        "customer reach or advertising revenue. Therefore genuine CTR, advertising conversion "
        "rate and ROI cannot be calculated directly. The dashboard uses valid customer, purchase "
        "and campaign-response KPIs instead of inventing unavailable metrics."
    )

# =========================================================
# DATA EXPLORER
# =========================================================
with tabs[6]:
    st.header("Data Explorer")
    st.caption(f"Showing {len(filtered):,} customers after applying all filters. Click a column header to sort.")

    display_columns = [
        "ID", "Year_Birth", "Age", "Education", "Marital_Status", "Income",
        "Recency", "Total_Spending", "Total_Purchases", "NumWebVisitsMonth",
        "Response", "Total_Campaign_Accepted", "Customer_Segment"
    ]
    column_labels = {
        "ID": st.column_config.NumberColumn("ID", format="%d"),
        "Year_Birth": st.column_config.NumberColumn("Birth year", format="%d"),
        "Marital_Status": st.column_config.Column("Marital status"),
        "Income": st.column_config.NumberColumn("Income", format="$%d"),
        "Recency": st.column_config.NumberColumn("Recency (days)"),
        "Total_Spending": st.column_config.NumberColumn("Total spending", format="$%d"),
        "Total_Purchases": st.column_config.NumberColumn("Total purchases"),
        "NumWebVisitsMonth": st.column_config.NumberColumn("Web visits / month"),
        "Response": st.column_config.NumberColumn("Latest response (1 = yes)"),
        "Total_Campaign_Accepted": st.column_config.NumberColumn("Campaigns accepted"),
        "Customer_Segment": st.column_config.NumberColumn("Segment"),
    }

    chosen = st.multiselect("Columns to show", display_columns, default=display_columns)
    if chosen:
        view = filtered[chosen]
        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
            column_config={k: v for k, v in column_labels.items() if k in chosen},
        )
        st.download_button(
            "Download filtered data (CSV)",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="filtered_customers.csv",
            mime="text/csv",
        )
    else:
        st.info("Select at least one column to display the table.")

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.markdown(
    '<div class="footer">Marketing Performance Dashboard • Built with Streamlit, Pandas, Plotly and Scikit-learn • '
    'Analysis aligned with the final market_fixed_final.ipynb notebook.</div>',
    unsafe_allow_html=True,
)

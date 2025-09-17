# ------------------------------------------------------------
# Retail Sales Forecasting Dashboard (Prophet & SARIMA)
# Feature parity with notebook + polished UI
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from io import StringIO
from pathlib import Path
import traceback

# Your utility module from earlier steps
from src_utils import (
    prepare_daily_sales,
    rolling_backtest,
    residual_bootstrap_quantiles,
    pinball_loss,
)

# ----------------------------
# Page config & light theming
# ----------------------------
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle CSS polish (cards, headings, metric spacing)
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    h1, h2, h3 { font-weight: 700; }
    .small-muted { color:#6b7280; font-size:0.9rem; }
    .card { background: #ffffff; border-radius: 16px; padding: 1rem 1.25rem; 
            border: 1px solid #e5e7eb; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
    .metric-row { margin-top: .25rem; margin-bottom: .25rem; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Retail Sales Forecasting")
st.caption("Prophet & SARIMA with rolling backtests and probabilistic forecasts (P10/P50/P90)")

# ----------------------------
# Sidebar controls (UX first)
# ----------------------------
with st.sidebar:
    st.header("Controls")
    st.markdown('<div class="small-muted">Set modeling and backtest parameters</div>', unsafe_allow_html=True)

    # Data loading
    use_sample = st.button("📥 Use sample (../data/train.csv)")
    uploaded = st.file_uploader("Or upload CSV", type=["csv"])

    st.markdown("---")
    st.subheader("Backtest")
    horizon = st.number_input("Forecast horizon (days)", 7, 180, 28, step=7)
    initial = st.number_input("Initial window (days)", 60, 2000, 365, step=7)
    step    = st.number_input("Step (days)", 7, 180, 28, step=7)

    run_both = st.checkbox("Compare Prophet & SARIMA", value=False)
    model = st.selectbox("Model (if not comparing)", ["prophet", "sarima"], index=0, disabled=run_both)

    st.markdown("---")
    st.subheader("Probabilistic")
    nsims = st.slider("Bootstrap simulations", 100, 2000, 400, step=100)
    holdout = st.number_input("Holdout length (days)", 7, 180, min(28, int(horizon)), step=7)

# ----------------------------
# Load data (robust)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_csv_from_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(StringIO(b.decode("utf-8", errors="ignore")))

@st.cache_data(show_spinner=False)
def load_sample_df() -> pd.DataFrame:
    for p in [Path("../data/train.csv"), Path("data/train.csv"), Path("../../data/train.csv")]:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError("Sample not found at ../data/train.csv (or data/train.csv).")

df = None
if use_sample and uploaded is None:
    try:
        df = load_sample_df()
        st.success("Loaded sample dataset from ../data/train.csv")
    except Exception as e:
        st.error(f"Failed to load sample: {e}")
elif uploaded is not None:
    try:
        df = load_csv_from_bytes(uploaded.read())
        st.success("Loaded dataset from upload.")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")

if df is None:
    st.info("Upload a CSV or click **Use sample** to proceed.")
    st.stop()

# ----------------------------
# DATA tab: Preview, schema, missing
# ----------------------------
tab_data, tab_forecast, tab_backtest, tab_prob, tab_downloads = st.tabs(
    ["📄 Data", "📊 Daily Series", "🧪 Backtesting", "🎯 Probabilistic", "⬇️ Exports"]
)

with tab_data:
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(15), use_container_width=True)

    # Column mapping
    st.markdown("### Column Mapping")
    date_candidates = [c for c in df.columns if "date" in c.lower()] or list(df.columns)
    num_candidates  = df.select_dtypes(include=[np.number]).columns.tolist() or list(df.columns)

    c1, c2, c3 = st.columns([1,1,1])
    date_col = c1.selectbox("Date column", options=date_candidates, index=0)
    value_col = c2.selectbox("Value column (numeric)", options=num_candidates,
                             index=(num_candidates.index("Sales") if "Sales" in num_candidates else 0))
    agg_mode = c3.selectbox("Aggregate", options=["sum", "mean"], index=0)

    # Info + missing
    st.markdown("### Schema & Missingness")
    info_df = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "missing": df.isna().sum().values,
        "non_null": df.notna().sum().values
    })
    st.dataframe(info_df, use_container_width=True)

# ----------------------------
# Prepare daily series
# ----------------------------
@st.cache_data(show_spinner=False)
def to_daily(df_in: pd.DataFrame, date_col: str, value_col: str, agg: str) -> pd.DataFrame:
    d = df_in.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce", infer_datetime_format=True)
    d = d.dropna(subset=[date_col])
    return prepare_daily_sales(d, date_col=date_col, value_col=value_col, agg=agg)

try:
    series = to_daily(df, date_col, value_col, agg_mode)
except Exception as e:
    st.error(f"Failed to aggregate to daily: {e}")
    st.stop()

with tab_forecast:
    st.subheader("Daily Series")
    st.markdown('<div class="small-muted">Aggregated by selected value and method</div>', unsafe_allow_html=True)
    st.dataframe(series.tail(20), use_container_width=True)

    # History chart
    hist = series.rename(columns={"ds": "date", "y": "sales"})
    hist_chart = (
        alt.Chart(hist)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("sales:Q", title="Sales"),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("sales:Q", format=",.2f")]
        )
        .properties(height=340)
    )
    st.altair_chart(hist_chart, use_container_width=True)

# ----------------------------
# Backtesting (Prophet/SARIMA)
# ----------------------------
@st.cache_data(show_spinner=True)
def run_backtest(series: pd.DataFrame, horizon: int, initial: int, step: int, model: str):
    return rolling_backtest(series, horizon=horizon, initial=initial, step=step, model=model)

with tab_backtest:
    st.subheader("Rolling Backtest")
    if len(series) <= (initial + horizon):
        st.warning("Not enough data for the chosen initial + horizon. Try a smaller window or horizon.")
    else:
        if run_both:
            cA, cB = st.columns(2)
            with st.spinner("Backtesting Prophet..."):
                bt_prophet = run_backtest(series, int(horizon), int(initial), int(step), "prophet")
            with st.spinner("Backtesting SARIMA..."):
                bt_sarima  = run_backtest(series, int(horizon), int(initial), int(step), "sarima")

            with cA:
                st.markdown("**Prophet metrics**")
                st.dataframe(bt_prophet["metrics"].round(3), use_container_width=True)
            with cB:
                st.markdown("**SARIMA metrics**")
                st.dataframe(bt_sarima["metrics"].round(3), use_container_width=True)

            # Combined metrics chart
            allm = []
            if not bt_prophet["metrics"].empty:
                allm.append(bt_prophet["metrics"].assign(model="prophet"))
            if not bt_sarima["metrics"].empty:
                allm.append(bt_sarima["metrics"].assign(model="sarima"))
            if allm:
                cm = pd.concat(allm, ignore_index=True)
                cm = cm.reset_index(drop=True).reset_index()
                cm_long = cm.melt(id_vars=["index", "fold_start", "fold_end", "model"],
                                  value_vars=["MAE", "RMSE", "sMAPE"],
                                  var_name="metric", value_name="value")
                chart = (
                    alt.Chart(cm_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("index:Q", title="Fold #"),
                        y="value:Q",
                        color="metric:N",
                        strokeDash="model:N",
                        tooltip=["model", "metric", "value", "fold_start", "fold_end"]
                    ).properties(height=320)
                )
                st.altair_chart(chart, use_container_width=True)

        else:
            with st.spinner(f"Backtesting {model}..."):
                bt = run_backtest(series, int(horizon), int(initial), int(step), model)
            st.dataframe(bt["metrics"].round(3), use_container_width=True)

            if not bt["metrics"].empty:
                cm = bt["metrics"].reset_index(drop=True).reset_index()
                cm_long = cm.melt(id_vars=["index", "fold_start", "fold_end", "model"],
                                  value_vars=["MAE", "RMSE", "sMAPE"],
                                  var_name="metric", value_name="value")
                chart = (
                    alt.Chart(cm_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("index:Q", title="Fold #"),
                        y="value:Q",
                        color="metric:N",
                        tooltip=["metric", "value", "fold_start", "fold_end"]
                    ).properties(height=320)
                )
                st.altair_chart(chart, use_container_width=True)

# ----------------------------
# Probabilistic forecast + Pinball loss
# ----------------------------
@st.cache_data(show_spinner=True)
def run_quantiles(history: pd.DataFrame, horizon: int, nsims: int):
    return residual_bootstrap_quantiles(history, horizon=int(horizon), n_sims=int(nsims), quantiles=[0.1, 0.5, 0.9])

with tab_prob:
    st.subheader("Probabilistic Forecast (P10 / P50 / P90)")
    if len(series) <= holdout:
        st.warning("Series is too short for the selected holdout. Reduce holdout days.")
    else:
        history = series.iloc[:-int(holdout)]
        test    = series.iloc[-int(holdout):]

        try:
            with st.spinner("Bootstrapping Prophet quantiles..."):
                qdf = run_quantiles(history, int(holdout), int(nsims))
        except Exception as e:
            st.error("Error generating quantiles. Prophet installed?\n\n" + traceback.format_exc())
            st.stop()

        merged = test.merge(
            qdf.rename(columns={"yhat": "p50", "q10": "p10", "q90": "p90"}),
            on="ds", how="left"
        )

        # KPIs: pinball loss
        pin10 = pinball_loss(merged["y"].values, merged["p10"].values, 0.10)
        pin50 = pinball_loss(merged["y"].values, merged["p50"].values, 0.50)
        pin90 = pinball_loss(merged["y"].values, merged["p90"].values, 0.90)

        k1, k2, k3 = st.columns(3)
        k1.metric("Pinball@10", f"{pin10:.3f}")
        k2.metric("Pinball@50", f"{pin50:.3f}")
        k3.metric("Pinball@90", f"{pin90:.3f}")

        # Band chart
        hist = series.rename(columns={"ds": "date", "y": "sales"})
        fut  = qdf.rename(columns={"ds": "date", "yhat": "p50", "q10": "p10", "q90": "p90"})

        base = (
            alt.Chart(hist)
            .mark_line()
            .encode(x=alt.X("date:T", title="Date"),
                    y=alt.Y("sales:Q", title="Sales"))
            .properties(height=340)
        )
        band = (
            alt.Chart(fut)
            .mark_area(opacity=0.18)
            .encode(x="date:T", y="p10:Q", y2="p90:Q", color=alt.value("#1f77b4"))
        )
        p50 = alt.Chart(fut).mark_line().encode(x="date:T", y="p50:Q", color=alt.value("#1f77b4"))
        st.altair_chart(base + band + p50, use_container_width=True)

        st.markdown("**Holdout Forecast Table**")
        out_csv = merged.rename(columns={"ds": "date", "y": "actual"})
        st.dataframe(out_csv, use_container_width=True)

# ----------------------------
# Downloads
# ----------------------------
with tab_downloads:
    st.subheader("Downloads")
    # Backtest metrics (single or both)
    if run_both:
        try:
            bt_prophet, bt_sarima  # noqa
            st.download_button(
                "⬇️ Prophet backtest metrics (CSV)",
                bt_prophet["metrics"].to_csv(index=False),
                file_name="prophet_backtest_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.download_button(
                "⬇️ SARIMA backtest metrics (CSV)",
                bt_sarima["metrics"].to_csv(index=False),
                file_name="sarima_backtest_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )
        except:
            st.info("Run backtests to enable metric downloads.")
    else:
        try:
            bt  # noqa
            st.download_button(
                f"⬇️ {model.upper()} backtest metrics (CSV)",
                bt["metrics"].to_csv(index=False),
                file_name=f"{model}_backtest_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )
        except:
            st.info("Run a backtest to enable metric downloads.")

    # Holdout forecast with quantiles
    try:
        st.download_button(
            "⬇️ Holdout forecast with P10/P50/P90 (CSV)",
            out_csv.to_csv(index=False),
            file_name="probabilistic_forecast_holdout.csv",
            mime="text/csv",
            use_container_width=True
        )
    except:
        st.info("Generate the probabilistic forecast to enable this download.")

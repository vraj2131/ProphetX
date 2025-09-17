import warnings
warnings.filterwarnings("ignore")

from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prophet import (works with both 'prophet' and legacy 'fbprophet')
try:
    from prophet import Prophet
except Exception:  # pragma: no cover
    from fbprophet import Prophet  # type: ignore


# ----------------------------
# Data preparation
# ----------------------------
def prepare_daily_sales(
    df: pd.DataFrame,
    date_col: str = "Order Date",
    value_col: str = "Sales",
    agg: str = "sum",
) -> pd.DataFrame:
    """
    Aggregate transactional data to a daily time series with columns: ds, y.
    Missing days are filled with 0 (sum) or forward fill (mean).
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])

    grp = d.groupby(pd.Grouper(key=date_col, freq="D"))[value_col]
    if agg == "sum":
        s = grp.sum()
        s = s.asfreq("D").fillna(0.0)
    elif agg == "mean":
        s = grp.mean()
        s = s.asfreq("D").fillna(method="ffill")
    else:
        s = grp.sum()
        s = s.asfreq("D").fillna(0.0)

    out = s.reset_index().rename(columns={date_col: "ds", value_col: "y"})
    return out


# ----------------------------
# Metrics
# ----------------------------
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss for quantile q in [0,1]. Lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


# ----------------------------
# Forecasting helpers
# ----------------------------
def residual_bootstrap_quantiles(
    history: pd.DataFrame,
    horizon: int,
    n_sims: int = 500,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    seed: int = 42,
    prophet_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Prophet-based forecast distribution via residual bootstrap.
    Returns DataFrame with columns: ds, yhat (median base), q10, q50, q90 (per `quantiles`).
    """
    prophet_kwargs = prophet_kwargs or dict(
        seasonality_mode="additive",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    rng = np.random.default_rng(seed)

    m = Prophet(**prophet_kwargs)
    m.fit(history)

    future = m.make_future_dataframe(periods=horizon, freq="D", include_history=False)

    # In-sample residuals (one-step)
    ins = m.make_future_dataframe(periods=0, freq="D", include_history=True)
    fc_ins = m.predict(ins)[["ds", "yhat"]].merge(history, on="ds", how="left")
    res = (fc_ins["y"] - fc_ins["yhat"]).dropna().values
    if len(res) == 0:
        res = np.array([0.0])

    base = m.predict(future)[["ds", "yhat"]].copy()
    sims = np.zeros((n_sims, len(base)), dtype=float)
    for i in range(n_sims):
        noise = rng.choice(res, size=len(base), replace=True)
        sims[i, :] = base["yhat"].values + noise

    q_cols = {}
    for q in quantiles:
        q_cols[f"q{int(q * 100)}"] = np.quantile(sims, q, axis=0)

    out = base.copy()
    for k, v in q_cols.items():
        out[k] = v
    return out


def sarimax_forecast(
    history: pd.DataFrame,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    horizon: int = 28,
    exog_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fit SARIMAX on history (ds,y) and forecast 'horizon' steps.
    Returns DataFrame with: ds, yhat, q10, q90 (from model conf_int).
    """
    y = history["y"].astype(float)
    exog = history[exog_cols] if exog_cols else None

    model = SARIMAX(
        endog=y,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    f_exog = None
    if exog_cols:
        # naive: reuse last-known exog for horizon; for real use, pass future exog
        f_exog = history[exog_cols].iloc[-horizon:]

    fc = res.get_forecast(steps=horizon, exog=f_exog)
    idx = pd.date_range(history["ds"].iloc[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")

    out = pd.DataFrame({"ds": idx, "yhat": fc.predicted_mean.values})
    conf = fc.conf_int()
    out["q10"] = conf.iloc[:, 0].values
    out["q90"] = conf.iloc[:, 1].values
    return out


# ----------------------------
# Rolling-origin backtest
# ----------------------------
def rolling_backtest(
    series: pd.DataFrame,
    horizon: int = 28,
    initial: int = 365,
    step: int = 28,
    model: str = "prophet",
    prophet_kwargs: Optional[Dict] = None,
    sarima_order=(1, 1, 1),
    sarima_seasonal=(1, 1, 1, 7),
) -> Dict[str, pd.DataFrame]:
    """
    Rolling-origin evaluation on a single time series (columns: ds, y).

    Returns:
      {
        "metrics": DataFrame[fold_start, fold_end, MAE, RMSE, sMAPE, model],
        "fold_preds": DataFrame[ds, y, yhat, fold, model]
      }
    """
    ds = series.sort_values("ds").reset_index(drop=True)
    prophet_kwargs = prophet_kwargs or dict(
        seasonality_mode="additive",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )

    metrics = []
    fold_preds = []
    start = initial

    while start + horizon <= len(ds):
        train = ds.iloc[:start].copy()
        test = ds.iloc[start : start + horizon].copy()

        if model.lower() == "prophet":
            m = Prophet(**prophet_kwargs)
            m.fit(train)
            future = m.make_future_dataframe(periods=horizon, freq="D", include_history=False)
            fc = m.predict(future)[["ds", "yhat"]]
        elif model.lower() == "sarima":
            fc = sarimax_forecast(
                train,
                order=sarima_order,
                seasonal_order=sarima_seasonal,
                horizon=horizon,
            )[["ds", "yhat"]]
        else:
            raise ValueError("model must be 'prophet' or 'sarima'")

        merged = test.merge(fc, on="ds", how="left")
        mae = mean_absolute_error(merged["y"], merged["yhat"])
        rmse = mean_squared_error(merged["y"], merged["yhat"], squared=False)
        smape = (100 / len(merged)) * np.sum(
            2 * np.abs(merged["y"] - merged["yhat"]) / (np.abs(merged["y"]) + np.abs(merged["yhat"]) + 1e-9)
        )

        metrics.append(
            {
                "model": model.lower(),
                "fold_start": merged["ds"].min(),
                "fold_end": merged["ds"].max(),
                "MAE": mae,
                "RMSE": rmse,
                "sMAPE": smape,
            }
        )
        fold_preds.append(merged.assign(fold=len(metrics), model=model.lower()))

        start += step

    metrics_df = pd.DataFrame(metrics)
    preds_df = pd.concat(fold_preds, ignore_index=True) if fold_preds else pd.DataFrame()
    return {"metrics": metrics_df, "fold_preds": preds_df}

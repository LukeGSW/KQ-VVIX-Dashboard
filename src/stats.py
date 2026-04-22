"""
stats.py — Statistiche descrittive, hit rate e profit factor sui rendimenti forward.

Metriche calcolate per ogni orizzonte N e tipo di segnale:
    N           : numero di eventi con dati validi (non-NaN)
    Media %     : rendimento medio aritmetico
    Mediana %   : rendimento mediano (50° percentile)
    P25 %       : 25° percentile
    P75 %       : 75° percentile
    Std Dev %   : deviazione standard campionaria
    Hit Rate %  : % di eventi con rendimento > 0 (direzione positiva)
    Profit Factor: somma(ret > 0) / |somma(ret < 0)|
                   > 1 indica edge statistico, = 1 è breakeven teorico

Nota sul Profit Factor: misura la "qualità" dell'edge indipendentemente
dalla frequenza. Un PF > 1 significa che i guadagni aggregati sui trade
vincenti superano le perdite aggregate sui trade perdenti.
"""

import numpy as np
import pandas as pd

from src.forward_returns import HORIZONS


def compute_summary_stats(
    fwd_returns: pd.DataFrame,
    signal_type: str,
    asset: str,
    horizons: list = None,
) -> pd.DataFrame:
    """
    Calcola statistiche descrittive per ogni orizzonte forward.

    Args:
        fwd_returns:  Output di compute_forward_returns.
        signal_type:  'overbought' | 'oversold'.
        asset:        'spx' | 'vix'.
        horizons:     Lista orizzonti in giorni (default: HORIZONS).

    Returns:
        pd.DataFrame con index = orizzonti (es. '5d', '10d', ...)
        e colonne: N, Media %, Mediana %, P25 %, P75 %, Std Dev %,
                   Hit Rate %, Profit Factor.
    """
    if horizons is None:
        horizons = HORIZONS

    subset = fwd_returns[fwd_returns["signal"] == signal_type]
    rows = []

    for h in horizons:
        col = f"{asset}_ret_{h}d"
        if col not in subset.columns:
            rows.append(_empty_row(f"{h}d"))
            continue

        series = subset[col].dropna()
        n = len(series)

        if n == 0:
            rows.append(_empty_row(f"{h}d"))
            continue

        pos_sum = series[series > 0].sum()
        neg_sum = abs(series[series < 0].sum())
        profit_factor = (pos_sum / neg_sum) if neg_sum > 0 else np.inf

        rows.append({
            "Orizzonte":     f"{h}d",
            "N":             n,
            "Media %":       round(series.mean(), 2),
            "Mediana %":     round(series.median(), 2),
            "P25 %":         round(series.quantile(0.25), 2),
            "P75 %":         round(series.quantile(0.75), 2),
            "Std Dev %":     round(series.std(ddof=1), 2),
            "Hit Rate %":    round((series > 0).mean() * 100, 1),
            "Profit Factor": round(profit_factor, 2),
        })

    return pd.DataFrame(rows).set_index("Orizzonte")


def _empty_row(label: str) -> dict:
    """Riga vuota (NaN) per orizzonti senza dati sufficienti."""
    return {
        "Orizzonte":     label,
        "N":             0,
        "Media %":       np.nan,
        "Mediana %":     np.nan,
        "P25 %":         np.nan,
        "P75 %":         np.nan,
        "Std Dev %":     np.nan,
        "Hit Rate %":    np.nan,
        "Profit Factor": np.nan,
    }

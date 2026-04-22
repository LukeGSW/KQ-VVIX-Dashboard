"""
conditional.py — Studio condizionato 2D: zScore VVIX × regime VIX.

L'idea: lo stesso estremo del VVIX si manifesta in contesti di volatilità
molto diversi. Stratificare gli eventi per livello del VIX al momento del
segnale rivela se l'edge è uniforme o concentrato in specifici regimi.

Bucket VIX:
    VIX < 15         : Bassa volatilità / complacenza di mercato
    15 ≤ VIX < 20    : Regime normale
    20 ≤ VIX < 30    : Volatilità elevata / attenzione
    VIX ≥ 30         : Alta stress / panico / tail risk

Per ogni combinazione (tipo segnale × bucket VIX × orizzonte) vengono
calcolati: rendimento medio %, N eventi, hit rate %.
"""

import numpy as np
import pandas as pd

from src.forward_returns import HORIZONS

# === DEFINIZIONE BUCKET VIX ===
VIX_BINS   = [0.0, 15.0, 20.0, 30.0, float("inf")]
VIX_LABELS = ["VIX < 15", "15 ≤ VIX < 20", "20 ≤ VIX < 30", "VIX ≥ 30"]


def assign_vix_regime(vix_series: pd.Series) -> pd.Series:
    """
    Classifica il livello VIX al momento del segnale in bucket di regime.

    Args:
        vix_series: pd.Series con i valori VIX ai giorni segnale.

    Returns:
        pd.Series (categorical ordinata) con etichette regime.
    """
    return pd.cut(
        vix_series,
        bins=VIX_BINS,
        labels=VIX_LABELS,
        right=False,      # intervallo chiuso a sinistra: [lower, upper)
        ordered=True,
    )


def compute_conditional_stats(
    fwd_returns: pd.DataFrame,
    signal_type: str,
    asset: str,
    horizons: list = None,
) -> pd.DataFrame:
    """
    Calcola rendimento medio % condizionato per ogni (regime VIX × orizzonte).

    Args:
        fwd_returns:  Output di compute_forward_returns.
                      Deve contenere la colonna 'vix_at_signal'.
        signal_type:  'overbought' | 'oversold'.
        asset:        'spx' | 'vix'.
        horizons:     Lista orizzonti in giorni (default: HORIZONS).

    Returns:
        pd.DataFrame con:
            index   = VIX_LABELS (regime VIX)
            colonne = ['N eventi'] + [f'{h}d' per h in horizons]
            valori  = rendimento medio % (arrotondato 2 decimali)
                      NaN se nessun evento nel bucket
    """
    if horizons is None:
        horizons = HORIZONS

    subset = fwd_returns[fwd_returns["signal"] == signal_type].copy()

    if subset.empty or "vix_at_signal" not in subset.columns:
        return _empty_conditional_df(horizons)

    subset["vix_regime"] = assign_vix_regime(subset["vix_at_signal"])

    rows = []
    for regime in VIX_LABELS:
        reg_data = subset[subset["vix_regime"] == regime]
        n = len(reg_data)
        row = {"Regime VIX": regime, "N eventi": n}

        for h in horizons:
            col = f"{asset}_ret_{h}d"
            if col in reg_data.columns and n > 0:
                vals = reg_data[col].dropna()
                row[f"{h}d"] = round(vals.mean(), 2) if len(vals) > 0 else np.nan
            else:
                row[f"{h}d"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).set_index("Regime VIX")


def compute_conditional_heatmap_data(
    fwd_returns: pd.DataFrame,
    signal_type: str,
    asset: str,
    horizons: list = None,
) -> pd.DataFrame:
    """
    Prepara la matrice (regime VIX × orizzonte) con solo rendimenti medi.
    Utilizzato direttamente dalla funzione build_conditional_heatmap in charts.py.

    Args:
        fwd_returns:  Output di compute_forward_returns.
        signal_type:  'overbought' | 'oversold'.
        asset:        'spx' | 'vix'.
        horizons:     Lista orizzonti (default: HORIZONS).

    Returns:
        pd.DataFrame (solo colonne orizzonti, senza 'N eventi').
    """
    if horizons is None:
        horizons = HORIZONS

    df = compute_conditional_stats(fwd_returns, signal_type, asset, horizons)
    horizon_cols = [f"{h}d" for h in horizons if f"{h}d" in df.columns]
    return df[horizon_cols]


def _empty_conditional_df(horizons: list) -> pd.DataFrame:
    """DataFrame vuoto con struttura corretta quando non ci sono eventi."""
    rows = [{"Regime VIX": r, "N eventi": 0, **{f"{h}d": np.nan for h in horizons}}
            for r in VIX_LABELS]
    return pd.DataFrame(rows).set_index("Regime VIX")

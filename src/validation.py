"""
validation.py — Robustezza delle regole: sensibilità alle soglie e walk-forward.

Due test che distinguono un edge reale da un overfitting:

  threshold_sensitivity():
      Ricalcola l'edge della regola al variare della soglia z (es. 2.0 → 3.0).
      Se l'edge esiste solo a un valore preciso (es. 2.5) e svanisce a 2.3/2.7,
      è curve-fitting; un edge robusto è stabile su un intorno di soglie.

  walk_forward_rule():
      Split temporale train/test (point-in-time: lo z-score usa sempre tutta la
      storia passata). Confronta la performance IN-SAMPLE vs OUT-OF-SAMPLE della
      regola: se l'edge OOS collassa, non è tradabile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals import compute_log_zscore, detect_events
from src.forward_returns import compute_forward_returns, HORIZONS


def _rule_subset(fwd: pd.DataFrame, signal: str, vix_min: float, vix_max: float) -> pd.DataFrame:
    if fwd.empty:
        return fwd
    return fwd[
        (fwd["signal"] == signal)
        & (fwd["vix_at_signal"] >= vix_min)
        & (fwd["vix_at_signal"] < vix_max)
    ]


def threshold_sensitivity(
    df: pd.DataFrame,
    signal: str,
    vix_min: float,
    vix_max: float,
    asset: str,
    horizon: int,
    z_grid: list[float],
    window: int,
    cooldown: int,
) -> pd.DataFrame:
    """
    Per ogni soglia z nella griglia, ricalcola eventi e l'edge della regola
    (N, media %, hit %) a un dato orizzonte. Mostra se l'edge è stabile.
    """
    z = compute_log_zscore(df["vvix"], window=window)
    rows = []
    for thr in z_grid:
        if signal == "overbought":
            ev = detect_events(z, lower_thresh=-99.0, upper_thresh=float(thr), cooldown=cooldown)
        else:
            ev = detect_events(z, lower_thresh=float(thr), upper_thresh=99.0, cooldown=cooldown)
        fwd = compute_forward_returns(df[["spx", "vix"]], ev, [horizon])
        sub = _rule_subset(fwd, signal, vix_min, vix_max)
        s = sub[f"{asset}_ret_{horizon}d"].dropna() if not sub.empty else pd.Series(dtype=float)
        rows.append({
            "Soglia z": round(float(thr), 2),
            "N":        len(s),
            "Media %":  round(s.mean(), 2) if len(s) else np.nan,
            "Hit %":    round((s > 0).mean() * 100, 1) if len(s) else np.nan,
        })
    return pd.DataFrame(rows)


def walk_forward_rule(
    df: pd.DataFrame,
    signal: str,
    vix_min: float,
    vix_max: float,
    asset: str,
    window: int,
    cooldown: int,
    upper: float,
    lower: float,
    horizons: list = None,
    split: float = 0.6,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Split temporale: gli eventi prima della data di taglio sono IN-SAMPLE, dopo
    OUT-OF-SAMPLE. Lo z-score è calcolato sull'intera storia (point-in-time), gli
    eventi vengono solo partizionati per data. Restituisce (tabella, data_taglio).
    """
    if horizons is None:
        horizons = HORIZONS

    z = compute_log_zscore(df["vvix"], window=window)
    ev = detect_events(z, lower_thresh=lower, upper_thresh=upper, cooldown=cooldown)
    fwd = compute_forward_returns(df[["spx", "vix"]], ev, horizons)
    sub = _rule_subset(fwd, signal, vix_min, vix_max)

    cut = df.index[int(len(df) * split)]
    rows = []
    for h in horizons:
        col = f"{asset}_ret_{h}d"
        is_s  = sub[sub.index <  cut][col].dropna() if not sub.empty else pd.Series(dtype=float)
        oos_s = sub[sub.index >= cut][col].dropna() if not sub.empty else pd.Series(dtype=float)
        rows.append({
            "Orizzonte": f"{h}d",
            "N IS":   len(is_s),
            "Media IS %":  round(is_s.mean(), 2) if len(is_s) else np.nan,
            "Hit IS %":    round((is_s > 0).mean() * 100, 1) if len(is_s) else np.nan,
            "N OOS":  len(oos_s),
            "Media OOS %": round(oos_s.mean(), 2) if len(oos_s) else np.nan,
            "Hit OOS %":   round((oos_s > 0).mean() * 100, 1) if len(oos_s) else np.nan,
        })
    return pd.DataFrame(rows).set_index("Orizzonte"), cut

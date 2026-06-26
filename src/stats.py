"""
stats.py — Statistiche descrittive, significatività e profit factor sui forward returns.

Metriche base per ogni orizzonte N e tipo di segnale:
    N, Media %, Mediana %, P25 %, P75 %, Std Dev %, Hit Rate %, Profit Factor.

Metriche di significatività (quando si passa un `baseline`):
    Baseline %  : rendimento forward INCONDIZIONATO (media su tutti i giorni) allo
                  stesso orizzonte → il drift di riferimento da battere.
    Excess %    : Media segnale − Baseline (l'edge vero, al netto del drift).
    CI low/high : intervallo di confidenza 95% della media (block-bootstrap).
    p-value     : test che l'EXCESS ≠ 0, con block-bootstrap che tiene conto della
                  SOVRAPPOSIZIONE dei forward returns (orizzonti > cooldown).
    Sig         : '✓' se p-value < 0.05.

Perché il block-bootstrap: con cooldown C e orizzonte H, fino a ⌈H/C⌉ eventi
consecutivi hanno finestre forward sovrapposte → non sono indipendenti. Il
bootstrap a blocchi di lunghezza ⌈H/C⌉ preserva questa dipendenza e produce
errori standard onesti (a differenza della media campionaria ingenua).
"""

import math

import numpy as np
import pandas as pd

from src.forward_returns import HORIZONS
from src.params import N_BOOT, BOOT_SEED


# ================================================================
# BASELINE INCONDIZIONATO + BLOCK BOOTSTRAP
# ================================================================

def baseline_forward_returns(prices: pd.DataFrame, horizons: list = None) -> dict:
    """
    Rendimento forward medio INCONDIZIONATO (su tutti i giorni) per asset/orizzonte.
    È il drift di riferimento: un segnale ha edge solo se batte questo valore.

    Returns: {'spx': {h: mean_pct, ...}, 'vix': {...}}
    """
    if horizons is None:
        horizons = HORIZONS
    out: dict[str, dict] = {}
    for asset in ("spx", "vix"):
        if asset not in prices.columns:
            continue
        p = prices[asset].to_numpy(dtype=float)
        out[asset] = {}
        for h in horizons:
            if len(p) > h:
                fr = (p[h:] - p[:-h]) / p[:-h] * 100.0
                out[asset][h] = float(np.nanmean(fr))
            else:
                out[asset][h] = np.nan
    return out


def block_bootstrap_mean(
    values: np.ndarray,
    block_len: int = 1,
    ref: float = 0.0,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float, float]:
    """
    Block-bootstrap (circolare) della media di `values`.

    Returns (mean, ci_low_95, ci_high_95, p_two_sided_vs_ref).
    Il p-value è la probabilità bootstrap che la media stia dalla parte sbagliata
    rispetto a `ref` (×2, two-sided). Con ref=baseline → test dell'EXCESS.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    obs = float(v.mean())
    if n == 1:
        return (obs, np.nan, np.nan, np.nan)

    bl = int(max(1, min(block_len, n)))
    n_blocks = int(np.ceil(n / bl))
    rng = np.random.default_rng(seed)

    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(bl)[None, None, :]).reshape(n_boot, -1) % n
    samp = v[idx][:, :n]
    means = samp.mean(axis=1)

    lo, hi = np.percentile(means, [2.5, 97.5])
    frac_le = float((means <= ref).mean())
    frac_ge = float((means >= ref).mean())
    p = min(1.0, 2.0 * min(frac_le, frac_ge))
    return (obs, float(lo), float(hi), float(p))


def _block_len_for(horizon_days: int, cooldown: int) -> int:
    """Blocchi che coprono la sovrapposizione: ⌈H / cooldown⌉ eventi."""
    return max(1, math.ceil(horizon_days / max(1, cooldown)))


# ================================================================
# RIGA STATISTICA (condivisa)
# ================================================================

def _stats_row(
    series: pd.Series,
    horizon_days: int,
    baseline_h: float | None,
    cooldown: int,
) -> dict:
    """Costruisce la riga statistica per un orizzonte; aggiunge i campi di
    significatività se è fornito `baseline_h`."""
    label = f"{horizon_days}d"
    s = series.dropna()
    n = len(s)
    if n == 0:
        return _empty_row(label, with_sig=baseline_h is not None)

    pos_sum = s[s > 0].sum()
    neg_sum = abs(s[s < 0].sum())
    profit_factor = (pos_sum / neg_sum) if neg_sum > 0 else np.inf

    row = {
        "Orizzonte":     label,
        "N":             n,
        "Media %":       round(s.mean(), 2),
        "Mediana %":     round(s.median(), 2),
        "P25 %":         round(s.quantile(0.25), 2),
        "P75 %":         round(s.quantile(0.75), 2),
        "Std Dev %":     round(s.std(ddof=1), 2) if n > 1 else np.nan,
        "Hit Rate %":    round((s > 0).mean() * 100, 1),
        "Profit Factor": round(profit_factor, 2),
    }

    if baseline_h is not None:
        mean, lo, hi, p = block_bootstrap_mean(
            s.to_numpy(), block_len=_block_len_for(horizon_days, cooldown),
            ref=baseline_h,
        )
        row.update({
            "Baseline %": round(baseline_h, 2),
            "Excess %":   round(mean - baseline_h, 2),
            "CI low %":   round(lo, 2) if not np.isnan(lo) else np.nan,
            "CI high %":  round(hi, 2) if not np.isnan(hi) else np.nan,
            "p-value":    round(p, 3) if not np.isnan(p) else np.nan,
            "Sig":        "✓" if (not np.isnan(p) and p < 0.05) else "",
        })

    return row


def compute_summary_stats(
    fwd_returns: pd.DataFrame,
    signal_type: str,
    asset: str,
    horizons: list = None,
    baseline: dict | None = None,
    cooldown: int = 20,
) -> pd.DataFrame:
    """
    Statistiche per orizzonte sui forward returns di un tipo di segnale.
    Se `baseline` è fornito (output di baseline_forward_returns), aggiunge le
    colonne di significatività (Excess/CI/p-value/Sig).
    """
    if horizons is None:
        horizons = HORIZONS
    subset = fwd_returns[fwd_returns["signal"] == signal_type]
    base_a = (baseline or {}).get(asset, {})

    rows = []
    for h in horizons:
        col = f"{asset}_ret_{h}d"
        series = subset[col] if col in subset.columns else pd.Series(dtype=float)
        rows.append(_stats_row(series, h, base_a.get(h) if baseline else None, cooldown))
    return pd.DataFrame(rows).set_index("Orizzonte")


def compute_subset_stats(
    subset: pd.DataFrame,
    asset: str,
    horizons: list = None,
    baseline: dict | None = None,
    cooldown: int = 20,
) -> pd.DataFrame:
    """
    Come compute_summary_stats ma su un sotto-campione già filtrato (es. una regola).
    """
    if horizons is None:
        horizons = HORIZONS
    base_a = (baseline or {}).get(asset, {})

    rows = []
    for h in horizons:
        col = f"{asset}_ret_{h}d"
        series = subset[col] if col in subset.columns else pd.Series(dtype=float)
        rows.append(_stats_row(series, h, base_a.get(h) if baseline else None, cooldown))
    return pd.DataFrame(rows).set_index("Orizzonte")


def _empty_row(label: str, with_sig: bool = False) -> dict:
    """Riga vuota (NaN) per orizzonti senza dati sufficienti."""
    row = {
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
    if with_sig:
        row.update({"Baseline %": np.nan, "Excess %": np.nan, "CI low %": np.nan,
                    "CI high %": np.nan, "p-value": np.nan, "Sig": ""})
    return row

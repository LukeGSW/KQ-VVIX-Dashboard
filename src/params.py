"""
params.py — Parametri operativi CONDIVISI (single source of truth).

Sia la dashboard (app.py) sia lo script live (telegram_alert.py) importano i
parametri e le definizioni delle regole da QUI. Evita che i valori letterali
(finestra, soglie, cooldown, bucket VIX) divergano tra backtest e segnale live
— la causa classica della rottura di parità.

Le 4 regole operative sono definite una sola volta (RULES) e usate sia per
classificare il segnale corrente sia per filtrare i sotto-campioni storici.
"""

from __future__ import annotations

import math

# === Parametri del segnale (devono coincidere tra dashboard e live) ===
WINDOW: int = 90            # finestra rolling log-zScore VVIX
UPPER_THRESH: float = 2.5   # soglia overbought
LOWER_THRESH: float = -2.0  # soglia oversold
COOLDOWN: int = 20          # anti-clustering (giorni)

# === Guardie statistiche ===
MIN_EVENTS: int = 5         # sotto questa N le statistiche non sono affidabili
N_BOOT: int = 2000          # ripetizioni block-bootstrap
BOOT_SEED: int = 12345      # seed → risultati riproducibili (no rumore run-to-run)

# === Definizione UNICA delle 4 regole (signal × bucket VIX) ===
# vix_min ≤ VIX < vix_max
RULES: dict[str, dict] = {
    "R1": dict(signal="overbought", vix_min=15.0, vix_max=20.0,
               emoji="🔴", name="Short Volatilità",
               desc="Mean reversion della volatilità in regime normale."),
    "R3": dict(signal="overbought", vix_min=20.0, vix_max=math.inf,
               emoji="🩸", name="Blow-off Top (Panico)",
               desc="Collasso violento dell'IV, esaurimento panico."),
    "R2": dict(signal="oversold", vix_min=0.0, vix_max=15.0,
               emoji="🟢", name="Complacenza / Long Vol",
               desc="Rischio latente, valutare coperture."),
    "R4": dict(signal="oversold", vix_min=15.0, vix_max=20.0,
               emoji="🟡", name="VIX Pop (Quiete pre-tempesta)",
               desc="Spike tattico del VIX a breve termine (5-15gg)."),
}


def classify_rule(signal: str, vix: float) -> str | None:
    """
    Restituisce la chiave di regola (R1/R2/R3/R4) per un dato tipo di segnale e
    livello VIX, oppure None se la combinazione non corrisponde ad alcuna regola.
    """
    for key, r in RULES.items():
        if signal == r["signal"] and r["vix_min"] <= vix < r["vix_max"]:
            return key
    return None


def active_rule(
    curr_z: float,
    curr_vix: float,
    upper: float = UPPER_THRESH,
    lower: float = LOWER_THRESH,
) -> str | None:
    """
    Regola attiva all'istante corrente dato lo z-score e il VIX.
    NB: è la condizione *istantanea*; la parità con il backtest richiede anche
    il filtro cooldown (evento fresco) — vedi telegram_alert.py.
    """
    if curr_z >= upper:
        return classify_rule("overbought", curr_vix)
    if curr_z <= lower:
        return classify_rule("oversold", curr_vix)
    return None

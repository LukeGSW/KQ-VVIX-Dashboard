"""
signals.py — Calcolo log-zScore sul VVIX e rilevamento eventi estremi.

Metodologia:
    1. Trasformazione logaritmica: log_vvix[t] = log(vvix[t])
       Il VVIX ha distribuzione asimmetrica a destra (skewed); il log
       normalizza la distribuzione e rende ±N deviazioni standard
       statisticamente simmetriche e confrontabili.

    2. Rolling log-zScore su finestra di N giorni:
         zscore[t] = (log_vvix[t] - mean(log_vvix[t-N+1:t])) / std(log_vvix[t-N+1:t])
       ZERO look-ahead bias: min_periods=window impone che lo z-score
       sia NaN fino a quando non sono disponibili almeno N osservazioni.
       La media e la std al giorno t usano esclusivamente dati storici
       fino a t incluso.

    3. Rilevamento eventi: primo crossing di soglia (overbought/oversold).

    4. Cooldown anti-clustering: dopo ogni segnale registrato, i successivi
       `cooldown` barre (giorni) vengono ignorati. Viene contato SOLO il
       primo segnale in ogni cluster di volatilità estrema.
"""

import numpy as np
import pandas as pd


def compute_log_zscore(vvix: pd.Series, window: int = 90) -> pd.Series:
    """
    Calcola il rolling log-zScore del VVIX.

    Applica log() al VVIX per simmetrizzare la distribuzione, poi calcola
    lo z-score rispetto alla media e deviazione standard mobili degli
    ultimi `window` giorni (incluso il giorno corrente t).

    Garanzia anti look-ahead:
        min_periods=window → NaN per i primi (window-1) giorni.
        Il calcolo a t usa solo dati [t-window+1 ... t].

    Args:
        vvix:   Serie storica VVIX (daily close), DatetimeIndex.
        window: Finestra rolling in giorni di trading (default: 90).

    Returns:
        pd.Series con il log-zScore, stesso indice di vvix.
        Valori NaN per i primi (window-1) giorni.
    """
    log_v  = np.log(vvix)
    roll_m = log_v.rolling(window=window, min_periods=window).mean()
    roll_s = log_v.rolling(window=window, min_periods=window).std(ddof=1)

    # Evita divisione per zero (caso teorico: tutti i valori identici)
    roll_s = roll_s.replace(0.0, np.nan)

    zscore = (log_v - roll_m) / roll_s
    return zscore.rename("log_zscore")


def detect_events(
    zscore: pd.Series,
    lower_thresh: float = -2.0,
    upper_thresh: float = 2.5,
    cooldown: int = 20,
) -> pd.DataFrame:
    """
    Rileva eventi estremi sul log-zScore del VVIX con deduplicazione cooldown.

    Logica:
        - Scansione cronologica dei valori non-NaN dello z-score.
        - Un evento è registrato quando z >= upper_thresh (overbought)
          o z <= lower_thresh (oversold).
        - Dopo ogni segnale registrato, i successivi `cooldown` giorni
          (posizioni nell'array ordinato) vengono ignorati.
          Questo conta solo il PRIMO crossing in ogni episodio estremo.

    Nessun look-ahead bias: la rilevazione usa esclusivamente il valore
    corrente dello z-score (già calcolato senza look-ahead in compute_log_zscore).

    Args:
        zscore:       Serie log-zScore (output di compute_log_zscore).
        lower_thresh: Soglia inferiore oversold (default: -2.0).
        upper_thresh: Soglia superiore overbought (default: +2.5).
        cooldown:     Giorni minimi tra due segnali consecutivi (default: 20).

    Returns:
        pd.DataFrame con DatetimeIndex (date segnale) e colonne:
            'signal'  : 'overbought' | 'oversold'
            'zscore'  : valore z-score al momento del segnale
    """
    records = []
    last_signal_pos = -(cooldown + 1)  # inizializza fuori cooldown

    valid = zscore.dropna()

    for i, (date, z) in enumerate(valid.items()):
        # Verifica cooldown in termini di posizione sequenziale
        if (i - last_signal_pos) < cooldown:
            continue

        if z >= upper_thresh:
            records.append({"date": date, "signal": "overbought", "zscore": z})
            last_signal_pos = i
        elif z <= lower_thresh:
            records.append({"date": date, "signal": "oversold", "zscore": z})
            last_signal_pos = i

    if not records:
        return pd.DataFrame(columns=["signal", "zscore"])

    df = pd.DataFrame(records).set_index("date")
    df.index.name = "date"
    return df

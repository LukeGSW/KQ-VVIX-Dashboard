"""
forward_returns.py — Calcolo rendimenti forward semplici per SPX e VIX.

Convenzione rendimento:
    ret = (price[t + N] - price[t]) / price[t] * 100  (percentuale semplice)

Il segnale viene generato sulla chiusura del giorno t. Il rendimento misura
la variazione tra la chiusura di t e la chiusura di t+N giorni di trading.
Questo è corretto e non introduce look-ahead: stiamo misurando cosa accade
DOPO il segnale, che è esattamente lo scopo dell'analisi.

Nota: eventi troppo recenti (dove t+N supera la fine della serie) producono
NaN per quegli orizzonti; vengono inclusi nel DataFrame ma esclusi
automaticamente dai calcoli statistici tramite dropna().
"""

import numpy as np
import pandas as pd

# Orizzonti forward in giorni di trading
HORIZONS = [5, 10, 20, 60, 90]


def compute_forward_returns(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    horizons: list = None,
) -> pd.DataFrame:
    """
    Calcola i rendimenti forward semplici (%) di SPX e VIX per ogni evento.

    Per ogni segnale alla data t:
        spx_ret_Nd = (spx[t+N] - spx[t]) / spx[t] * 100
        vix_ret_Nd = (vix[t+N] - vix[t]) / vix[t] * 100

    dove N è espresso in giorni di trading (posizioni nell'indice ordinato).

    Garanzia no look-ahead bias: il calcolo usa esclusivamente dati
    cronologicamente successivi al segnale (t+1 ... t+N).

    Args:
        prices:   DataFrame con colonne ['spx', 'vix'] e DatetimeIndex daily.
        events:   DataFrame prodotto da detect_events
                  (DatetimeIndex = date segnale, colonne: 'signal', 'zscore').
        horizons: Lista orizzonti in giorni di trading
                  (default: [5, 10, 20, 60, 90]).

    Returns:
        pd.DataFrame con una riga per ogni evento e colonne:
            signal, zscore, vix_at_signal,
            spx_ret_5d, spx_ret_10d, ..., spx_ret_90d,
            vix_ret_5d, vix_ret_10d, ..., vix_ret_90d
        DatetimeIndex = date dei segnali.
    """
    if horizons is None:
        horizons = HORIZONS

    if events.empty:
        cols = ["signal", "zscore", "vix_at_signal"]
        for h in horizons:
            cols += [f"spx_ret_{h}d", f"vix_ret_{h}d"]
        return pd.DataFrame(columns=cols)

    # Array numpy delle date dell'indice prezzi (per lookup efficiente)
    price_index = prices.index

    results = []
    valid_dates = []

    for event_date, row in events.iterrows():
        # La data del segnale deve essere nell'indice dei prezzi
        if event_date not in price_index:
            continue

        # Posizione del giorno del segnale nella serie prezzi
        loc = price_index.get_loc(event_date)

        spx_t = prices["spx"].iloc[loc]
        vix_t = prices["vix"].iloc[loc]

        record = {
            "signal":        row["signal"],
            "zscore":        row["zscore"],
            "vix_at_signal": vix_t,
        }

        for h in horizons:
            future_loc = loc + h
            if future_loc < len(prices):
                # Rendimento semplice percentuale t → t+h
                record[f"spx_ret_{h}d"] = (
                    (prices["spx"].iloc[future_loc] - spx_t) / spx_t * 100
                )
                record[f"vix_ret_{h}d"] = (
                    (prices["vix"].iloc[future_loc] - vix_t) / vix_t * 100
                )
            else:
                # Evento troppo recente: orizzonte non ancora disponibile
                record[f"spx_ret_{h}d"] = np.nan
                record[f"vix_ret_{h}d"] = np.nan

        results.append(record)
        valid_dates.append(event_date)

    if not results:
        cols = ["signal", "zscore", "vix_at_signal"]
        for h in horizons:
            cols += [f"spx_ret_{h}d", f"vix_ret_{h}d"]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(results, index=pd.DatetimeIndex(valid_dates))
    df.index.name = "date"
    return df

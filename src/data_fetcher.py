"""
data_fetcher.py — Fetch e caching dei dati da EODHD.

Serie caricate:
    VVIX.INDX  : CBOE VVIX Index (volatility of VIX)
    VIX.INDX   : CBOE VIX Index
    GSPC.INDX  : S&P 500 Index

Tutto lo storico disponibile viene scaricato in una sola chiamata;
il DataFrame finale è allineato sulle date comuni alle tre serie (inner join).
"""

import requests
import pandas as pd
import streamlit as st

# === TICKER EODHD ===
TICKER_VVIX = "VVIX.INDX"
TICKER_VIX  = "VIX.INDX"
TICKER_SPX  = "GSPC.INDX"


def _extract_close(df: pd.DataFrame) -> pd.Series:
    """
    Estrae la serie di prezzi dal DataFrame EODHD.

    Preferisce adjusted_close se presente e non vuota; altrimenti usa close.
    Per gli indici CBOE (VVIX, VIX) adjusted_close = close.

    Args:
        df: DataFrame con colonne OHLCV restituite da EODHD.

    Returns:
        pd.Series float con la serie di prezzi.
    """
    if "adjusted_close" in df.columns and df["adjusted_close"].notna().any():
        return df["adjusted_close"].astype(float)
    return df["close"].astype(float)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_index(ticker: str, api_key: str) -> pd.Series:
    """
    Scarica l'intera storia disponibile di un indice da EODHD.

    Args:
        ticker:  Simbolo EODHD (es. 'VVIX.INDX', 'VIX.INDX', 'GSPC.INDX').
        api_key: Chiave API EODHD.

    Returns:
        pd.Series con DatetimeIndex (daily, ordinato ASC) e valori float.

    Raises:
        requests.HTTPError: se la chiamata API fallisce (401, 404, 429...).
        ValueError:         se la risposta è vuota.
    """
    url = (
        f"https://eodhd.com/api/eod/{ticker}"
        f"?period=d&api_token={api_key}&fmt=json"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        raise ValueError(f"Nessun dato restituito da EODHD per {ticker}.")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    return _extract_close(df).rename(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data(api_key: str) -> pd.DataFrame:
    """
    Carica VVIX, VIX e SPX da EODHD e li allinea in un DataFrame comune.

    Il periodo coperto inizia dalla prima data disponibile del VVIX
    (serie più corta tra le tre) e termina all'ultimo dato comune.
    Vengono mantenute solo le date in cui tutte e tre le serie hanno
    un valore valido (inner join via dropna).

    Args:
        api_key: Chiave API EODHD (letta da st.secrets['EODHD_API_KEY']).

    Returns:
        pd.DataFrame con colonne ['vvix', 'vix', 'spx'] e DatetimeIndex daily,
        ordinato in ordine cronologico ascendente.

    Raises:
        requests.HTTPError: propagata dal fetch se la chiamata fallisce.
        ValueError:         se una delle serie è vuota.
    """
    vvix = _fetch_index(TICKER_VVIX, api_key)
    vix  = _fetch_index(TICKER_VIX,  api_key)
    spx  = _fetch_index(TICKER_SPX,  api_key)

    df = pd.DataFrame({"vvix": vvix, "vix": vix, "spx": spx})
    df.sort_index(inplace=True)

    # Inner join: solo date con tutte e tre le serie valide
    df.dropna(subset=["vvix", "vix", "spx"], inplace=True)

    return df

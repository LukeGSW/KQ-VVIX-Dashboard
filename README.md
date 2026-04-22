# 📊 VVIX Behavior Analysis Dashboard

**Kriterion Quant** — Studio quantitativo del comportamento di VIX e S&P 500
in risposta agli estremi statistici del VVIX (CBOE Volatility of VIX Index).

---

## Struttura repository

```
vvix-dashboard/
├── app.py                  # Entry point Streamlit
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Dark theme
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py     # Fetch EODHD: VVIX, VIX, SPX
│   ├── signals.py          # Log-zScore rolling + detect_events (cooldown)
│   ├── forward_returns.py  # Rendimenti forward 5/10/20/60/90gg
│   ├── stats.py            # Hit rate, profit factor, percentili
│   ├── conditional.py      # Studio 2D: zScore VVIX × regime VIX
│   └── charts.py           # Grafici Plotly dark theme
└── .gitignore
```

---

## Deploy su Streamlit Cloud

1. Fai fork/clone di questo repository su GitHub.
2. Vai su [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**.
3. Connetti il repository GitHub e seleziona `app.py` come entry point.
4. In **Advanced settings → Secrets**, incolla:
   ```toml
   EODHD_API_KEY = "la-tua-chiave-eodhd"
   ```
5. Clicca **Deploy**.

## Test locale

```bash
pip install -r requirements.txt

# Crea il file secrets locale (NON committare)
mkdir -p .streamlit
echo 'EODHD_API_KEY = "la-tua-chiave"' > .streamlit/secrets.toml

streamlit run app.py
```

---

## Metodologia

- **Log-ZScore rolling:** applicato al logaritmo del VVIX per correggere l'asimmetria
  della distribuzione. Finestra default: 90 giorni. Zero look-ahead bias garantito
  da `min_periods=window` nel calcolo rolling.

- **Soglie asimmetriche:** default Oversold ≤ −2.0 / Overbought ≥ +2.5 (configurabili).

- **Anti-clustering:** cooldown di 20 giorni tra segnali consecutivi (configurabile).
  Viene registrato solo il primo crossing in ogni episodio estremo.

- **Rendimenti forward:** semplici percentuali a 5, 10, 20, 60, 90 giorni di trading
  misurate dalla chiusura del giorno segnale.

- **Studio condizionale:** i segnali vengono stratificati per livello VIX al momento
  del trigger (< 15, 15–20, 20–30, ≥ 30) per rivelare dipendenze dal regime.

---

## Dati

Fonte: [EODHD Historical Data](https://eodhd.com) via REST API.
Tickers: `VVIX.INDX` · `VIX.INDX` · `GSPC.INDX`

---

*Kriterion Quant — [kriterionquant.com](https://kriterionquant.com)*

"""
app.py — VVIX Behavior Analysis Dashboard | Kriterion Quant

Struttura:
    [0] Setup pagina e sidebar
    [1] Header operativo — lettura corrente + segnale attivo
    [2] Grafico storico VVIX + log-zScore + markers segnali
    [3] Forward Returns Analysis — SPX e VIX per tipo segnale
    [4] Conditional Study — rendimenti per regime VIX × orizzonte
    [5] Segnali Operativi — edge distillato dalla ricerca + signal log
    [6] Footer disclaimer

Requisiti:
    - EODHD_API_KEY nei Secrets di Streamlit (Settings → Secrets)
    - Python >= 3.10, dipendenze in requirements.txt
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from src.data_fetcher import load_all_data
from src.signals import compute_log_zscore, detect_events
from src.forward_returns import compute_forward_returns, HORIZONS
from src.stats import compute_summary_stats, compute_subset_stats

from src.conditional import (
    compute_conditional_stats,
    compute_conditional_heatmap_data,
    VIX_LABELS,
)
from src.charts import (
    build_vvix_chart,
    build_zscore_recent,
    build_forward_returns_bar,
    build_distribution_chart,
    build_conditional_heatmap,
    build_signal_timeline,
)

# ═══════════════════════════════════════════════════════════════════════════════
# [0] CONFIGURAZIONE PAGINA
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VVIX Behavior Dashboard | Kriterion Quant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Key da Streamlit Secrets (no hardcoding)
try:
    EODHD_API_KEY = st.secrets["EODHD_API_KEY"]
except KeyError:
    st.error(
        "❌ **EODHD_API_KEY non trovata.**\n\n"
        "Configura la chiave nei Secrets di Streamlit: "
        "*Settings → Secrets → `EODHD_API_KEY = \"la-tua-chiave\"`*"
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — PARAMETRI
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Parametri")
    st.divider()

    st.markdown("**Finestra Log-ZScore**")
    zscore_window = st.slider(
        "Giorni rolling",
        min_value=60, max_value=252, value=90, step=5,
        help=(
            "Finestra rolling per calcolo log-zScore sul VVIX. "
            "Default: 90 giorni (trimestrale). "
            "Finestre più corte = più segnali, più rumore."
        ),
    )

    st.divider()
    st.markdown("**Soglie segnale**")
    col_s1, col_s2 = st.columns(2)
    upper_thresh = col_s1.number_input(
        "Overbought ↑", min_value=1.5, max_value=4.0, value=2.5, step=0.1,
        help="zScore ≥ soglia → segnale Overbought VVIX",
    )
    lower_thresh = col_s2.number_input(
        "Oversold ↓", min_value=-4.0, max_value=-1.0, value=-2.0, step=0.1,
        help="zScore ≤ soglia → segnale Oversold VVIX",
    )

    st.divider()
    st.markdown("**Anti-clustering**")
    cooldown_days = st.slider(
        "Cooldown tra segnali (giorni)",
        min_value=5, max_value=60, value=20, step=5,
        help=(
            "Giorni minimi tra due segnali consecutivi dello stesso tipo. "
            "Evita di contare più volte lo stesso episodio di volatilità estrema."
        ),
    )

    st.divider()
    st.markdown("**Distribuzione rendimenti**")
    distr_horizon = st.selectbox(
        "Orizzonte da visualizzare",
        options=HORIZONS, index=2,
        help="Orizzonte forward mostrato nel grafico distribuzione (istogramma).",
    )

    st.divider()
    st.caption("📡 Dati: EODHD Historical Data")
    st.caption(f"🕒 {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.caption("🔗 [Kriterion Quant](https://kriterionquant.com)")

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📊 VVIX Behavior Analysis Dashboard")
st.markdown("""
Il **VVIX** (CBOE VVIX Index) misura la volatilità implicita attesa del VIX,
ovvero la *"volatilità della volatilità"* dell'S&P 500. Quando il VVIX raggiunge
estremi statistici — misurati tramite log-zScore rolling — si creano condizioni
anomale che storicamente hanno anticipato movimenti significativi di **VIX** e **S&P 500**.

Questa dashboard studia il comportamento storico dei due indici nei giorni successivi
agli estremi del VVIX, con analisi condizionata al regime di volatilità prevalente
(livello VIX al momento del segnale).

> **Come si usa:** imposta soglie e finestra zScore nella sidebar → esplora le sezioni.
""")
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# CARICAMENTO DATI
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner("⏳ Caricamento dati da EODHD (VVIX · VIX · S&P 500)..."):
    try:
        df = load_all_data(EODHD_API_KEY)
    except Exception as e:
        st.error(f"❌ Errore nel caricamento dati: {e}")
        st.stop()

min_bars_required = zscore_window + max(HORIZONS) + 10
if df.empty or len(df) < min_bars_required:
    st.error(
        f"⚠️ Dati insufficienti ({len(df)} barre). "
        f"Necessarie almeno {min_bars_required} barre. "
        "Verifica la chiave API e la connessione di rete."
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# CALCOLI PRINCIPALI (no look-ahead bias garantito da min_periods=window)
# ═══════════════════════════════════════════════════════════════════════════════

# 1. Log-zScore rolling
zscore = compute_log_zscore(df["vvix"], window=zscore_window)

# 2. Rilevamento eventi con cooldown
events = detect_events(
    zscore,
    lower_thresh=lower_thresh,
    upper_thresh=upper_thresh,
    cooldown=cooldown_days,
)

# 3. Rendimenti forward (misura il comportamento successivo al segnale)
fwd_returns = compute_forward_returns(
    prices=df[["spx", "vix"]],
    events=events,
    horizons=HORIZONS,
)

# Separazione per tipo segnale
ob_rets = fwd_returns[fwd_returns["signal"] == "overbought"]
os_rets = fwd_returns[fwd_returns["signal"] == "oversold"]
n_ob    = len(ob_rets)
n_os    = len(os_rets)

# ═══════════════════════════════════════════════════════════════════════════════
# [1] HEADER OPERATIVO — SEGNALE CORRENTE
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🎯 Stato Operativo Corrente")
st.markdown("""
Lettura del VVIX, VIX e S&P 500 all'ultima data disponibile, con il corrispondente
log-zScore. Un valore estremo dello z-score indica che il VVIX si trova in una zona
storicamente rara rispetto alla sua storia recente (finestra rolling impostata in sidebar).
""")

# Valori correnti
curr_vvix   = df["vvix"].iloc[-1]
curr_vix    = df["vix"].iloc[-1]
curr_spx    = df["spx"].iloc[-1]
curr_z      = zscore.dropna().iloc[-1]
prev_z      = zscore.dropna().iloc[-2] if len(zscore.dropna()) > 1 else curr_z
prev_vvix   = df["vvix"].iloc[-2]
prev_vix    = df["vix"].iloc[-2]
prev_spx    = df["spx"].iloc[-2]

# Status segnale
if curr_z >= upper_thresh:
    signal_txt = "🔴 OVERBOUGHT"
elif curr_z <= lower_thresh:
    signal_txt = "🟢 OVERSOLD"
else:
    signal_txt = "⚪ NEUTRO"

# KPI row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("VVIX",      f"{curr_vvix:.2f}",   delta=f"{curr_vvix - prev_vvix:+.2f}")
c2.metric("VIX",       f"{curr_vix:.2f}",    delta=f"{curr_vix  - prev_vix:+.2f}")
c3.metric("S&P 500",   f"{curr_spx:,.2f}",   delta=f"{(curr_spx / prev_spx - 1) * 100:+.2f}%")
c4.metric("Log-ZScore", f"{curr_z:.3f}",     delta=f"{curr_z - prev_z:+.3f}")
c5.metric("Segnale",   signal_txt)

# Mini-chart log-zScore ultimo anno
zscore_clean = zscore.dropna()
fig_recent   = build_zscore_recent(zscore_clean, upper_thresh, lower_thresh, lookback=252)
st.plotly_chart(fig_recent, width='stretch')

# Riepilogo dataset
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Periodo",              f"{df.index[0].strftime('%b %Y')} – {df.index[-1].strftime('%b %Y')}")
col_b.metric("Barre totali",         f"{len(df):,}")
col_c.metric("Segnali Overbought",   str(n_ob))
col_d.metric("Segnali Oversold",     str(n_os))

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# [2] GRAFICO STORICO VVIX + LOG-ZSCORE
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📈 VVIX Storico e Log-ZScore")
st.markdown(f"""
Il pannello superiore mostra la serie storica del VVIX con i segnali estremi marcati:
**triangoli rossi ↑** per Overbought (zScore ≥ {upper_thresh}),
**triangoli verdi ↓** per Oversold (zScore ≤ {lower_thresh}).

Il pannello inferiore mostra il **log-zScore rolling {zscore_window}gg**.
Il logaritmo normalizza la distribuzione asimmetrica del VVIX, rendendo le deviazioni
superiori e inferiori statisticamente confrontabili. La zona rossa evidenzia i periodi
in cui lo z-score supera la soglia overbought.
""")

fig_main = build_vvix_chart(df, zscore_clean, events, upper_thresh, lower_thresh)
st.plotly_chart(fig_main, width='stretch')

with st.expander("ℹ️ Metodologia — Log-ZScore e Anti-Clustering"):
    st.markdown(f"""
    **Formula log-zScore:**
    ```
    log_zscore[t] = (log(VVIX[t]) − media(log(VVIX[t−{zscore_window}+1 : t])))
                    / std(log(VVIX[t−{zscore_window}+1 : t]))
    ```
    La trasformazione logaritmica corregge la asimmetria a destra del VVIX
    (distribuzione con floor ma senza cap), rendendo le soglie ±N statisticamente simmetriche.

    **Zero look-ahead bias:** lo z-score al giorno *t* usa esclusivamente dati
    storici fino a *t* incluso (`min_periods={zscore_window}`). Le prime {zscore_window - 1}
    barre non producono segnali.

    **Anti-clustering (cooldown {cooldown_days}gg):** dopo ogni segnale registrato,
    i successivi {cooldown_days} giorni vengono ignorati. Viene contato solo il PRIMO
    crossing in ogni episodio di volatilità estrema, evitando di gonfiare artificialmente
    le statistiche con eventi ripetuti dallo stesso cluster.

    **Fonte dati:** EODHD Historical Data · VVIX.INDX · VIX.INDX · GSPC.INDX
    """)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# [3] FORWARD RETURNS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Analisi Rendimenti Forward")
st.markdown("""
Per ogni evento estremo del VVIX viene misurato il rendimento semplice (%) di
**S&P 500** e **VIX** a **5, 10, 20, 60 e 90 giorni** di trading successivi alla data del segnale.

Le barre mostrano il **rendimento medio** con **banda interquartile** (P25 – P75).
Il testo nelle barre riporta media, **hit rate** (% eventi con rendimento > 0)
e **profit factor** (guadagni aggregati / perdite aggregate in valore assoluto).
Un profit factor > 1 indica edge statistico direzionale.
""")

MIN_EVENTS = 5  # soglia minima campione per mostrare statistiche

tab_ob, tab_os = st.tabs(["🔴 Overbought VVIX", "🟢 Oversold VVIX"])

# ── Tab Overbought ──
with tab_ob:
    if n_ob < MIN_EVENTS:
        st.warning(
            f"⚠️ Solo **{n_ob}** eventi overbought nel dataset. "
            f"Abbassa la soglia (attuale: {upper_thresh}) o riduci il cooldown."
        )
    else:
        st.markdown(f"Campione: **{n_ob} eventi overbought** nel periodo analizzato.")

        # SPX dopo OB
        st.markdown("#### S&P 500 — dopo Overbought VVIX")
        stats_ob_spx = compute_summary_stats(fwd_returns, "overbought", "spx", HORIZONS)
        st.plotly_chart(
            build_forward_returns_bar(stats_ob_spx, "overbought", "SPX"),
            width='stretch',
        )
        _fmt_ob_spx = {
            "Media %": "{:+.2f}", "Mediana %": "{:+.2f}",
            "P25 %": "{:+.2f}",  "P75 %": "{:+.2f}",
            "Std Dev %": "{:.2f}", "Hit Rate %": "{:.1f}",
            "Profit Factor": "{:.2f}", "N": "{:.0f}",
        }
        st.dataframe(
            stats_ob_spx.style
                .format(_fmt_ob_spx)
                .background_gradient(subset=["Media %", "Hit Rate %"], cmap="RdYlGn"),
            width='stretch',
        )

        # Distribuzione SPX OB
        st.plotly_chart(
            build_distribution_chart(fwd_returns, "overbought", "spx", distr_horizon),
            width='stretch',
        )

        st.markdown("---")

        # VIX dopo OB
        st.markdown("#### VIX — dopo Overbought VVIX")
        stats_ob_vix = compute_summary_stats(fwd_returns, "overbought", "vix", HORIZONS)
        st.plotly_chart(
            build_forward_returns_bar(stats_ob_vix, "overbought", "VIX"),
            width='stretch',
        )
        st.dataframe(
            stats_ob_vix.style
                .format(_fmt_ob_spx)
                .background_gradient(subset=["Media %", "Hit Rate %"], cmap="RdYlGn"),
            width='stretch',
        )
        st.plotly_chart(
            build_distribution_chart(fwd_returns, "overbought", "vix", distr_horizon),
            width='stretch',
        )

# ── Tab Oversold ──
with tab_os:
    if n_os < MIN_EVENTS:
        st.warning(
            f"⚠️ Solo **{n_os}** eventi oversold nel dataset. "
            f"Abbassa la soglia in valore assoluto (attuale: {lower_thresh}) "
            "o riduci il cooldown."
        )
    else:
        st.markdown(f"Campione: **{n_os} eventi oversold** nel periodo analizzato.")

        # SPX dopo OS
        st.markdown("#### S&P 500 — dopo Oversold VVIX")
        stats_os_spx = compute_summary_stats(fwd_returns, "oversold", "spx", HORIZONS)
        st.plotly_chart(
            build_forward_returns_bar(stats_os_spx, "oversold", "SPX"),
            width='stretch',
        )
        _fmt_os = {
            "Media %": "{:+.2f}", "Mediana %": "{:+.2f}",
            "P25 %": "{:+.2f}",  "P75 %": "{:+.2f}",
            "Std Dev %": "{:.2f}", "Hit Rate %": "{:.1f}",
            "Profit Factor": "{:.2f}", "N": "{:.0f}",
        }
        st.dataframe(
            stats_os_spx.style
                .format(_fmt_os)
                .background_gradient(subset=["Media %", "Hit Rate %"], cmap="RdYlGn"),
            width='stretch',
        )
        st.plotly_chart(
            build_distribution_chart(fwd_returns, "oversold", "spx", distr_horizon),
            width='stretch',
        )

        st.markdown("---")

        # VIX dopo OS
        st.markdown("#### VIX — dopo Oversold VVIX")
        stats_os_vix = compute_summary_stats(fwd_returns, "oversold", "vix", HORIZONS)
        st.plotly_chart(
            build_forward_returns_bar(stats_os_vix, "oversold", "VIX"),
            width='stretch',
        )
        st.dataframe(
            stats_os_vix.style
                .format(_fmt_os)
                .background_gradient(subset=["Media %", "Hit Rate %"], cmap="RdYlGn"),
            width='stretch',
        )
        st.plotly_chart(
            build_distribution_chart(fwd_returns, "oversold", "vix", distr_horizon),
            width='stretch',
        )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# [4] CONDITIONAL STUDY — VVIX zScore × REGIME VIX
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🔬 Studio Condizionale: VVIX × Regime VIX")
st.markdown("""
**Un segnale estremo del VVIX non è sempre uguale.** Il comportamento successivo di
SPX e VIX dipende significativamente dal regime di volatilità in atto al momento del segnale,
misurato dal livello assoluto del VIX:

| Bucket | Interpretazione |
|--------|----------------|
| **VIX < 15** | Bassa volatilità / complacenza — mercato tranquillo |
| **15 ≤ VIX < 20** | Regime normale — volatilità fisiologica |
| **20 ≤ VIX < 30** | Volatilità elevata — attenzione al rischio |
| **VIX ≥ 30** | Alta stress / panico — potenziale tail risk |

Le heatmap mostrano il **rendimento medio (%)** di SPX e VIX per ogni combinazione
(regime VIX × orizzonte forward). Verde = rendimenti storicamente positivi, Rosso = negativi.

> ⚠️ **Nota statistica:** bucket con **N < 5** eventi devono essere interpretati
> con cautela: la stima della media è poco affidabile su campioni piccoli.
""")

tab_cond_ob, tab_cond_os = st.tabs(
    ["🔴 Conditional — Overbought", "🟢 Conditional — Oversold"]
)

# ── Conditional Overbought ──
with tab_cond_ob:
    if n_ob < MIN_EVENTS:
        st.warning("Campione overbought insufficiente per lo studio condizionale.")
    else:
        col_h1, col_h2 = st.columns(2)

        with col_h1:
            st.markdown("**S&P 500 — Rendimento medio per regime VIX**")
            hm_ob_spx  = compute_conditional_heatmap_data(fwd_returns, "overbought", "spx", HORIZONS)
            tbl_ob_spx = compute_conditional_stats(fwd_returns, "overbought", "spx", HORIZONS)
            st.plotly_chart(
                build_conditional_heatmap(hm_ob_spx, "overbought", "spx"),
                width='stretch',
            )
            horizon_cols_ob = [c for c in tbl_ob_spx.columns if c != "N eventi"]
            st.dataframe(
                tbl_ob_spx.style.format(
                    {c: "{:+.2f}" for c in horizon_cols_ob}
                ),
                width='stretch',
            )

        with col_h2:
            st.markdown("**VIX — Rendimento medio per regime VIX**")
            hm_ob_vix  = compute_conditional_heatmap_data(fwd_returns, "overbought", "vix", HORIZONS)
            tbl_ob_vix = compute_conditional_stats(fwd_returns, "overbought", "vix", HORIZONS)
            st.plotly_chart(
                build_conditional_heatmap(hm_ob_vix, "overbought", "vix"),
                width='stretch',
            )
            horizon_cols_ob_v = [c for c in tbl_ob_vix.columns if c != "N eventi"]
            st.dataframe(
                tbl_ob_vix.style.format(
                    {c: "{:+.2f}" for c in horizon_cols_ob_v}
                ),
                width='stretch',
            )

# ── Conditional Oversold ──
with tab_cond_os:
    if n_os < MIN_EVENTS:
        st.warning("Campione oversold insufficiente per lo studio condizionale.")
    else:
        col_h3, col_h4 = st.columns(2)

        with col_h3:
            st.markdown("**S&P 500 — Rendimento medio per regime VIX**")
            hm_os_spx  = compute_conditional_heatmap_data(fwd_returns, "oversold", "spx", HORIZONS)
            tbl_os_spx = compute_conditional_stats(fwd_returns, "oversold", "spx", HORIZONS)
            st.plotly_chart(
                build_conditional_heatmap(hm_os_spx, "oversold", "spx"),
                width='stretch',
            )
            horizon_cols_os = [c for c in tbl_os_spx.columns if c != "N eventi"]
            st.dataframe(
                tbl_os_spx.style.format(
                    {c: "{:+.2f}" for c in horizon_cols_os}
                ),
                width='stretch',
            )

        with col_h4:
            st.markdown("**VIX — Rendimento medio per regime VIX**")
            hm_os_vix  = compute_conditional_heatmap_data(fwd_returns, "oversold", "vix", HORIZONS)
            tbl_os_vix = compute_conditional_stats(fwd_returns, "oversold", "vix", HORIZONS)
            st.plotly_chart(
                build_conditional_heatmap(hm_os_vix, "oversold", "vix"),
                width='stretch',
            )
            horizon_cols_os_v = [c for c in tbl_os_vix.columns if c != "N eventi"]
            st.dataframe(
                tbl_os_vix.style.format(
                    {c: "{:+.2f}" for c in horizon_cols_os_v}
                ),
                width='stretch',
            )

# ═══════════════════════════════════════════════════════════════════════════════
# [5] SEGNALI OPERATIVI — EDGE DISTILLATO DALLA RICERCA
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("⚡ Segnali Operativi — Edge Distillato dalla Ricerca")
st.markdown("""
Lo studio del comportamento di VIX e S&P 500 dopo gli estremi del VVIX ha evidenziato
due **pattern statisticamente robusti**. Le regole seguenti sintetizzano i setup con
il profilo rischio/rendimento più favorevole emerso dall'analisi storica.

> ⚠️ Queste regole hanno finalità di **ricerca quantitativa** e non costituiscono
> consulenza finanziaria. Ogni operatività deve essere calata nel proprio risk management.
""")

# ─── Cards Edge ────────────────────────────────────────────────────────────────
col_r1, col_r2 = st.columns(2)

with col_r1:
    st.markdown(
        """
        <div style="background:#2a1515;border-left:4px solid #F44336;
                    padding:18px 20px;border-radius:8px;height:100%;">
          <h4 style="color:#F44336;margin:0 0 12px 0;">
            🔴 Regola 1 — VVIX Overbought → Short Volatilità
          </h4>
          <p style="color:#E0E0E0;margin:0 0 8px 0;">
            <b>Filtro:</b> VVIX zScore ≥ soglia OB &amp; VIX tra 15 e 30
          </p>
          <p style="color:#E0E0E0;margin:0 0 8px 0;">
            <b>Edge:</b> il VIX tende a calare sistematicamente nei 20–60 giorni
            successivi al segnale. Hit rate storicamente &gt;65%,
            Profit Factor &gt;2.0 sulla finestra 20–60 giorni.
          </p>
          <p style="color:#9E9E9E;margin:0;font-size:0.88em;">
            <i>Operatività tipica:</i> short VXX/UVXY · sell VIX call spread ·
            sell premium (straddle/strangle SPX)
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_r2:
    st.markdown(
        """
        <div style="background:#152a15;border-left:4px solid #4CAF50;
                    padding:18px 20px;border-radius:8px;height:100%;">
          <h4 style="color:#4CAF50;margin:0 0 12px 0;">
            🟢 Regola 2 — VVIX Oversold + VIX &lt; 15 → Cautela / Long Vol
          </h4>
          <p style="color:#E0E0E0;margin:0 0 8px 0;">
            <b>Filtro:</b> VVIX zScore ≤ soglia OS &amp; VIX &lt; 15 (complacenza)
          </p>
          <p style="color:#E0E0E0;margin:0 0 8px 0;">
            <b>Edge:</b> SPX flat/debole e VIX potenzialmente in risalita.
            Ambiente di volatilità compressa con fear index anomalmente basso
            = zona di rischio latente da presidiare con hedge.
          </p>
          <p style="color:#9E9E9E;margin:0;font-size:0.88em;">
            <i>Operatività tipica:</i> long VIX calls / VXX ·
            riduzione esposizione delta SPX · hedging portafoglio opzioni
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")  # spacer

# ─── Helper: VIX regime label ─────────────────────────────────────────────────
def _vix_regime(v: float) -> str:
    if v < 15:   return "VIX < 15"
    elif v < 20: return "15–20"
    elif v < 30: return "20–30"
    else:        return "VIX ≥ 30"


# ─── Stato segnale corrente ────────────────────────────────────────────────────
st.markdown("#### 🎯 Segnale Corrente rispetto alle Regole Operative")

curr_regime = _vix_regime(curr_vix)
rule1_active = (curr_z >= upper_thresh) and (15 <= curr_vix < 30)
rule2_active = (curr_z <= lower_thresh) and (curr_vix < 15)

if rule1_active:
    st.error(
        f"🔴 **REGOLA 1 ATTIVA** — VVIX Overbought (zScore {curr_z:.3f} ≥ {upper_thresh}) "
        f"con VIX = {curr_vix:.2f} (regime {curr_regime}). "
        "Setup short volatilità potenzialmente attivo."
    )
elif rule2_active:
    st.success(
        f"🟢 **REGOLA 2 ATTIVA** — VVIX Oversold (zScore {curr_z:.3f} ≤ {lower_thresh}) "
        f"con VIX = {curr_vix:.2f} (regime {curr_regime}). "
        "Segnale di cautela: valutare hedge su portafoglio."
    )
elif curr_z >= upper_thresh:
    st.warning(
        f"🔴 VVIX Overbought (zScore {curr_z:.3f}) ma VIX = {curr_vix:.2f} fuori dal "
        f"range 15–30 (regime: {curr_regime}). Regola 1 non soddisfatta nel filtro VIX."
    )
elif curr_z <= lower_thresh:
    st.info(
        f"🟢 VVIX Oversold (zScore {curr_z:.3f}) ma VIX = {curr_vix:.2f} non in zona "
        f"complacenza <15 (regime: {curr_regime}). Regola 2 non soddisfatta nel filtro VIX."
    )
else:
    st.info(
        f"⚪ Nessuna regola attiva — zScore corrente {curr_z:.3f} in zona neutra. "
        f"VIX = {curr_vix:.2f} (regime: {curr_regime})."
    )

st.markdown("")

# ─── Performance storica Regola 1 ─────────────────────────────────────────────
st.markdown("#### 📊 Performance Storica — Regola 1 (VVIX OB, VIX 15–30)")

r1_subset = fwd_returns[
    (fwd_returns["signal"] == "overbought") &
    (fwd_returns["vix_at_signal"] >= 15) &
    (fwd_returns["vix_at_signal"] < 30)
]
n_r1 = len(r1_subset)

if n_r1 < MIN_EVENTS:
    st.warning(
        f"⚠️ Solo **{n_r1}** eventi per Regola 1 con i parametri correnti. "
        "Prova ad abbassare la soglia OB o ridurre il cooldown per ampliare il campione."
    )
else:
    st.markdown(
        f"Campione: **{n_r1} eventi** (OB con VIX 15–30) | "
        f"Periodo: {r1_subset.index.min().strftime('%b %Y')} – "
        f"{r1_subset.index.max().strftime('%b %Y')}"
    )
    col_r1a, col_r1b = st.columns(2)

    _fmt_rule = {
        "Media %": "{:+.2f}", "Mediana %": "{:+.2f}",
        "P25 %": "{:+.2f}",  "P75 %": "{:+.2f}",
        "Std Dev %": "{:.2f}", "Hit Rate %": "{:.1f}",
        "Profit Factor": "{:.2f}", "N": "{:.0f}",
    }

    with col_r1a:
        st.markdown("**VIX — Rendimenti Forward (target primario)**")
        stats_r1_vix = compute_subset_stats(r1_subset, "vix", HORIZONS)
        st.plotly_chart(
            build_forward_returns_bar(stats_r1_vix, "overbought", "VIX"),
            width='stretch',
        )
        st.dataframe(
            stats_r1_vix.style
                .format(_fmt_rule)
                .background_gradient(subset=["Media %", "Hit Rate %"], cmap="RdYlGn"),
            width='stretch',
        )

    with col_r1b:
        st.markdown("**S&P 500 — Rendimenti Forward**")
        stats_r1_spx = compute_subset_stats(r1_subset, "spx", HORIZONS)
        st.plotly_chart(
            build_forward_returns_bar(stats_r1_spx, "overbought", "SPX"),
            width='stretch',
        )
        st.dataframe(
            stats_r1_spx.style
                .format(_fmt_rule)
                .background_gradient(subset=["Media %", "Hit Rate %"], cmap="RdYlGn"),
            width='stretch',
        )

st.markdown("")

# ─── Performance storica Regola 2 ─────────────────────────────────────────────
st.markdown("#### 📊 Performance Storica — Regola 2 (VVIX OS, VIX < 15)")

r2_subset = fwd_returns[
    (fwd_returns["signal"] == "oversold") &
    (fwd_returns["vix_at_signal"] < 15)
]
n_r2 = len(r2_subset)

if n_r2 < MIN_EVENTS:
    st.warning(
        f"⚠️ Solo **{n_r2}** eventi per Regola 2 con i parametri correnti. "
        "Prova ad abbassare la soglia OS (in valore assoluto) o ridurre il cooldown."
    )
else:
    st.markdown(
        f"Campione: **{n_r2} eventi** (OS con VIX < 15) | "
        f"Periodo: {r2_subset.index.min().strftime('%b %Y')} – "
        f"{r2_subset.index.max().strftime('%b %Y')}"
    )
    col_r2a, col_r2b = st.columns(2)

    with col_r2a:
        st.markdown("**VIX — Rendimenti Forward (target: spike atteso)**")
        stats_r2_vix = compute_subset_stats(r2_subset, "vix", HORIZONS)
        st.plotly_chart(
            build_forward_returns_bar(stats_r2_vix, "oversold", "VIX"),
            width='stretch',
        )
        st.dataframe(
            stats_r2_vix.style
                .format(_fmt_rule)
                .background_gradient(subset=["Media %", "Hit Rate %"], cmap="RdYlGn"),
            width='stretch',
        )

    with col_r2b:
        st.markdown("**S&P 500 — Rendimenti Forward**")
        stats_r2_spx = compute_subset_stats(r2_subset, "spx", HORIZONS)
        st.plotly_chart(
            build_forward_returns_bar(stats_r2_spx, "oversold", "SPX"),
            width='stretch',
        )
        st.dataframe(
            stats_r2_spx.style
                .format(_fmt_rule)
                .background_gradient(subset=["Media %", "Hit Rate %"], cmap="RdYlGn"),
            width='stretch',
        )

st.markdown("")

# ─── Timeline segnali storici ──────────────────────────────────────────────────
st.markdown("#### 📅 Timeline Storica di Tutti i Segnali VVIX")
st.markdown(
    "Ogni punto rappresenta un evento estremo del VVIX nel tempo. "
    "L'asse Y mostra il livello VIX al momento del segnale (regime). "
    "La dimensione del punto è proporzionale al rendimento VIX assoluto a 20 giorni."
)

if not fwd_returns.empty:
    fig_timeline = build_signal_timeline(fwd_returns, horizon_key="vix_ret_20d", horizon_label="20d")
    st.plotly_chart(fig_timeline, width='stretch')

# ─── Log segnali (tabella dettaglio) ──────────────────────────────────────────
with st.expander("📋 Log completo di tutti i segnali storici"):
    if fwd_returns.empty:
        st.info("Nessun segnale rilevato con i parametri correnti.")
    else:
        log_df = fwd_returns[
            ["signal", "zscore", "vix_at_signal",
             "spx_ret_5d", "spx_ret_20d", "spx_ret_60d",
             "vix_ret_5d", "vix_ret_20d", "vix_ret_60d"]
        ].copy()
        log_df.index = log_df.index.strftime("%Y-%m-%d")
        log_df.index.name = "Data"
        log_df["Regime VIX"] = fwd_returns["vix_at_signal"].apply(_vix_regime).values
        log_df["signal"] = log_df["signal"].map(
            {"overbought": "🔴 OB", "oversold": "🟢 OS"}
        )
        log_df = log_df.rename(columns={
            "signal":        "Tipo",
            "zscore":        "ZScore",
            "vix_at_signal": "VIX@segnale",
            "spx_ret_5d":    "SPX 5d%",
            "spx_ret_20d":   "SPX 20d%",
            "spx_ret_60d":   "SPX 60d%",
            "vix_ret_5d":    "VIX 5d%",
            "vix_ret_20d":   "VIX 20d%",
            "vix_ret_60d":   "VIX 60d%",
        })
        log_df = log_df.sort_index(ascending=False)

        _fmt_log = {
            "ZScore": "{:.3f}", "VIX@segnale": "{:.2f}",
            "SPX 5d%": "{:+.2f}", "SPX 20d%": "{:+.2f}", "SPX 60d%": "{:+.2f}",
            "VIX 5d%": "{:+.2f}", "VIX 20d%": "{:+.2f}", "VIX 60d%": "{:+.2f}",
        }
        st.dataframe(
            log_df.style
                .format(_fmt_log, na_rep="—")
                .background_gradient(subset=["VIX 20d%", "SPX 20d%"], cmap="RdYlGn"),
            width='stretch',
            height=400,
        )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# [6] FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "⚠️ **Disclaimer:** questa dashboard ha finalità esclusivamente educative e di "
    "ricerca quantitativa. Le analisi storiche non garantiscono risultati futuri e "
    "non costituiscono consulenza finanziaria. | "
    "**Kriterion Quant** — [kriterionquant.com](https://kriterionquant.com)"
)

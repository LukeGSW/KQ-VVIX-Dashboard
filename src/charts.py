"""
charts.py — Grafici Plotly dark theme per VVIX Behavior Dashboard.

Tutte le funzioni restituiscono go.Figure pronto per st.plotly_chart().
Palette colori e layout centralizzati per coerenza visiva.

Grafici disponibili:
    build_vvix_chart()            : VVIX price + log-zScore subplot + markers segnali
    build_zscore_recent()         : log-zScore ultimi 252 giorni (header operativo)
    build_forward_returns_bar()   : rendimenti medi forward con banda interquartile
    build_distribution_chart()    : istogramma distribuzione rendimenti
    build_conditional_heatmap()   : heatmap regime VIX × orizzonte
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === PALETTE COLORI DARK THEME ===
COLORS = {
    "primary":    "#2196F3",   # blu   — VVIX price, SPX
    "secondary":  "#FF9800",   # arancio — VIX, secondarie
    "positive":   "#4CAF50",   # verde — rendimenti positivi
    "negative":   "#F44336",   # rosso — rendimenti negativi
    "neutral":    "#9E9E9E",   # grigio — linee di riferimento
    "background": "#1E1E2E",   # sfondo
    "surface":    "#2A2A3E",   # card / pannelli
    "text":       "#E0E0E0",   # testo principale
    "accent":     "#AB47BC",   # viola — log-zScore
    "ob_marker":  "#FF5252",   # rosso chiaro — markers overbought
    "os_marker":  "#69F0AE",   # verde menta — markers oversold
}


def _base_layout(title: str, x_title: str = "", y_title: str = "") -> dict:
    """Layout Plotly condiviso da tutti i grafici della dashboard."""
    return dict(
        title=dict(text=title, font=dict(size=15, color=COLORS["text"])),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif"),
        xaxis=dict(
            title=x_title,
            showgrid=True, gridcolor="#333355",
            zeroline=False, color=COLORS["text"],
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True, gridcolor="#333355",
            zeroline=False, color=COLORS["text"],
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#444466"),
        hovermode="x unified",
        margin=dict(l=60, r=30, t=65, b=55),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. GRAFICO PRINCIPALE: VVIX + LOG-ZSCORE
# ─────────────────────────────────────────────────────────────────────────────

def build_vvix_chart(
    df: pd.DataFrame,
    zscore: pd.Series,
    events: pd.DataFrame,
    upper_thresh: float,
    lower_thresh: float,
) -> go.Figure:
    """
    Grafico a due subplot: VVIX price (top) + log-zScore (bottom).

    Subplot superiore: serie storica VVIX con markers colorati sugli eventi.
        - Triangolo rosso ↑ = overbought
        - Triangolo verde ↓ = oversold

    Subplot inferiore: log-zScore con linee soglia e zona overbought evidenziata.

    Args:
        df:           DataFrame con colonna 'vvix', DatetimeIndex.
        zscore:       Serie log-zScore (già dropna).
        events:       DataFrame eventi (index = date, col 'signal').
        upper_thresh: Soglia overbought.
        lower_thresh: Soglia oversold.

    Returns:
        go.Figure a due subplot verticali con asse X condiviso.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.55, 0.45],
        subplot_titles=[
            "VVIX — Serie Storica",
            f"Log-ZScore rolling (finestra configurabile) | Soglie: {lower_thresh} / +{upper_thresh}",
        ],
    )

    # ── Top: VVIX price ──
    fig.add_trace(go.Scatter(
        x=df.index, y=df["vvix"],
        name="VVIX",
        line=dict(color=COLORS["primary"], width=1.5),
        hovertemplate="VVIX: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Markers overbought
    ob = events[events["signal"] == "overbought"]
    if not ob.empty:
        vvix_ob = df["vvix"].reindex(ob.index).dropna()
        fig.add_trace(go.Scatter(
            x=vvix_ob.index, y=vvix_ob.values,
            name="Overbought",
            mode="markers",
            marker=dict(
                symbol="triangle-up", size=11,
                color=COLORS["ob_marker"],
                line=dict(color="white", width=0.5),
            ),
            hovertemplate="<b>OB</b> %{x|%Y-%m-%d}<br>VVIX: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # Markers oversold
    os_ = events[events["signal"] == "oversold"]
    if not os_.empty:
        vvix_os = df["vvix"].reindex(os_.index).dropna()
        fig.add_trace(go.Scatter(
            x=vvix_os.index, y=vvix_os.values,
            name="Oversold",
            mode="markers",
            marker=dict(
                symbol="triangle-down", size=11,
                color=COLORS["os_marker"],
                line=dict(color="white", width=0.5),
            ),
            hovertemplate="<b>OS</b> %{x|%Y-%m-%d}<br>VVIX: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # ── Bottom: log-zScore ──
    fig.add_trace(go.Scatter(
        x=zscore.index, y=zscore.values,
        name="Log-ZScore",
        line=dict(color=COLORS["accent"], width=1.5),
        hovertemplate="ZScore: %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    # Zona overbought evidenziata (fill sopra la soglia superiore)
    z_clipped_ob = zscore.where(zscore >= upper_thresh)
    fig.add_trace(go.Scatter(
        x=zscore.index, y=z_clipped_ob,
        fill="tozeroy",
        fillcolor="rgba(255,82,82,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Zona OB",
        showlegend=False,
        hoverinfo="skip",
    ), row=2, col=1)

    # Zona oversold evidenziata (fill sotto la soglia inferiore)
    z_clipped_os = zscore.where(zscore <= lower_thresh)
    fig.add_trace(go.Scatter(
        x=zscore.index, y=z_clipped_os,
        fill="tozeroy",
        fillcolor="rgba(105,240,174,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Zona OS",
        showlegend=False,
        hoverinfo="skip",
    ), row=2, col=1)

    # Linee soglia e zero
    for y_val, color, label in [
        (upper_thresh, COLORS["ob_marker"], f"OB +{upper_thresh}"),
        (lower_thresh, COLORS["os_marker"], f"OS {lower_thresh}"),
        (0.0,          COLORS["neutral"],   ""),
    ]:
        fig.add_hline(
            y=y_val, row=2, col=1,
            line_dash="dash" if y_val != 0 else "dot",
            line_color=color,
            line_width=1.2 if y_val != 0 else 0.8,
            annotation_text=label if label else None,
            annotation_font_color=color,
            annotation_position="right",
        )

    # Layout globale
    fig.update_layout(
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif"),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#444466"),
        margin=dict(l=60, r=40, t=70, b=55),
        height=620,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#333355", zeroline=False, color=COLORS["text"])
    fig.update_yaxes(showgrid=True, gridcolor="#333355", zeroline=False, color=COLORS["text"])
    fig.update_yaxes(title_text="VVIX", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOG-ZSCORE RECENTE (HEADER OPERATIVO)
# ─────────────────────────────────────────────────────────────────────────────

def build_zscore_recent(
    zscore: pd.Series,
    upper_thresh: float,
    lower_thresh: float,
    lookback: int = 252,
) -> go.Figure:
    """
    Grafico compatto del log-zScore per gli ultimi `lookback` giorni.
    Usato nell'header operativo per visualizzare la posizione corrente.

    Args:
        zscore:      Serie log-zScore (già dropna).
        upper_thresh: Soglia overbought.
        lower_thresh: Soglia oversold.
        lookback:    Numero di barre da visualizzare (default: 252 = ~1 anno).

    Returns:
        go.Figure compatta (altezza 260px).
    """
    recent = zscore.iloc[-lookback:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent.values,
        name="Log-ZScore",
        line=dict(color=COLORS["accent"], width=1.8),
        fill="tozeroy",
        fillcolor="rgba(171,71,188,0.08)",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
    ))

    for y_val, color, dash in [
        (upper_thresh, COLORS["ob_marker"], "dash"),
        (lower_thresh, COLORS["os_marker"], "dash"),
        (0.0,          COLORS["neutral"],   "dot"),
    ]:
        fig.add_hline(y=y_val, line_dash=dash, line_color=color, line_width=1.0)

    fig.update_layout(
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif"),
        xaxis=dict(showgrid=True, gridcolor="#333355", zeroline=False, color=COLORS["text"]),
        yaxis=dict(
            title="Z-Score",
            showgrid=True, gridcolor="#333355",
            zeroline=False, color=COLORS["text"],
        ),
        hovermode="x unified",
        showlegend=False,
        height=260,
        margin=dict(l=55, r=20, t=30, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. FORWARD RETURNS — BARRE CON BANDA INTERQUARTILE
# ─────────────────────────────────────────────────────────────────────────────

def build_forward_returns_bar(
    stats_df: pd.DataFrame,
    signal_type: str,
    asset: str,
) -> go.Figure:
    """
    Grafico a barre: rendimento medio forward + error bar interquartile (P25/P75).

    Ogni barra rappresenta il rendimento medio per un orizzonte.
    Verde = media > 0, Rosso = media < 0.
    Le error bar mostrano la dispersione (P25 in basso, P75 in alto).
    Il testo mostra media e hit rate.

    Args:
        stats_df:    Output di compute_summary_stats (index = orizzonti).
        signal_type: 'overbought' | 'oversold'.
        asset:       'SPX' | 'VIX' (usato nel titolo).

    Returns:
        go.Figure con altezza 420px.
    """
    horizons  = stats_df.index.tolist()
    means     = stats_df["Media %"].values
    p25       = stats_df["P25 %"].values
    p75       = stats_df["P75 %"].values
    hit_rates = stats_df["Hit Rate %"].values
    pf_vals   = stats_df["Profit Factor"].values

    bar_colors = [
        COLORS["positive"] if (not np.isnan(m) and m > 0) else COLORS["negative"]
        for m in means
    ]

    error_plus  = [
        (p - m) if not (np.isnan(p) or np.isnan(m)) else 0
        for m, p in zip(means, p75)
    ]
    error_minus = [
        (m - p) if not (np.isnan(p) or np.isnan(m)) else 0
        for m, p in zip(means, p25)
    ]

    signal_label = "Overbought (zScore ↑)" if signal_type == "overbought" else "Oversold (zScore ↓)"

    text_labels = []
    for m, hr, pf in zip(means, hit_rates, pf_vals):
        if np.isnan(m):
            text_labels.append("N/D")
        else:
            pf_str = f"{pf:.2f}" if not np.isinf(pf) else "∞"
            text_labels.append(f"{m:+.2f}%<br>HR:{hr:.0f}% PF:{pf_str}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=horizons,
        y=means,
        name=f"Media {asset} %",
        marker_color=bar_colors,
        error_y=dict(
            type="data",
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
            color=COLORS["neutral"],
            thickness=1.8,
            width=7,
        ),
        text=text_labels,
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
        hovertemplate=(
            "<b>Orizzonte: %{x}</b><br>"
            "Media: %{y:.2f}%<br>"
            "<extra></extra>"
        ),
    ))

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], line_width=1)

    fig.update_layout(**_base_layout(
        title=f"{asset} — Rendimenti Forward dopo segnale {signal_label}",
        x_title="Orizzonte (giorni trading)",
        y_title="Rendimento Semplice (%)",
    ))
    fig.update_layout(height=420, showlegend=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. DISTRIBUZIONE RENDIMENTI (ISTOGRAMMA)
# ─────────────────────────────────────────────────────────────────────────────

def build_distribution_chart(
    fwd_returns: pd.DataFrame,
    signal_type: str,
    asset: str,
    horizon: int = 20,
) -> go.Figure:
    """
    Istogramma della distribuzione dei rendimenti forward per un orizzonte.

    Mostra la forma della distribuzione (code, asimmetria) con linea
    verticale sulla media e linea dello zero.

    Args:
        fwd_returns:  Output di compute_forward_returns.
        signal_type:  'overbought' | 'oversold'.
        asset:        'spx' | 'vix'.
        horizon:      Orizzonte in giorni da visualizzare.

    Returns:
        go.Figure con altezza 360px.
    """
    col    = f"{asset}_ret_{horizon}d"
    subset = fwd_returns[fwd_returns["signal"] == signal_type][col].dropna()

    signal_label = "Overbought" if signal_type == "overbought" else "Oversold"
    asset_upper  = asset.upper()

    fig = go.Figure()

    if len(subset) == 0:
        fig.add_annotation(
            text="Nessun dato disponibile",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=COLORS["neutral"]),
        )
    else:
        fig.add_trace(go.Histogram(
            x=subset,
            nbinsx=max(10, min(30, len(subset) // 2)),
            name=f"{asset_upper} ret {horizon}d",
            marker_color=COLORS["primary"],
            opacity=0.78,
            hovertemplate="Intervallo: %{x}<br>N eventi: %{y}<extra></extra>",
        ))

        mean_val   = subset.mean()
        median_val = subset.median()

        fig.add_vline(
            x=mean_val, line_dash="dash", line_color=COLORS["secondary"],
            annotation_text=f"Media: {mean_val:+.2f}%",
            annotation_font_color=COLORS["secondary"],
            annotation_position="top right",
        )
        fig.add_vline(
            x=median_val, line_dash="dot", line_color=COLORS["accent"],
            annotation_text=f"Mediana: {median_val:+.2f}%",
            annotation_font_color=COLORS["accent"],
            annotation_position="top left",
        )
        fig.add_vline(x=0.0, line_dash="solid", line_color=COLORS["neutral"], line_width=1)

    fig.update_layout(**_base_layout(
        title=f"Distribuzione {asset_upper} a {horizon}gg | Segnale: {signal_label} ({len(subset)} eventi)",
        x_title="Rendimento (%)",
        y_title="Frequenza",
    ))
    fig.update_layout(height=360)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. HEATMAP CONDIZIONALE (regime VIX × orizzonte)
# ─────────────────────────────────────────────────────────────────────────────

def build_conditional_heatmap(
    heatmap_data: pd.DataFrame,
    signal_type: str,
    asset: str,
) -> go.Figure:
    """
    Heatmap RdYlGn: rendimenti medi condizionati per regime VIX × orizzonte.

    Verde = rendimento medio positivo, Rosso = negativo, Giallo = neutro.
    I valori sono annotati in ogni cella.

    Args:
        heatmap_data: DataFrame (regimi come righe, orizzonti come colonne).
                      Output di compute_conditional_heatmap_data().
        signal_type:  'overbought' | 'oversold'.
        asset:        'spx' | 'vix'.

    Returns:
        go.Figure con altezza 360px.
    """
    signal_label = "Overbought" if signal_type == "overbought" else "Oversold"
    asset_upper  = asset.upper()

    z_vals  = heatmap_data.values.astype(float)
    x_labs  = heatmap_data.columns.tolist()
    y_labs  = heatmap_data.index.tolist()

    text_vals = [
        [f"{v:+.2f}%" if not np.isnan(v) else "—" for v in row]
        for row in z_vals
    ]

    # Calcola zmid simmetrico attorno allo zero per colorscale corretta
    abs_max = np.nanmax(np.abs(z_vals)) if not np.all(np.isnan(z_vals)) else 1.0

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=x_labs,
        y=y_labs,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        colorscale="RdYlGn",
        zmid=0,
        zmin=-abs_max,
        zmax=abs_max,
        hoverongaps=False,
        hovertemplate=(
            "Regime: %{y}<br>"
            "Orizzonte: %{x}<br>"
            "Rendimento medio: %{z:.2f}%<extra></extra>"
        ),
        colorbar=dict(
            title="Ret %",
            titlefont=dict(color=COLORS["text"]),
            tickfont=dict(color=COLORS["text"]),
            len=0.8,
        ),
    ))

    fig.update_layout(**_base_layout(
        title=f"{asset_upper} — {signal_label} VVIX per Regime VIX",
        x_title="Orizzonte Forward",
        y_title="Regime VIX al segnale",
    ))
    fig.update_layout(height=340)
    return fig

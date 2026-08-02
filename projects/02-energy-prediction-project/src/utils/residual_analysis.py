import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def plot_residual_analysis(y_test, y_pred, X_test):
    """
    Menghasilkan figure 2-panel Plotly untuk analisis residual model.

    Hanya dua chart yang dipertahankan:
    1. Distribusi Residual — membuktikan model tidak punya bias sistematis.
    2. Residual per Jam Operasional — memastikan konsistensi di seluruh periode.

    Chart yang dihapus (tidak relevan untuk XGBoost tree-based ensemble):
    - Q-Q Plot: menguji normalitas residual (asumsi regresi linear) — tidak diperlukan.
    - Scatter Aktual vs Prediksi: redundan dengan metrik R2 yang sudah ditampilkan.
    """
    # Calculate residuals
    residuals = y_test - y_pred
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Distribusi Residual",
            "Residual per Jam Operasional",
        ),
        column_widths=[0.45, 0.55],
        horizontal_spacing=0.10,
    )

    # Panel kiri: Distribusi Residual
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=60,
        name="Residual",
        marker_color='#636EFA',
        opacity=0.75,
    ), row=1, col=1)

    fig.add_vline(x=0, line_dash="dash", line_color="#EF553B", line_width=1.5, row=1, col=1)

    fig.add_annotation(
        x=0.04, y=0.96,
        xref="x domain", yref="y domain",
        text=f"Mean: {mean_res:,.2f} W<br>Std: {std_res:,.2f} W",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#CBD5E1",
        borderwidth=1,
        font=dict(color="#1C1B1B", size=12),
        align="left",
        row=1, col=1,
    )

    # Panel kanan: Residual per Jam Operasional
    fig.add_trace(go.Box(
        x=X_test['Hour'],
        y=residuals,
        name="Residual per Jam",
        marker_color='#AB63FA',
        line=dict(color='#7C3AED'),
        fillcolor='rgba(171,99,250,0.18)',
        boxmean='sd',
    ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dash", line_color="#EF553B", line_width=1.5, row=1, col=2)

    # Layout
    fig.update_layout(
        height=440,
        template='plotly_white',
        showlegend=False,
        margin=dict(l=20, r=20, t=48, b=20),
        font=dict(family='Inter', color='#61707A'),
    )

    fig.update_xaxes(title_text="Residual (W)", title_font=dict(size=12), row=1, col=1)
    fig.update_yaxes(title_text="Frekuensi", title_font=dict(size=12), row=1, col=1)

    fig.update_xaxes(
        title_text="Jam Operasional (0-23)",
        title_font=dict(size=12),
        tickmode='linear', tick0=0, dtick=2,
        row=1, col=2,
    )
    fig.update_yaxes(title_text="Residual (W)", title_font=dict(size=12), row=1, col=2)

    return fig

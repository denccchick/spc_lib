import numpy as np
import plotly.graph_objects as go

# ==========================================
# Цвета
# ==========================================
COLOR_SBER_GREEN = "#21a358"
COLOR_SBER_RED = "#dc3545"
COLOR_CL = "#999999"


def _plot_single_chart(
        dates,
        stats,
        ucl,
        cl,
        lcl,
        title):

    stats = np.asarray(stats)

    violations = (stats > ucl) | (stats < lcl)

    marker_colors = np.where(
        violations,
        COLOR_SBER_RED,
        COLOR_SBER_GREEN
    )

    marker_sizes = np.where(
        violations,
        14,
        10
    )

    fig = go.Figure()

    # UCL
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[ucl] * len(dates),
            mode="lines",
            line=dict(
                color=COLOR_SBER_RED,
                width=2.5
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )

    # LCL
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[lcl] * len(dates),
            mode="lines",
            line=dict(
                color=COLOR_SBER_RED,
                width=2.5
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )

    # CL
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[cl] * len(dates),
            mode="lines",
            line=dict(
                color=COLOR_CL,
                width=2,
                dash="dash"
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )

    # Основная линия
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stats,
            mode="lines+markers",
            line=dict(
                color=COLOR_SBER_GREEN,
                width=3.5
            ),
            marker=dict(
                color=marker_colors,
                size=marker_sizes,
                line=dict(
                    color="white",
                    width=2
                )
            ),
            hovertemplate=
            "Дата: %{x}<br>"
            "Значение: <b>%{y:.4f}</b>"
            "<extra></extra>",
            showlegend=False
        )
    )

    fig.update_layout(
        height=450,
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor="center"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        margin=dict(
            l=60,
            r=30,
            t=70,
            b=50
        )
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="#E0E0E0",
        linewidth=1,
        tickfont=dict(
            color="#A0A0A0",
            size=12
        )
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#F5F5F5",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor="#E0E0E0",
        linewidth=1,
        tickfont=dict(
            color="#A0A0A0",
            size=12
        )
    )

    return fig


def plot_control_chart(
        chart,
        start=None,
        end=None,
        last_n=30):

    dates = np.asarray(chart.datetimes)

    if start is not None:
        mask = dates >= np.datetime64(start)

    elif end is not None:
        mask = dates <= np.datetime64(end)

    else:
        mask = np.arange(len(dates)) >= max(0, len(dates) - last_n)

    # X-bar
    fig_xbar = _plot_single_chart(
        dates[mask],
        np.asarray(chart.stat_main)[mask],
        chart.ucl_main,
        chart.cl_main,
        chart.lcl_main,
        "График средних (X-bar)"
    )

    # R
    fig_r = _plot_single_chart(
        dates[mask],
        np.asarray(chart.stat_disp)[mask],
        chart.ucl_disp,
        chart.cl_disp,
        chart.lcl_disp,
        "График размаха (R)"
    )

    return fig_xbar, fig_r

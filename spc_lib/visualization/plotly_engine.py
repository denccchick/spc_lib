import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List
from spc_lib.rules import detect_violations

COLOR_MAIN = "#21a358"
COLOR_RED = "#dc3545"
COLOR_CL = "#999999"
COLOR_TARGET = "#121111"
COLOR_SPEC = "#FF0000"

RULE_COLORS = {
    1: '#FF0000', 2: '#FF6B00', 3: '#FFD700', 4: '#32CD32',
    5: '#1E90FF', 6: '#8A2BE2', 7: '#FF1493', 8: '#00CED1'
}

RULE_NAMES = {
    1: 'Rule 1: Точка за 3s',
    2: 'Rule 2: 9 точек на одной стороне',
    3: 'Rule 3: 6 точек с трендом',
    4: 'Rule 4: 14 точек с чередованием',
    5: 'Rule 5: 2 из 3 в зоне A',
    6: 'Rule 6: 4 из 5 в зоне B',
    7: 'Rule 7: 15 точек в зоне C',
    8: 'Rule 8: 8 точек вне зоны C'
}


def _plot_single_chart(dates, stats, ucl, cl, lcl, title, target=None, usl=None, lsl=None, show_spec=False):
    stats = np.asarray(stats)
    valid = ~np.isnan(stats)
    stats = stats[valid]
    dates = dates[valid]

    violations = (stats > ucl) | (stats < lcl)

    marker_colors = np.where(violations, COLOR_RED, COLOR_MAIN)
    marker_sizes = np.where(violations, 14, 10)

    fig = go.Figure()

    # UCL
    fig.add_trace(go.Scatter(
        x=dates, y=[ucl] * len(dates), mode="lines",
        line=dict(color=COLOR_RED, width=2.5), showlegend=False, hoverinfo='skip'
    ))

    # LCL
    fig.add_trace(go.Scatter(
        x=dates, y=[lcl] * len(dates), mode="lines",
        line=dict(color=COLOR_RED, width=2.5), showlegend=False, hoverinfo='skip'
    ))

    # CL
    fig.add_trace(go.Scatter(
        x=dates, y=[cl] * len(dates), mode="lines",
        line=dict(color=COLOR_CL, width=2, dash='dash'), showlegend=False, hoverinfo='skip'
    ))

    # TARGET (если задан)
    if target is not None:
        fig.add_trace(go.Scatter(
            x=dates, y=[target] * len(dates), mode="lines",
            line=dict(color=COLOR_TARGET, width=2, dash='dot'),
            name='Target'
        ))

    # USL/LSL (если заданы и show_spec=True)
    if show_spec:
        if usl is not None:
            fig.add_trace(go.Scatter(
                x=dates, y=[usl] * len(dates), mode="lines",
                line=dict(color=COLOR_SPEC, width=2, dash='dashdot'),
                name='USL'
            ))
        if lsl is not None:
            fig.add_trace(go.Scatter(
                x=dates, y=[lsl] * len(dates), mode="lines",
                line=dict(color=COLOR_SPEC, width=2, dash='dashdot'),
                name='LSL'
            ))

    # Основная статистика
    fig.add_trace(go.Scatter(
        x=dates, y=stats, mode='lines+markers',
        line=dict(color=COLOR_MAIN, width=3.5),
        marker=dict(color=marker_colors, size=marker_sizes, line=dict(color='white', width=2)),
        hovertemplate="Дата: %{x}<br>Значение: <b>%{y:.4f}</b><extra></extra>",
        showlegend=False
    ))

    fig.update_layout(
        height=450,
        title=dict(text=f"<b>{title}</b>", x=0.5),
        plot_bgcolor='white', paper_bgcolor='white',
        hovermode='x unified',
        margin=dict(l=60, r=30, t=70, b=50)
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linecolor="#E0E0E0", tickfont=dict(color="#A0A0A0", size=12))
    fig.update_yaxes(showgrid=True, gridcolor="#F5F5F5", gridwidth=1, zeroline=False, showline=True, linecolor="#E0E0E0", tickfont=dict(color="#A0A0A0", size=12))

    return fig


def plot_control_chart(chart, start=None, end=None, last_n=30, show_spec=False):
    dates = np.asarray(chart.datetimes)

    if start is not None:
        mask = dates >= np.datetime64(start)
    elif end is not None:
        mask = dates <= np.datetime64(end)
    else:
        mask = np.arange(len(dates)) >= (len(dates) - last_n)

    fig_main = _plot_single_chart(
        dates[mask],
        np.asarray(chart.stat_main)[mask],
        chart.ucl_main, chart.cl_main, chart.lcl_main,
        chart.main_label,
        target=chart.target,
        usl=chart.usl,
        lsl=chart.lsl,
        show_spec=show_spec
    )

    if chart.stat_disp is None:
        return fig_main

    fig_disp = _plot_single_chart(
        dates[mask],
        np.asarray(chart.stat_disp)[mask],
        chart.ucl_disp, chart.cl_disp, chart.lcl_disp,
        chart.disp_label,
        target=None,
        usl=None,
        lsl=None,
        show_spec=False
    )

    return fig_main, fig_disp


def plot_rules_violations(chart, start=None, end=None, last_n=30, rules=None, n_cols=2, show_spec=False):
    dates = np.asarray(chart.datetimes)

    if start is not None:
        mask = dates >= np.datetime64(start)
    elif end is not None:
        mask = dates <= np.datetime64(end)
    else:
        mask = np.arange(len(dates)) >= (len(dates) - last_n)

    stats_main = np.asarray(chart.stat_main)[mask]
    dates_main = dates[mask]

    sigma_main = (chart.ucl_main - chart.cl_main) / 3
    violations_by_rule = detect_violations(
        data=stats_main,
        center=chart.cl_main,
        sigma=sigma_main,
        last_n=None,
        rules=rules
    )

    active_rules = {r: indices for r, indices in violations_by_rule.items() if indices}

    if not active_rules:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="<b>Нарушений правил Western Electric не обнаружено</b>",
            showarrow=False, font=dict(size=16, color="#666666")
        )
        fig.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white')
        fig.update_xaxes(showgrid=False, zeroline=False, showline=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showline=False)
        return fig

    rule_list = list(active_rules.keys())
    n_figs = len(rule_list)
    n_rows = (n_figs + n_cols - 1) // n_cols

    subplot_titles = [RULE_NAMES[r] for r in rule_list]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    stats = stats_main
    valid = ~np.isnan(stats)
    stats = stats[valid]
    dates_plot = dates_main[valid]

    if len(dates_plot) > 0:
        if isinstance(dates_plot[0], np.datetime64):
            dates_plot = dates_plot.astype('datetime64[ms]').astype('O')
        elif isinstance(dates_plot[0], str):
            dates_plot = np.array([np.datetime64(d) for d in dates_plot])

    n_points = len(stats)

    rule_idx = 0
    for rule_num, indices in active_rules.items():
        row = rule_idx // n_cols + 1
        col = rule_idx % n_cols + 1

        rule_violations = np.zeros(n_points, dtype=bool)
        for idx in indices:
            if idx < n_points:
                rule_violations[idx] = True

        marker_colors_rule = np.where(rule_violations, RULE_COLORS.get(rule_num, COLOR_RED), COLOR_MAIN)
        marker_sizes_rule = np.where(rule_violations, 14, 10)

        # UCL
        fig.add_trace(go.Scatter(
            x=dates_plot, y=[chart.ucl_main] * n_points, mode="lines",
            line=dict(color=COLOR_RED, width=2.5), showlegend=False, hoverinfo='skip'
        ), row=row, col=col)

        # LCL
        fig.add_trace(go.Scatter(
            x=dates_plot, y=[chart.lcl_main] * n_points, mode="lines",
            line=dict(color=COLOR_RED, width=2.5), showlegend=False, hoverinfo='skip'
        ), row=row, col=col)

        # CL
        fig.add_trace(go.Scatter(
            x=dates_plot, y=[chart.cl_main] * n_points, mode="lines",
            line=dict(color=COLOR_CL, width=2, dash='dash'), showlegend=False, hoverinfo='skip'
        ), row=row, col=col)

        # Target (если задан)
        if chart.target is not None:
            fig.add_trace(go.Scatter(
                x=dates_plot, y=[chart.target] * n_points, mode="lines",
                line=dict(color=COLOR_TARGET, width=2, dash='dot'),
                showlegend=False, hoverinfo='skip'
            ), row=row, col=col)

        # USL/LSL (если заданы и show_spec=True)
        if show_spec:
            if chart.usl is not None:
                fig.add_trace(go.Scatter(
                    x=dates_plot, y=[chart.usl] * n_points, mode="lines",
                    line=dict(color=COLOR_SPEC, width=2, dash='dashdot'),
                    showlegend=False, hoverinfo='skip'
                ), row=row, col=col)
            if chart.lsl is not None:
                fig.add_trace(go.Scatter(
                    x=dates_plot, y=[chart.lsl] * n_points, mode="lines",
                    line=dict(color=COLOR_SPEC, width=2, dash='dashdot'),
                    showlegend=False, hoverinfo='skip'
                ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=dates_plot, y=stats, mode='lines+markers',
            line=dict(color=COLOR_MAIN, width=3.5),
            marker=dict(
                color=marker_colors_rule,
                size=marker_sizes_rule,
                line=dict(color='white', width=2)
            ),
            hovertemplate="Дата: %{x}<br>Значение: <b>%{y:.4f}</b><extra></extra>",
            showlegend=False
        ), row=row, col=col)

        fig.update_xaxes(
            showgrid=False, zeroline=False, showline=True,
            linecolor="#E0E0E0", tickfont=dict(color="#A0A0A0", size=9),
            row=row, col=col
        )

        fig.update_yaxes(
            showgrid=True, gridcolor="#F5F5F5", gridwidth=1,
            zeroline=False, showline=True, linecolor="#E0E0E0",
            tickfont=dict(color="#A0A0A0", size=9),
            row=row, col=col
        )

        rule_idx += 1

    height = max(400, 400 * n_rows)

    fig.update_layout(
        height=height,
        plot_bgcolor='white', paper_bgcolor='white',
        hovermode='x unified',
        title=dict(
            text=f"<b>Нарушения правил Western Electric</b><br>"
                 f"<span style='font-size:14px;color:#666;'>"
                 f"Всего нарушений: {sum(len(v) for v in active_rules.values())} в правилах {', '.join(map(str, active_rules.keys()))}"
                 f"</span>",
            x=0.5
        ),
        margin=dict(l=50, r=30, t=100, b=50)
    )

    return fig

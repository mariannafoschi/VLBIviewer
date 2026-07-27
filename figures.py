"""Plotly figure builders for the uvfits viewer."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------- palette --
# Colour-blind-safe categorical hues, stepped separately for each surface.
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500",
             "#d55181", "#008300", "#9085e9", "#e66767"],
}
# "warm" and "teal" only change the *surface*, so they reuse the hue steps
# already validated for a light and a dark background; check_theme_contrast.py
# recomputes the one check that does depend on the surface.
PALETTE["warm"] = PALETTE["light"]
PALETTE["teal"] = PALETTE["dark"]

# single colour for every series, when the user asks for one flat colour
MONO = {"light": "#2a78d6", "dark": "#3987e5",
        "warm": "#2a78d6", "teal": "#3987e5"}

# themes that sit on a dark surface (picks the plotly template and CSS class)
DARK_THEMES = {"dark", "teal"}
THEMES = [("Light", "light"), ("Warm", "warm"),
          ("Dark", "dark"), ("Teal", "teal")]
# A second channel so identity never rests on hue alone: 8 hues x 4 symbols
SYMBOLS = ["circle", "square", "diamond", "triangle-up"]

INK = {
    "light": dict(surface="#fcfcfb", plane="#f9f9f7", primary="#0b0b0b",
                  secondary="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7"),
    "dark": dict(surface="#1a1a19", plane="#0d0d0d", primary="#ffffff",
                 secondary="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835"),
    # warm off-white paper
    "warm": dict(surface="#faf5ec", plane="#f4ede0", primary="#1c1813",
                 secondary="#544c3d", muted="#8a8271", grid="#e7dfcf",
                 axis="#cdc3ae"),
    # deep blue-teal
    "teal": dict(surface="#0f2027", plane="#0a171c", primary="#f2fafc",
                 secondary="#a9c8ce", muted="#7d9ba3", grid="#1c3742",
                 axis="#2b4c58"),
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# phase-like quantities live on a circle and are drawn wrapped into this range
PHASE_RANGE = (-180.0, 180.0)
PHASE_TICKS = dict(tickmode="array", tickvals=[-180, -120, -60, 0, 60, 120, 180])


def style_map(names) -> dict:
    """Stable colour+symbol per series name (order-independent within a run)."""
    out = {}
    for i, n in enumerate(sorted(names)):
        out[n] = {"ci": i % 8, "symbol": SYMBOLS[(i // 8) % len(SYMBOLS)]}
    return out


def dataset_style_map(labels) -> dict:
    """One colour AND one marker per dataset, for comparison mode.

    Colour steps every dataset and the marker steps every time the colour wraps,
    so the pairs stay unique for up to 8x4 = 32 datasets (the first eight share
    the circle and differ only by colour, then squares, and so on).  Order
    follows the given ``labels`` so the header legend and the plots agree.
    """
    return {lab: {"ci": i % 8, "symbol": SYMBOLS[(i // 8) % len(SYMBOLS)]}
            for i, lab in enumerate(labels)}


def series_color(theme, style, mono=False):
    return MONO[theme] if mono else PALETTE[theme][style["ci"]]


def _hover_template(name, hover, text_col=None):
    """Build a hovertemplate: bold series name, an optional per-point label
    (``%{text}``, used for the baseline/triangle when colour encodes the
    dataset), then each ``(label, column, fmt)`` from ``hover``.  A blank
    ``fmt`` is emitted without a format spec so string columns render cleanly.
    """
    tmpl = f"<b>{name}</b><br>"
    if text_col is not None:
        tmpl += "%{text}<br>"
    for k, (label, _, fmt) in enumerate(hover):
        ref = f"customdata[{k + 1}]"
        tmpl += f"{label}: %{{{ref}:{fmt}}}<br>" if fmt else f"{label}: %{{{ref}}}<br>"
    return tmpl + "<extra></extra>"


def robust_range(values, pad=0.15, q=(1.0, 99.0)):
    """Y range that ignores the low-S/N tail; double-click restores autorange."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 10:
        return None
    lo, hi = np.percentile(v, q)
    if hi <= lo:
        return None
    m = (hi - lo) * pad
    return [lo - m, hi + m]


def wrap_phase(values, lo=PHASE_RANGE[0], hi=PHASE_RANGE[1]):
    """Fold angles into [lo, hi)."""
    period = hi - lo
    return (np.asarray(values, float) - lo) % period + lo


def wrapped_error_segments(x, y, err, lo=PHASE_RANGE[0], hi=PHASE_RANGE[1]):
    """Error bars for a circular quantity, as line segments that wrap.

    A bar reaching past +180 continues from -180, so nothing is drawn outside
    the range and no information is lost to clipping.  Returns x/y arrays with
    ``None`` separators, ready for a single ``mode="lines"`` trace.
    """
    x = np.asarray(x, float)
    y = wrap_phase(y, lo, hi)
    err = np.asarray(err, float)
    period = hi - lo
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(err) & (err > 0)
    x, y, err = x[good], y[good], err[good]

    full = 2 * err >= period               # bar covers the whole circle
    a, b = y - err, y + err
    under = (a < lo) & ~full               # spills past the bottom edge
    over = (b > hi) & ~full                # spills past the top edge

    # main segment, clipped to the range
    seg_lo = np.where(full, lo, np.maximum(a, lo))
    seg_hi = np.where(full, hi, np.minimum(b, hi))
    xs = [np.repeat(x, 3)]
    ys = [np.column_stack([seg_lo, seg_hi, np.full(x.size, np.nan)]).ravel()]

    # the piece that wrapped around, drawn on the opposite edge
    for mask, w_lo, w_hi in ((under, a + period, np.full(x.size, hi)),
                             (over, np.full(x.size, lo), b - period)):
        if mask.any():
            xs.append(np.repeat(x[mask], 3))
            ys.append(np.column_stack([np.asarray(w_lo)[mask],
                                       np.asarray(w_hi)[mask],
                                       np.full(mask.sum(), np.nan)]).ravel())
    xs = np.concatenate(xs) if xs else np.array([])
    ys = np.concatenate(ys) if ys else np.array([])
    # ``None`` rather than NaN so Scattergl breaks the line
    return xs, np.where(np.isnan(ys), None, ys)


def apply_theme(fig, theme, height=None, title=None):
    ink = INK[theme]
    fig.update_layout(
        template="plotly_dark" if theme in DARK_THEMES else "plotly_white",
        paper_bgcolor=ink["surface"], plot_bgcolor=ink["surface"],
        font=dict(family=FONT, size=12, color=ink["secondary"]),
        title=dict(text=title, font=dict(size=14, color=ink["primary"])) if title else None,
        margin=dict(l=64, r=16, t=44 if title else 22, b=48),
        hovermode="closest",
        dragmode="zoom",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11),
                    itemsizing="constant", tracegroupgap=2),
        # explicit modebar colours: the plotly_white default icons are near-white
        # and vanish on a light surface
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=ink["muted"],
                     activecolor=ink["primary"]),
        uirevision="keep",
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(showgrid=True, gridcolor=ink["grid"], gridwidth=1,
                     zeroline=False, linecolor=ink["axis"],
                     tickfont=dict(color=ink["muted"], size=11),
                     title_font=dict(color=ink["secondary"], size=12))
    fig.update_yaxes(showgrid=True, gridcolor=ink["grid"], gridwidth=1,
                     zeroline=False, linecolor=ink["axis"],
                     tickfont=dict(color=ink["muted"], size=11),
                     title_font=dict(color=ink["secondary"], size=12))
    return fig


def _empty(theme, msg="No data for this selection"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False,
                       font=dict(size=13, color=INK[theme]["muted"]))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_theme(fig, theme, height=260)


def _scatter(gl):
    """The scatter trace class to use.

    ``go.Scattergl`` (WebGL) is fast for the many-baseline single-file view, but
    Safari's WebGL silently drops overlaid gl traces -- so comparison mode, with
    one big trace per dataset, renders as plain SVG ``go.Scatter`` instead, which
    every browser draws reliably.
    """
    return go.Scattergl if gl else go.Scatter


def _error_trace(x, y, err, color, name, wrap, gl=True):
    """Wrapped error bars as a line trace (used for phase-like quantities)."""
    xs, ys = wrapped_error_segments(x, y, err, *wrap)
    return _scatter(gl)(x=xs, y=ys, mode="lines", name=name, legendgroup=name,
                        line=dict(width=1, color=color), hoverinfo="skip",
                        showlegend=False, connectgaps=False)


# --------------------------------------------------------------- scatters --
def grouped_scatter(df, x, y, err, group, theme, *, xlabel="", ylabel="",
                    hover=(), errorbars=True, facet=False, height=None,
                    yrange=None, logy=False, marker_size=6, title=None,
                    styles=None, hline=None, showlegend=True, xrange=None,
                    connect=False, mono=False, same_symbol=False,
                    wrap=None, text_col=None, gl=True):
    """Scatter of ``y`` vs ``x`` split by ``group``, one trace per series.

    ``hover`` is a list of (label, column, format) shown in the tooltip; the
    row index of the source dataframe always travels as ``customdata[0]`` so a
    click can be resolved back to the full record.  ``wrap=(lo, hi)`` marks the
    quantity as circular: points are folded into the range and error bars are
    drawn wrapped instead of running off the axis.
    """
    if df is None or len(df) == 0:
        return _empty(theme)

    names = sorted(dict.fromkeys(df[group]))
    styles = styles or style_map(names)
    ink = INK[theme]
    if wrap:
        yrange = list(wrap)

    if facet:
        nrow = len(names)
        height = height or max(240, 150 * nrow)
        fig = make_subplots(rows=nrow, cols=1, shared_xaxes=True,
                            vertical_spacing=min(0.03, 12.0 / height))
    else:
        fig = go.Figure()

    cust_cols = [c for _, c, _ in hover]
    for i, name in enumerate(names):
        g = df[df[group] == name]
        st = styles.get(name, {"ci": i % 8, "symbol": "circle"})
        color = series_color(theme, st, mono)
        cd = np.column_stack([g.index.values.astype(float)]
                             + [g[c].values for c in cust_cols])
        tmpl = _hover_template(name, hover, text_col)
        yvals = wrap_phase(g[y], *wrap) if wrap else g[y]
        trace = _scatter(gl)(
            x=g[x], y=yvals, name=name, legendgroup=name,
            mode="lines+markers" if connect else "markers",
            marker=dict(size=marker_size, color=color, symbol="circle" if same_symbol else st["symbol"],
                        line=dict(width=0)),
            line=dict(width=1, color=color) if connect else None,
            error_y=dict(type="data", array=g[err], visible=True, width=0,
                         thickness=1, color=color)
            if (errorbars and err and not wrap) else None,
            customdata=cd, hovertemplate=tmpl,
            text=g[text_col] if text_col is not None else None,
            showlegend=showlegend and not facet,
        )
        row = dict(row=i + 1, col=1) if facet else {}
        fig.add_trace(trace, **row)
        if errorbars and err and wrap:
            fig.add_trace(_error_trace(g[x], g[y], g[err], color, name, wrap, gl),
                          **row)
        if facet:
            # series name inside its own panel: no collision with the axis above
            fig.add_annotation(
                text=name, xref=f"x{i + 1 if i else ''} domain",
                yref=f"y{i + 1 if i else ''} domain", x=0.008, y=0.94,
                xanchor="left", yanchor="top", showarrow=False,
                font=dict(size=11, color=color),
                bgcolor=ink["surface"], borderpad=2, opacity=0.92)

    if hline is not None:
        for r in (range(1, len(names) + 1) if facet else [None]):
            fig.add_hline(y=hline, line=dict(color=ink["axis"], width=1,
                                             dash="dot"), row=r, col=1 if r else None)

    apply_theme(fig, theme, height=height, title=title)
    fig.update_layout(legend=dict(font=dict(size=10 if len(names) > 14 else 11)))
    fig.update_xaxes(title_text=xlabel, range=xrange,
                     row=len(names) if facet else None, col=1 if facet else None)
    yaxis_kw = dict(range=yrange, type="log" if logy else "linear")
    if wrap:
        yaxis_kw.update(PHASE_TICKS)
    if facet:
        fig.update_yaxes(title_text=None, **yaxis_kw)
        fig.update_layout(height=height, yaxis=dict(title=None))
        fig.add_annotation(text=ylabel, textangle=-90, xref="paper", yref="paper",
                           x=-0.058, y=0.5, showarrow=False,
                           font=dict(size=12, color=ink["secondary"]))
    else:
        fig.update_yaxes(title_text=ylabel, **yaxis_kw)
    return fig


def multi_panel(df, x, panels, group, theme, *, xlabel="", hover=(),
                styles=None, errorbars=True, height=None, connect=False,
                marker_size=6, title=None, mono=False, same_symbol=False,
                text_col=None, gl=True):
    """Stacked panels (one per quantity) sharing the x axis.

    Zooming any panel zooms them all, and one legend entry per series drives
    every panel through a shared legendgroup.  A panel may declare
    ``wrap=(lo, hi)`` to be drawn as a circular quantity.
    """
    if df is None or len(df) == 0 or not panels:
        return _empty(theme)
    ink = INK[theme]
    names = sorted(dict.fromkeys(df[group]))
    styles = styles or style_map(names)
    n = len(panels)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        vertical_spacing=0.055 if n > 1 else 0.02)
    cust_cols = [c for _, c, _ in hover]
    for i, name in enumerate(names):
        g = df[df[group] == name]
        st = styles.get(name, {"ci": i % 8, "symbol": "circle"})
        color = series_color(theme, st, mono)
        cd = np.column_stack([g.index.values.astype(float)]
                             + [g[c].values for c in cust_cols])
        tmpl = _hover_template(name, hover, text_col)
        txt = g[text_col] if text_col is not None else None
        for r, p in enumerate(panels, start=1):
            err, wrap = p.get("err"), p.get("wrap")
            yvals = wrap_phase(g[p["y"]], *wrap) if wrap else g[p["y"]]
            show_err = errorbars and err and err in g
            fig.add_trace(_scatter(gl)(
                x=g[x], y=yvals, name=name, legendgroup=name,
                mode="lines+markers" if connect else "markers",
                marker=dict(size=marker_size, color=color, symbol="circle" if same_symbol else st["symbol"],
                            line=dict(width=0)),
                line=dict(width=1, color=color) if connect else None,
                error_y=dict(type="data", array=g[err], visible=True, width=0,
                             thickness=1, color=color)
                if (show_err and not wrap) else None,
                customdata=cd, hovertemplate=tmpl, text=txt,
                showlegend=(r == 1)), row=r, col=1)
            if show_err and wrap:
                fig.add_trace(
                    _error_trace(g[x], g[p["y"]], g[err], color, name, wrap, gl),
                    row=r, col=1)
    for r, p in enumerate(panels, start=1):
        if p.get("hline") is not None:
            fig.add_hline(y=p["hline"], line=dict(color=ink["axis"], width=1,
                                                  dash="dot"), row=r, col=1)
    apply_theme(fig, theme, height=height or max(300, 230 * n), title=title)
    fig.update_layout(legend=dict(font=dict(size=10 if len(names) > 14 else 11)))
    for r, p in enumerate(panels, start=1):
        kw = dict(PHASE_TICKS) if p.get("wrap") else {}
        fig.update_yaxes(title_text=p["label"], row=r, col=1,
                         type="log" if p.get("logy") else "linear",
                         range=list(p["wrap"]) if p.get("wrap") else p.get("range"),
                         **kw)
    fig.update_xaxes(title_text=xlabel, row=n, col=1)
    fig.update_layout(legend=dict(orientation="v", x=1.005, y=1,
                                  xanchor="left", yanchor="top"),
                      margin=dict(r=8))
    return fig


# ------------------------------------------------------------ uv coverage --
def uv_coverage(tab, theme, highlight=None, height=420, styles=None,
                title="uv coverage", mono=False, same_symbol=False,
                lock_aspect=True, group="baseline", text_col=None, gl=True):
    """uv points and their conjugates, coloured by ``group`` (baseline by
    default; ``dataset`` in comparison mode).

    ``lock_aspect`` keeps u and v on the same scale (the default).  A locked
    aspect and an arbitrary drag rectangle cannot both be honoured -- Plotly
    grows the rectangle away from the cursor -- so while locked the plot pans
    on drag and zooms on scroll, which preserves the scaling exactly.  Unlocked,
    the axes are independent: drag zooms to the box and scrolling over one axis
    rescales only that axis.  ``text_col`` adds a per-point label to the hover
    (e.g. the baseline, when colour already encodes the dataset).
    """
    if tab is None or len(tab) == 0:
        return _empty(theme)
    ink = INK[theme]
    names = sorted(tab[group].unique())
    styles = styles or style_map(names)
    fig = go.Figure()
    for i, name in enumerate(names):
        g = tab[tab[group] == name]
        st = styles.get(name, {"ci": i % 8, "symbol": "circle"})
        dim = highlight is not None and name not in highlight
        color = ink["muted"] if dim else series_color(theme, st, mono)
        u = np.concatenate([g["u"].values, -g["u"].values]) / 1e9
        v = np.concatenate([g["v"].values, -g["v"].values]) / 1e9
        cd = np.column_stack([
            np.tile(g.index.values.astype(float), 2),
            np.tile(g["time"].values, 2),
            np.tile(g["uvdist"].values / 1e9, 2),
            np.tile(g["amp"].values, 2)])
        label = f"%{{text}}<br>" if text_col is not None else ""
        fig.add_trace(_scatter(gl)(
            x=u, y=v, name=name, legendgroup=name, mode="markers",
            marker=dict(size=4 if dim else 6, color=color, symbol="circle" if same_symbol else st["symbol"],
                        opacity=0.35 if dim else 1.0,
                        line=dict(width=0)),
            customdata=cd,
            text=np.tile(g[text_col].values, 2) if text_col is not None else None,
            hovertemplate=(f"<b>{name}</b><br>{label}UT: %{{customdata[1]:.3f}} h<br>"
                           "u: %{x:.3f} Gλ<br>v: %{y:.3f} Gλ<br>"
                           "|uv|: %{customdata[2]:.3f} Gλ<br>"
                           "amp: %{customdata[3]:.4f} Jy<extra></extra>")))
    hint = " · 1:1, scroll to zoom" if lock_aspect else " · axes independent"
    apply_theme(fig, theme, height=height, title=f"{title}{hint}" if title else None)
    fig.update_layout(legend=dict(font=dict(size=10 if len(names) > 14 else 11)),
                      dragmode="pan" if lock_aspect else "zoom")
    fig.update_xaxes(title_text="u [Gλ]", autorange="reversed",
                     constrain="domain" if lock_aspect else "range")
    fig.update_yaxes(title_text="v [Gλ]",
                     scaleanchor="x" if lock_aspect else None, scaleratio=1,
                     constrain="domain" if lock_aspect else "range")
    return fig


# ------------------------------------------------------------- amp vs uv ---
def radplot(tab, theme, styles=None, height=380, errorbars=True, logy=False,
            mono=False, same_symbol=False, group="baseline", text_col=None,
            gl=True):
    return grouped_scatter(
        tab.assign(uvG=tab["uvdist"] / 1e9), "uvG", "amp", "sigma", group,
        theme, xlabel="uv distance [Gλ]", ylabel="amplitude [Jy]",
        hover=[("UT [h]", "time", ".4f"), ("amp [Jy]", "amp", ".4f"),
               ("σ [Jy]", "sigma", ".4f"), ("S/N", "snr", ".1f"),
               ("|uv| [Gλ]", "uvG", ".3f")],
        errorbars=errorbars, height=height, logy=logy, styles=styles, mono=mono,
        same_symbol=same_symbol, text_col=text_col, gl=gl,
        title="Amplitude vs uv distance")


def vis_hover(extra=()):
    return [("UT [h]", "time", ".4f"), ("amp [Jy]", "amp", ".4f"),
            ("σ_amp [Jy]", "sigma", ".4f"), ("phase [°]", "phase", ".2f"),
            ("σ_ph [°]", "sigma_phase", ".2f"), ("S/N", "snr", ".1f"),
            ("|uv| [Gλ]", "uvG", ".3f")] + list(extra)


# ----------------------------------------------------------------- export --
def apply_view(fig, relayout):
    """Re-apply the axis ranges the user zoomed to, for a static export."""
    if not relayout:
        return fig
    ranges = {}
    for key, val in relayout.items():
        if ".range[" in key:
            ax, idx = key.split(".range[")
            ranges.setdefault(ax, [None, None])[int(idx[0])] = val
        elif key.endswith(".range") and isinstance(val, (list, tuple)):
            ranges[key[: -len(".range")]] = list(val)
        elif key.endswith(".autorange") and val:
            ranges[key[: -len(".autorange")]] = None
    for ax, rng in ranges.items():
        if ax not in fig.layout:
            continue
        if rng is None:
            fig.layout[ax].autorange = True
        elif None not in rng:
            fig.layout[ax].range = rng
            fig.layout[ax].autorange = False
    return fig

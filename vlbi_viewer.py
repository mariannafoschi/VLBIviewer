"""Interactive uvfits visibility inspector.

    python vlbi_viewer.py [file.uvfits] [--port 8050]

Opens a browser dashboard: visibility amplitudes/phases and their errors per
baseline, uv coverage, and closure phases / closure amplitudes over all
non-redundant triangles and quadrangles.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import socket
import sys
import threading
import webbrowser
from functools import lru_cache

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate


def _enable_static_export():
    """Let kaleido's headless Chrome find libgbm.

    Chrome (used for PNG/PDF/SVG export) needs ``libgbm.so.1``; in a conda env
    it lives in ``$PREFIX/lib`` but is not on the loader path by default, so the
    browser dies immediately and the export silently produces nothing.  Adding
    it here means the whole feature works with a plain ``python vlbi_viewer.py`` -- no
    ``LD_LIBRARY_PATH`` incantation required.
    """
    libdir = os.path.join(sys.prefix, "lib")
    if os.path.exists(os.path.join(libdir, "libgbm.so.1")):
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if libdir not in cur.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = (libdir + os.pathsep + cur).rstrip(os.pathsep)


_enable_static_export()

import figures as fg
import uvdata
from uvdata import hours_to_hms

# --------------------------------------------------------------------------
# Data (loaded once, then cached per control combination)
# --------------------------------------------------------------------------
# One or more loaded files.  DATASETS holds them all; UV is an alias for the
# first (the "primary"), kept so every single-file code path reads exactly as
# before.  With two or more files the viewer enters comparison mode.
DATASETS: list[uvdata.UVData] = []
UV: uvdata.UVData | None = None


def set_datasets(datasets):
    """Replace the loaded set and reset the per-control caches."""
    global DATASETS, UV
    DATASETS = list(datasets)
    UV = DATASETS[0] if DATASETS else None
    _clear_caches()
    print(f"[datasets] now {len(DATASETS)}: {[ds.filename for ds in DATASETS]}",
          flush=True)


def comparing() -> bool:
    return len(DATASETS) > 1


def dataset_labels() -> list[str]:
    """Display label per dataset, in DATASETS order, deduped on collision.

    These strings are the ``dataset`` group values and the legend/header keys,
    so they must be stable for a given DATASETS list.
    """
    seen, out = {}, []
    for ds in DATASETS:
        base = ds.filename or "dataset"
        seen[base] = seen.get(base, 0) + 1
        out.append(f"{base} ({seen[base]})" if seen[base] > 1 else base)
    # a later duplicate renamed the *second* occurrence; leave the first as-is
    return out


# ---- union of controls across all datasets (for comparison mode) ----------
def all_baselines() -> list[str]:
    return sorted({b for ds in DATASETS for b in ds.baselines})


def all_sites() -> list[str]:
    out = []
    for ds in DATASETS:
        for s in ds.sites:
            if s not in out:
                out.append(s)
    return out


def all_pols() -> list[str]:
    return [p for p in uvdata.POL_CHOICES
            if any(p in ds.available_pols for ds in DATASETS)]


def time_span():
    t0 = min(ds.summary()["t_start"] for ds in DATASETS)
    t1 = max(ds.summary()["t_stop"] for ds in DATASETS)
    return t0, t1


AVG_OPTIONS = [("No averaging (native)", "none"), ("10 s", "10"), ("30 s", "30"),
               ("60 s", "60"), ("120 s", "120"), ("300 s", "300"),
               ("Whole scan", "scan")]

# every plot that can be exported, in menu order
GRAPHS = [("Overview · uv coverage", "ov-uv"),
          ("Overview · amplitude vs uv distance", "ov-rad"),
          ("Overview · amplitude & phase vs time", "ov-vis"),
          ("Overview · closure phase", "ov-cp"),
          ("Overview · log closure amplitude", "ov-ca"),
          ("Visibilities · main panel", "vis-main"),
          ("Visibilities · uv coverage", "vis-uv"),
          ("Closure phases", "cp-main"),
          ("Closure amplitudes", "ca-main")]
GRAPH_IDS = [g for _, g in GRAPHS]

PHASE_WRAP = fg.PHASE_RANGE


def hhmm(hours):
    """Hours -> 'HH:MM', NOT wrapped at 24.

    A VLBI track can cross midnight, and the reader keeps its time axis
    continuous (so ``time`` may exceed 24 h rather than jumping back to 0).  The
    UT-range boxes must round-trip such values faithfully -- wrapping 31.7 h to
    '07:44' used to collapse the default window and silently drop every dataset
    observing after 24 h, which is why comparing tracks at different epochs
    showed only some of them.
    """
    h = float(hours)
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


def hhmm_floor(hours):
    return hhmm(int(float(hours)) + int((float(hours) % 1) * 60) / 60.0)


def hhmm_ceil(hours):
    """Round up to the next whole minute, so the default box covers all data."""
    return hhmm(float(hours) + (1.0 - (float(hours) * 60) % 1) / 60.0)


def parse_hhmm(text, default):
    """'HH:MM', 'HH:MM:SS' or a bare number of hours -> hours.

    Returns ``default`` for an empty box and ``None`` for something unparseable,
    so the caller can tell "reset me" from "you typed nonsense".
    """
    if text is None or str(text).strip() == "":
        return default
    parts = str(text).strip().split(":")
    if len(parts) > 3:
        return None
    try:
        vals = [float(p) if p.strip() else 0.0 for p in parts]
    except ValueError:
        return None
    while len(vals) < 3:
        vals.append(0.0)
    h, m, sec = vals
    if not (0 <= m < 60 and 0 <= sec < 60):
        return None
    return h + m / 60.0 + sec / 3600.0


def keyed_cache(fn):
    """Memoise ``fn`` on its positional args, computing each key at most once
    even under concurrent callers.

    ``lru_cache`` locks only around the dict, so with a threaded server several
    callbacks that all miss the same key at once each run the (expensive)
    body -- e.g. the overview and both closure tabs recomputing the same
    per-dataset closures on load.  A per-key lock makes the first caller compute
    and the rest wait and reuse the result.
    """
    cache, locks, glock = {}, {}, threading.Lock()

    def wrapper(*args):
        try:
            return cache[args]
        except KeyError:
            pass
        with glock:
            lock = locks.setdefault(args, threading.Lock())
        with lock:
            if args not in cache:
                cache[args] = fn(*args)
            return cache[args]

    def cache_clear():
        with glock:
            cache.clear()
            locks.clear()

    wrapper.cache_clear = cache_clear
    return wrapper


# Every cached builder takes the dataset index ``di`` as its first key and
# reads ``DATASETS[di]``, so the same control combination is cached separately
# per file.  Single-file callers simply pass di=0.
@keyed_cache
def get_table(di, pol, avg, snr_cut, debias, ut0, ut1):
    tab = DATASETS[di].table(pol=pol, avg=avg, snr_cut=snr_cut, debias=debias)
    if ut0 is not None:
        tab = tab[(tab["time"] >= ut0) & (tab["time"] <= ut1)]
    return tab.reset_index(drop=True)


def _drop_sites(tab, exclude):
    if not exclude:
        return tab
    keep = ~tab["t1"].isin(exclude) & ~tab["t2"].isin(exclude)
    return tab[keep].reset_index(drop=True)


@keyed_cache
def get_cphase(di, pol, avg, snr_cut, debias, ut0, ut1,
               count, ref, uv_min, leg_snr, exclude):
    tab = _drop_sites(get_table(di, pol, avg, snr_cut, debias, ut0, ut1), exclude)
    return uvdata.closure_phases(tab, DATASETS[di].site_order(), count=count,
                                 ref=ref or None, uv_min=uv_min * 1e9,
                                 snr_cut=leg_snr)


@keyed_cache
def get_camp(di, pol, avg, snr_cut, debias, ut0, ut1,
             count, ref, uv_min, leg_snr, exclude, log):
    tab = _drop_sites(get_table(di, pol, avg, snr_cut, debias, ut0, ut1), exclude)
    return uvdata.closure_amplitudes(tab, DATASETS[di].site_order(), count=count,
                                     ref=ref or None, uv_min=uv_min * 1e9,
                                     snr_cut=leg_snr, debias=True, log=log)


def styles_for(names):
    return fg.style_map(names)


def _ref_for(di, ref):
    """The reference station to use for dataset ``di``: the chosen one if it
    observes in that file, otherwise None so closure_* falls back to the file's
    own first station (keeps a valid minimal set when arrays differ)."""
    return ref if ref in DATASETS[di].sites else None


def combined(build):
    """Concatenate a per-dataset frame across all DATASETS, tagged with a
    ``dataset`` label column and reindexed 0..N-1.

    ``build(di)`` returns the DataFrame for dataset ``di``.  The fresh index is
    what rides along as ``customdata[0]``, so a click resolves back with
    ``.loc[i]`` on an identically rebuilt frame -- exactly the single-file
    contract, extended across files.
    """
    frames = []
    for di, lab in enumerate(dataset_labels()):
        d = build(di)
        if d is None or len(d) == 0:
            continue
        d = d.copy()
        d["dataset"] = lab
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


# ---- frame builders that transparently combine datasets when comparing ----
# Each returns the primary file's frame in single mode and the concatenated,
# dataset-tagged frame in comparison mode.  Figure, click and table callbacks
# all go through these so the row index behind a click is always resolvable.
def vis_frame(p):
    if comparing():
        return combined(lambda di: get_table(di, *p))
    return get_table(0, *p)


def cphase_frame(p, count, ref, uvmin, legsnr, excl):
    if comparing():
        return combined(lambda di: get_cphase(di, *p, count, _ref_for(di, ref),
                                               uvmin, legsnr, excl))
    return get_cphase(0, *p, count, ref, uvmin, legsnr, excl)


def camp_frame(p, count, ref, uvmin, legsnr, excl, log):
    if comparing():
        return combined(lambda di: get_camp(di, *p, count, _ref_for(di, ref),
                                            uvmin, legsnr, excl, log))
    return get_camp(0, *p, count, ref, uvmin, legsnr, excl, log)


# --------------------------------------------------------------------------
# Loading a file (from the command line or from a browser upload)
# --------------------------------------------------------------------------
def _clear_caches():
    """Drop everything keyed on the *previous* file's data.

    The ``lru_cache`` wrappers key only on the control values (pol, avg, ...),
    not on the file, and read ``UV`` from the module global -- so after a new
    upload a repeated control combination would otherwise return the old
    file's cached table.  Clear them whenever ``UV`` is replaced.
    """
    get_table.cache_clear()
    get_cphase.cache_clear()
    get_camp.cache_clear()


def _load_one(contents: str, filename: str | None) -> uvdata.UVData:
    """Decode one ``dcc.Upload`` payload (``"data:...;base64,<payload>"``) into
    a UVData.  Decoded in memory -- astropy reads a file-like object as happily
    as a path -- so no temp file is written."""
    _, b64 = contents.split(",", 1)
    raw = base64.b64decode(b64)
    return uvdata.UVData.load(io.BytesIO(raw), name=filename or "uploaded.uvfits")


def load_from_upload(contents, filename):
    """Load the uploaded file(s) and *add* them to :data:`DATASETS`.

    ``dcc.Upload(multiple=True)`` may hand back a list of payloads (several
    files at once) or bare strings (one file) -- and, depending on the browser,
    a multi-file drop can arrive either as one list or as several separate
    callback fires.  Appending rather than replacing makes every one of those
    cases end up comparing the dropped files instead of the last one winning.
    Use the "start over" control to clear the set.
    """
    if isinstance(contents, str):
        contents, filename = [contents], [filename]
    new = [_load_one(c, n) for c, n in zip(contents, filename or [])]
    set_datasets(DATASETS + new)
    return DATASETS


def _error_banner(filename, err):
    return html.Div([
        html.Div("Could not read this file", className="err-title"),
        html.Div(f"{filename or 'file'} — {err}", className="err-msg"),
        html.Div("Expecting a VLBI uvfits file. Drop another one to try again.",
                 className="err-sub"),
    ], className="load-error")


def _dropzone():
    """The drag-and-drop / click-to-browse control shared by both states.

    On the landing page CSS blows it up into a full centred panel; once a file
    is loaded the same element shrinks to a slim bar above the dashboard, so a
    user can drop another uvfits to swap files without restarting the server.
    """
    return dcc.Upload(
        id="uploader", multiple=True,
        accept=".uvfits,.uvf,.fits,.UVFITS,.UVF,.FITS",
        className="uploader",
        children=html.Div([
            html.Span("⤢", className="uploader-icon"),
            html.Span("Drop a uvfits file here", className="uploader-title"),
            html.Span("to compare, click to browse and pick several — or drop "
                      "them one at a time (each is added)",
                      className="uploader-sub"),
        ], className="uploader-inner"))


def app_shell(initial_children, page_class):
    """The always-present outer frame: the dropzone (plus a reset button once
    files are loaded) and a container that holds either nothing (landing) or the
    full dashboard."""
    return html.Div([
        _dropzone(),
        html.Button("✕ start over", id="reset-btn", n_clicks=0,
                    className="reset-btn",
                    title="Clear all loaded files and start from an empty page"),
        # NB: no dcc.Loading wrapper here.  Each graph has its own spinner; a
        # wrapper around the whole container would overlay a single spinner that
        # hides the header, controls and every already-drawn plot until the
        # slowest callback (per-dataset closures) finished -- which made a
        # many-file comparison look like a blank/frozen page.
        html.Div(initial_children, id="app-container"),
    ], id="page", className=page_class)


# --------------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------------
def control(label, comp, width="170px", hint=None, aside=None):
    """A labelled control.  ``aside`` is an extra element shown to the right of
    the label on the same line (used for the UT record count)."""
    head = html.Label(label, className="ctl-label", title=hint or "")
    if aside is not None:
        head = html.Div([head, aside], className="ctl-head")
    return html.Div([head, comp], className="ctl", style={"width": width})


def graph(gid, height=None):
    return dcc.Loading(
        dcc.Graph(
            id=gid, className="graph",
            config={"displaylogo": False, "scrollZoom": True,
                    "doubleClick": "reset",
                    # box- and lasso-select drove nothing, so drop them; keep
                    # zoom / pan / autoscale / reset / snapshot
                    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                    "toImageButtonOptions": {"format": "png", "scale": 2}},
            style={"height": height} if height else None),
        type="dot", color="#2a78d6", delay_show=250)


def info_card(cid):
    return html.Div(id=cid, className="info-card",
                    children=html.Div("Click any point for its full record.",
                                      className="info-empty"))


def _ds_legend():
    """Compare-mode header: one coloured chip per dataset, which doubles as the
    colour key for the overlaid plots (the swatch colour is set per theme in
    style.css so it tracks the plots)."""
    chips = []
    for i, (ds, lab) in enumerate(zip(DATASETS, dataset_labels())):
        s = ds.summary()
        chips.append(html.Div([
            html.Span(className=f"ds-sw ds-sw-{i % 8}"),
            html.Div([
                html.B(lab, className="ds-name"),
                html.Span(f"{s['source']} · MJD {s['mjd']} · "
                          f"{s['n_baselines']} baselines", className="ds-meta"),
            ], className="ds-text"),
        ], className="ds-chip"))
    return html.Div(chips, className="ds-legend")


def build_layout(uv: uvdata.UVData, filename: str):
    s = uv.summary()
    cmp = comparing()
    bls = all_baselines() if cmp else uv.baselines
    pols = all_pols() if cmp else uv.available_pols
    sites = all_sites() if cmp else list(uv.sites)
    t0, t1 = time_span() if cmp else (s["t_start"], s["t_stop"])

    stats = html.Div([
        html.Div([html.Span("SOURCE"), html.B(f"{s['source']}")],
                 className="stat"),
        html.Div([html.Span("MJD"), html.B(f"{s['mjd']}")], className="stat"),
        html.Div([html.Span("FREQ"), html.B(f"{s['freq_GHz']:.2f} GHz")],
                 className="stat"),
        html.Div([html.Span("BW"), html.B(f"{s['bw_GHz']:.3f} GHz")], className="stat"),
        html.Div([html.Span("sites"), html.B(f"{s['n_sites']}")], className="stat"),
        html.Div([html.Span("baselines"), html.B(f"{s['n_baselines']}")], className="stat"),
        html.Div([html.Span("visibilities"), html.B(f"{s['n_vis']}")], className="stat"),
        html.Div([html.Span("UT range"),
                  html.B(f"{hours_to_hms(t0)[:5]}–{hours_to_hms(t1)[:5]}")],
                 className="stat"),
    ], className="stats")

    subtitle = (f"comparing {len(DATASETS)} datasets" if cmp
                else os.path.basename(filename))
    header = html.Div([
        html.Div([
            html.H1("VLBI Viewer", className="title"),
            html.Div(subtitle, className="subtitle"),
        ]),
        _ds_legend() if cmp else stats,
        control("Theme", dcc.Dropdown(searchable=False,
            id="theme-sel", clearable=False, value="light",
            options=[{"label": lbl, "value": v} for lbl, v in fg.THEMES]),
            width="118px",
            hint="Light and Warm sit on a pale surface, Dark and Teal on a "
                 "dark one; the series colours are stepped for each"),
    ], className="header")

    controls = html.Div([
        control("Polarization", dcc.Dropdown(searchable=False, 
            id="pol", options=[{"label": p, "value": p} for p in pols],
            value="I" if "I" in pols else pols[0], clearable=False),
            hint="Stokes I = (RR+LL)/2; single-hand data used as I when only one exists"),
        control("Time averaging", dcc.Dropdown(searchable=False, 
            id="avg", options=[{"label": a, "value": b} for a, b in AVG_OPTIONS],
            value="none", clearable=False),
            hint="Coherent (complex, inverse-variance weighted) averaging per baseline"),
        control("S/N cut", dcc.Input(id="snr", type="number", value=0, min=0,
                                     step=0.5, debounce=True), width="80px",
                hint="Drop visibilities below this amplitude S/N"),
        control("UT range  HH:MM", html.Div([
            dcc.Input(id="ut-start", type="text", debounce=True,
                      value=hhmm_floor(t0), placeholder="HH:MM"),
            html.Span("–", className="dash"),
            dcc.Input(id="ut-end", type="text", debounce=True,
                      value=hhmm_ceil(t1), placeholder="HH:MM"),
        ], className="ut-row"), width="196px",
            aside=html.Span(id="ut-label", className="ctl-count"),
            hint="Start and end of the window, as HH:MM (HH:MM:SS also works). "
                 "Every tab -- visibilities, closures and table -- is restricted "
                 "to it; clear a box to go back to the full track"),
        control("Options", dcc.Checklist(
            id="opts", options=[{"label": " error bars", "value": "err"}],
            value=["err"], className="checks"), width="110px",
            hint="Show 1σ error bars on every quantity"),
        control("Style", dcc.Checklist(
            id="style", options=[{"label": " one colour", "value": "mono"},
                                 {"label": " circle markers", "value": "circle"},
                                 {"label": " uv 1:1", "value": "lock"}],
            value=["lock"], className="checks"), width="140px",
            hint="One colour: draw every series in the same colour.  "
                 "Circle markers: use a circle for every baseline instead of "
                 "per-baseline symbols.  uv 1:1: keep u and v on the same scale "
                 "(drag pans, scroll zooms); unlock to zoom the axes independently"),
        control("Export current plot", html.Div([
            dcc.Dropdown(searchable=False, id="export-target", clearable=False, value="vis-main",
                         options=[{"label": lbl, "value": gid}
                                  for lbl, gid in GRAPHS]),
            html.Div([
                dcc.Dropdown(searchable=False, id="export-fmt", clearable=False, value="png",
                             options=[{"label": f.upper(), "value": f}
                                      for f in ("png", "pdf", "svg", "html")],
                             className="fmt"),
                html.Button("⤓ save", id="export-btn", className="btn"),
                dcc.Download(id="export-dl"),
            ], className="export-row"),
            html.Div(id="export-msg", className="hint"),
        ]), width="min(300px, 24vw)",
            hint="Saves exactly what is on screen: current zoom and every "
                 "control setting.  PNG/PDF/SVG are static, HTML stays "
                 "interactive"),
    ], className="controls")

    # ---- tab: overview ---------------------------------------------------
    tab_overview = html.Div([
        html.Div([graph("ov-uv"), graph("ov-rad")], className="row2"),
        graph("ov-vis"),
        html.Div([graph("ov-cp"), graph("ov-ca")], className="row2"),
    ], className="tab-body")

    # ---- tab: visibilities ----------------------------------------------
    tab_vis = html.Div([
        html.Div([
            control("Baselines", dcc.Dropdown(searchable=False, 
                id="bl-sel", options=[{"label": b, "value": b} for b in bls],
                value=bls, multi=True, placeholder="all baselines"),
                width="min(560px, 45vw)"),
            html.Button("all", id="bl-all", className="btn"),
            html.Button("none", id="bl-none", className="btn"),
            control("x axis", dcc.Dropdown(searchable=False, 
                id="vis-x", clearable=False, value="time",
                options=[{"label": "UT time", "value": "time"},
                         {"label": "uv distance", "value": "uvG"}])),
            control("Panels", dcc.Dropdown(searchable=False, 
                id="vis-y", multi=True, value=["amp", "phase"],
                options=[{"label": l, "value": v} for l, v in [
                    ("amplitude", "amp"), ("phase", "phase"),
                    ("σ amplitude", "sigma"), ("σ phase", "sigma_phase"),
                    ("S/N", "snr"), ("real", "re"), ("imag", "im")]]),
                width="260px"),
            control("Layout", dcc.Dropdown(searchable=False, 
                id="vis-facet", clearable=False, value="overlay",
                options=[{"label": "overlay baselines", "value": "overlay"},
                         {"label": "one panel per baseline", "value": "facet"}]),
                width="200px"),
            control("Log amp", dcc.Checklist(
                id="vis-log", options=[{"label": " log y", "value": "log"}],
                value=[], className="checks"), width="90px"),
        ], className="controls sub"),
        graph("vis-main"),
        info_card("vis-info"),
        html.Details([
            html.Summary("uv coverage of the selected baselines"),
            graph("vis-uv"),
        ], open=False, className="details"),
    ], className="tab-body")

    # ---- closure controls (shared shape) ---------------------------------
    def closure_controls(kind, thing):
        return html.Div([
            control("Set", dcc.Dropdown(searchable=False, 
                id=f"{kind}-set", clearable=False, value="min",
                options=[{"label": "non-redundant (minimal)", "value": "min"},
                         {"label": "all combinations", "value": "max"}]),
                width="215px",
                hint="Minimal: every triangle/quadrangle shares one reference "
                     "station, giving an independent set"),
            control("Reference site", dcc.Dropdown(searchable=False, 
                id=f"{kind}-ref", value=sites[0], clearable=False,
                options=[{"label": t, "value": t} for t in sites]), width="125px"),
            control("Antennas — untick to drop", dcc.Checklist(
                id=f"{kind}-ants", value=sites, inline=True, className="ants",
                options=[{"label": t, "value": t} for t in sites]),
                width="260px",
                hint=f"Unticking a station drops it and rebuilds the {thing} "
                     "from the baselines that are left"),
            control("Drop baselines <", dcc.Input(
                id=f"{kind}-uvmin", type="number", value=0.1, min=0, step=0.05,
                debounce=True), width="115px",
                hint="In Gλ. Removes trivial intra-site triangles/quadrangles"),
            control("Leg S/N >", dcc.Input(
                id=f"{kind}-legsnr", type="number", value=0, min=0, step=0.5,
                debounce=True), width="95px"),
            control("Layout", dcc.Dropdown(searchable=False, 
                id=f"{kind}-facet", clearable=False, value="overlay",
                options=[{"label": "overlay", "value": "overlay"},
                         {"label": "one panel each", "value": "facet"}]),
                width="160px"),
        ], className="controls sub")

    tab_cp = html.Div([
        closure_controls("cp", "triangles"),
        html.Div([
            control("Triangles", dcc.Dropdown(searchable=False, id="cp-sel", multi=True,
                                              placeholder="all triangles"),
                    width="min(700px, 60vw)"),
            html.Button("all", id="cp-all", className="btn"),
            html.Button("none", id="cp-none", className="btn"),
            html.Div(id="cp-count", className="hint"),
        ], className="controls sub"),
        graph("cp-main"),
        info_card("cp-info"),
    ], className="tab-body")

    tab_ca = html.Div([
        closure_controls("ca", "quadrangles"),
        html.Div([
            control("Quadrangles", dcc.Dropdown(searchable=False, id="ca-sel", multi=True,
                                                placeholder="all quadrangles"),
                    width="min(700px, 60vw)"),
            html.Button("all", id="ca-all", className="btn"),
            html.Button("none", id="ca-none", className="btn"),
            control("Quantity", dcc.Dropdown(searchable=False, 
                id="ca-log", clearable=False, value="log",
                options=[{"label": "log closure amplitude", "value": "log"},
                         {"label": "closure amplitude", "value": "lin"}]),
                width="230px"),
            control("Y range", dcc.Checklist(
                id="ca-clip", options=[{"label": " clip outliers", "value": "clip"}],
                value=["clip"], className="checks"), width="130px",
                hint="Show the 1–99 percentile range; double-click the plot to "
                     "autoscale over everything"),
            html.Div(id="ca-count", className="hint"),
        ], className="controls sub"),
        graph("ca-main"),
        info_card("ca-info"),
    ], className="tab-body")

    # ---- tab: table ------------------------------------------------------
    tab_table = html.Div([
        html.Div([
            control("Dataset", dcc.Dropdown(searchable=False, 
                id="tbl-kind", clearable=False, value="vis",
                options=[{"label": "visibilities", "value": "vis"},
                         {"label": "closure phases", "value": "cp"},
                         {"label": "closure amplitudes", "value": "ca"}]),
                width="220px"),
            html.Button("⤓ download CSV", id="tbl-dl", className="btn"),
            dcc.Download(id="tbl-download"),
            html.Div(id="tbl-count", className="hint"),
        ], className="controls sub"),
        html.Div(id="tbl-wrap", className="table-wrap"),
    ], className="tab-body")

    return html.Div([
        dcc.Store(id="theme", data="light"),
        dcc.Store(id="ut", data=[t0, t1]),
        header,
        controls,
        dcc.Tabs(id="tabs", value="overview", className="tabs", children=[
            dcc.Tab(label="Overview", value="overview", children=tab_overview,
                    className="tab", selected_className="tab-sel"),
            dcc.Tab(label="Visibilities", value="vis", children=tab_vis,
                    className="tab", selected_className="tab-sel"),
            dcc.Tab(label="Closure phases", value="cp", children=tab_cp,
                    className="tab", selected_className="tab-sel"),
            dcc.Tab(label="Closure amplitudes", value="ca", children=tab_ca,
                    className="tab", selected_className="tab-sel"),
            dcc.Tab(label="Table", value="table", children=tab_table,
                    className="tab", selected_className="tab-sel"),
        ]),
        html.Div("drag to zoom · double-click to reset · click a point for details "
                 "· click a legend entry to isolate a series",
                 className="footer"),
    ], id="root", className="root theme-light")


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------
def register(app: Dash):

    @app.callback(Output("app-container", "children"),
                  Output("page", "className"),
                  Input("uploader", "contents"),
                  Input("reset-btn", "n_clicks"),
                  State("uploader", "filename"),
                  prevent_initial_call=True)
    def _on_upload(contents, reset, filename):
        if ctx.triggered_id == "reset-btn":
            set_datasets([])
            return None, "landing"
        if not contents:
            raise PreventUpdate
        names = filename if isinstance(filename, list) else [filename]
        n_recv = len(contents) if isinstance(contents, list) else 1
        print(f"[upload] browser sent {n_recv} file(s): {names}", flush=True)
        try:
            load_from_upload(contents, filename)
        except Exception as exc:                       # any malformed input
            return _error_banner(", ".join(n or "file" for n in names), exc), "landing"
        return build_layout(UV, UV.filename), "loaded"

    @app.callback(Output("theme", "data"), Output("root", "className"),
                  Input("theme-sel", "value"))
    def _set_theme(theme):
        return theme, f"root theme-{theme}"

    @app.callback(Output("ut", "data"), Output("ut-label", "children"),
                  Input("ut-start", "value"), Input("ut-end", "value"))
    def _ut_window(start, end):
        t0, t1 = (time_span() if comparing()
                  else (UV.summary()["t_start"], UV.summary()["t_stop"]))
        a, b = parse_hhmm(start, t0), parse_hhmm(end, t1)
        if a is None or b is None:
            return [t0, t1], "unreadable time — use HH:MM (full track shown)"
        if a > b:
            a, b = b, a
        a, b = max(a, t0), min(b, t1)
        if b <= a:
            return [t0, t1], "empty window — full track shown"
        n = sum(int(((ds.df["time"] >= a) & (ds.df["time"] <= b)).sum())
                for ds in DATASETS)
        return [a, b], f"{n} records"

    # ---- shared parameter bundle ----------------------------------------
    def params(pol, avg, snr, opts, ut):
        ut = ut or [None, None]
        return (pol, avg, float(snr or 0), False,        # debias always off
                round(float(ut[0]), 6), round(float(ut[1]), 6))

    def excluded(ants):
        """Stations that are unticked -> the exclusion tuple the cache expects."""
        universe = all_sites() if comparing() else list(UV.sites)
        return tuple(sorted(set(universe) - set(ants if ants is not None
                                                else universe)))

    def style_flags(style):
        style = style or []
        return "mono" in style, "circle" in style, "lock" in style

    # ---- overview --------------------------------------------------------
    # Split into a fast (visibility) callback and a separate closures callback:
    # closures cost ~seconds *per dataset*, so folding them into one callback
    # made the whole Overview block until every file's closures were done
    # (looking frozen for 3+ files).  Now the uv / amplitude / phase panels
    # appear immediately and the two closure panels fill in on their own.
    @app.callback(
        Output("ov-uv", "figure"), Output("ov-rad", "figure"),
        Output("ov-vis", "figure"),
        Input("pol", "value"), Input("avg", "value"), Input("snr", "value"),
        Input("opts", "value"), Input("ut", "data"), Input("style", "value"),
        Input("theme", "data"))
    def _overview_vis(pol, avg, snr, opts, ut, style, theme):
        p = params(pol, avg, snr, opts, ut)
        err = "err" in (opts or [])
        mono, circ, lock = style_flags(style)
        cmp = comparing()
        # In compare mode colour encodes the dataset (one colour+marker each),
        # baseline rides on hover, and mono/circle toggles are ignored so the
        # datasets stay distinguishable.
        if cmp:
            grp, tcol = "dataset", "baseline"
            st = fg.dataset_style_map(dataset_labels())
            mono = circ = False
            tab = combined(lambda di: get_table(di, *p))
        else:
            grp, tcol = "baseline", None
            st = styles_for(UV.baselines)
            tab = get_table(0, *p)
        tab = tab.assign(uvG=tab["uvdist"] / 1e9)

        f_uv = fg.uv_coverage(tab, theme, styles=st, height=380, mono=mono,
                              same_symbol=circ, lock_aspect=lock, group=grp,
                              text_col=tcol)
        f_rad = fg.radplot(tab, theme, styles=st, height=380, errorbars=err,
                           mono=mono, same_symbol=circ, group=grp, text_col=tcol)
        f_vis = fg.multi_panel(
            tab, "time",
            [{"y": "amp", "err": "sigma", "label": "amplitude [Jy]"},
             {"y": "phase", "err": "sigma_phase", "label": "phase [°]",
              "wrap": PHASE_WRAP, "hline": 0}],
            grp, theme, xlabel="UT [hours]", hover=fg.vis_hover(),
            styles=st, errorbars=err, height=520, mono=mono, same_symbol=circ,
            text_col=tcol,
            title=f"Stokes {pol} — all baselines")
        return f_uv, f_rad, f_vis

    @app.callback(
        Output("ov-cp", "figure"), Output("ov-ca", "figure"),
        Input("pol", "value"), Input("avg", "value"), Input("snr", "value"),
        Input("opts", "value"), Input("ut", "data"), Input("style", "value"),
        Input("theme", "data"))
    def _overview_closures(pol, avg, snr, opts, ut, style, theme):
        p = params(pol, avg, snr, opts, ut)
        err = "err" in (opts or [])
        mono, circ, _ = style_flags(style)
        cmp = comparing()
        if cmp:
            st = fg.dataset_style_map(dataset_labels())
            mono = circ = False
            ref0 = all_sites()[0]
            cp = combined(lambda di: get_cphase(di, *p, "min", _ref_for(di, ref0),
                                                0.1, 0.0, ()))
            ca = combined(lambda di: get_camp(di, *p, "min", _ref_for(di, ref0),
                                              0.1, 0.0, (), True))
        else:
            st = None
            cp = get_cphase(0, *p, "min", UV.sites[0], 0.1, 0.0, ())
            ca = get_camp(0, *p, "min", UV.sites[0], 0.1, 0.0, (), True)

        # closures group by triangle/quadrangle in single mode, by dataset when
        # comparing (their frames have no 'baseline' column)
        f_cp = fg.grouped_scatter(
            cp, "time", "cphase", "sigma_cp", "dataset" if cmp else "triangle",
            theme, xlabel="UT [hours]", ylabel="closure phase [°]",
            hover=[("UT [h]", "time", ".4f"), ("Ψ [°]", "cphase", ".2f"),
                   ("σ [°]", "sigma_cp", ".2f"), ("S/N", "snr", ".1f")],
            errorbars=err, height=380, wrap=PHASE_WRAP, hline=0, mono=mono,
            same_symbol=circ, styles=st, text_col="triangle" if cmp else None,
            title="Closure phase — non-redundant set")
        f_ca = fg.grouped_scatter(
            ca, "time", "camp", "sigma_camp", "dataset" if cmp else "quadrangle",
            theme, xlabel="UT [hours]", ylabel="log closure amplitude",
            hover=[("UT [h]", "time", ".4f"), ("log C", "camp", ".3f"),
                   ("σ", "sigma_camp", ".3f"), ("S/N", "snr", ".1f")],
            errorbars=err, height=380, hline=0, mono=mono, same_symbol=circ,
            styles=st, text_col="quadrangle" if cmp else None,
            yrange=fg.robust_range(ca["camp"]) if ca is not None else None,
            title="Log closure amplitude — non-redundant set")
        for f in (f_cp, f_ca):
            f.update_layout(showlegend=cmp)
        return f_cp, f_ca

    # ---- visibilities ----------------------------------------------------
    @app.callback(Output("bl-sel", "value"),
                  Input("bl-all", "n_clicks"), Input("bl-none", "n_clicks"),
                  prevent_initial_call=True)
    def _bl_buttons(a, b):
        allbl = all_baselines() if comparing() else UV.baselines
        return allbl if ctx.triggered_id == "bl-all" else []

    @app.callback(
        Output("vis-main", "figure"), Output("vis-uv", "figure"),
        Input("pol", "value"), Input("avg", "value"), Input("snr", "value"),
        Input("opts", "value"), Input("ut", "data"), Input("style", "value"),
        Input("bl-sel", "value"), Input("vis-x", "value"),
        Input("vis-y", "value"), Input("vis-facet", "value"),
        Input("vis-log", "value"), Input("theme", "data"))
    def _vis(pol, avg, snr, opts, ut, style, bls, xcol, ycols, facet, logy, theme):
        p = params(pol, avg, snr, opts, ut)
        mono, circ, lock = style_flags(style)
        err = "err" in (opts or [])
        cmp = comparing()
        if cmp:
            grp, tcol = "dataset", "baseline"
            st = fg.dataset_style_map(dataset_labels())
            mono = circ = False
            tab = combined(lambda di: get_table(di, *p))
        else:
            grp, tcol = "baseline", None
            st = styles_for(UV.baselines)
            tab = get_table(0, *p)
        tab = tab.assign(uvG=lambda d: d["uvdist"] / 1e9)
        sel = tab[tab["baseline"].isin(bls or [])]
        xlabel = "UT [hours]" if xcol == "time" else "uv distance [Gλ]"

        labels = {"amp": "amplitude [Jy]", "phase": "phase [°]",
                  "sigma": "σ amplitude [Jy]", "sigma_phase": "σ phase [°]",
                  "snr": "amplitude S/N", "re": "real [Jy]", "im": "imag [Jy]"}
        errs = {"amp": "sigma", "phase": "sigma_phase", "re": "sigma",
                "im": "sigma"}
        panels = []
        for y in (ycols or ["amp"]):
            panels.append({
                "y": y, "err": errs.get(y), "label": labels.get(y, y),
                "logy": ("log" in (logy or []) and y in ("amp", "sigma", "snr")),
                "wrap": PHASE_WRAP if y == "phase" else None,
                "hline": 0 if y in ("phase", "re", "im") else None})

        if facet == "facet":
            y0 = panels[0]
            f = fg.grouped_scatter(
                sel, xcol, y0["y"], y0["err"], grp, theme,
                xlabel=xlabel, ylabel=y0["label"], hover=fg.vis_hover(),
                errorbars=err, facet=True, styles=st, logy=y0["logy"],
                wrap=y0["wrap"], hline=y0["hline"], mono=mono, same_symbol=circ,
                text_col=tcol,
                title=f"Stokes {pol} — {y0['label']} per {grp}")
        else:
            f = fg.multi_panel(sel, xcol, panels, grp, theme,
                               xlabel=xlabel, hover=fg.vis_hover(), styles=st,
                               errorbars=err, mono=mono, same_symbol=circ,
                               text_col=tcol,
                               title=f"Stokes {pol} — {len(bls or [])} baselines")
        # single: all baselines with the selection highlighted; compare: the
        # selected baselines coloured by dataset (highlight is per-baseline)
        f_uv = fg.uv_coverage(sel if cmp else tab, theme,
                              highlight=None if cmp else set(bls or []),
                              styles=st, height=430, mono=mono, lock_aspect=lock,
                              group=grp, text_col=tcol)
        return f, f_uv

    # ---- closure phases --------------------------------------------------
    @app.callback(
        Output("cp-sel", "options"), Output("cp-sel", "value"),
        Output("cp-count", "children"),
        Input("pol", "value"), Input("avg", "value"), Input("snr", "value"),
        Input("opts", "value"), Input("ut", "data"),
        Input("cp-set", "value"), Input("cp-ref", "value"),
        Input("cp-ants", "value"), Input("cp-uvmin", "value"),
        Input("cp-legsnr", "value"),
        Input("cp-all", "n_clicks"), Input("cp-none", "n_clicks"),
        State("cp-sel", "value"), prevent_initial_call=False)
    def _cp_options(pol, avg, snr, opts, ut, count, ref, ants, uvmin, legsnr,
                    a, b, cur):
        p = params(pol, avg, snr, opts, ut)
        excl = excluded(ants)
        cp = cphase_frame(p, count, ref, float(uvmin or 0), float(legsnr or 0),
                          excl)
        tris = sorted(cp["triangle"].unique()) if cp is not None else []
        options = [{"label": t, "value": t} for t in tris]
        msg = f"{0 if cp is None else len(cp)} closure phases on {len(tris)} triangles"
        if excl:
            msg += f", without {', '.join(sorted(excl))}"
        if ctx.triggered_id == "cp-none":
            return options, [], msg
        if ctx.triggered_id == "cp-all" or not cur:
            return options, tris, msg
        keep = [t for t in cur if t in tris]
        return options, keep or tris, msg

    @app.callback(
        Output("cp-main", "figure"),
        Input("pol", "value"), Input("avg", "value"), Input("snr", "value"),
        Input("opts", "value"), Input("ut", "data"), Input("style", "value"),
        Input("cp-set", "value"), Input("cp-ref", "value"),
        Input("cp-ants", "value"), Input("cp-uvmin", "value"),
        Input("cp-legsnr", "value"), Input("cp-sel", "value"),
        Input("cp-facet", "value"), Input("theme", "data"))
    def _cp_fig(pol, avg, snr, opts, ut, style, count, ref, ants, uvmin,
                legsnr, sel, facet, theme):
        p = params(pol, avg, snr, opts, ut)
        mono, circ, _ = style_flags(style)
        excl = excluded(ants)
        cmp = comparing()
        cp = cphase_frame(p, count, ref, float(uvmin or 0), float(legsnr or 0),
                          excl)
        sub = cp[cp["triangle"].isin(sel or [])] if cp is not None else None
        grp = "dataset" if cmp else "triangle"
        st = fg.dataset_style_map(dataset_labels()) if cmp else None
        if cmp:
            mono = circ = False
        drop = f", without {'+'.join(sorted(excl))}" if excl else ""
        return fg.grouped_scatter(
            sub, "time", "cphase", "sigma_cp", grp, theme,
            xlabel="UT [hours]", ylabel="closure phase [°]",
            hover=[("UT [h]", "time", ".4f"), ("Ψ [°]", "cphase", ".2f"),
                   ("σ [°]", "sigma_cp", ".2f"), ("S/N", "snr", ".1f"),
                   ("|uv|min [λ]", "uvmin", ".3s")],
            errorbars="err" in (opts or []), facet=(facet == "facet"),
            wrap=PHASE_WRAP, hline=0, mono=mono, same_symbol=circ,
            styles=st, text_col="triangle" if cmp else None,
            height=None if facet == "facet" else 560,
            title=f"Closure phase — Stokes {pol}, {count} set, ref {ref}{drop}")

    # ---- closure amplitudes ---------------------------------------------
    @app.callback(
        Output("ca-sel", "options"), Output("ca-sel", "value"),
        Output("ca-count", "children"),
        Input("pol", "value"), Input("avg", "value"), Input("snr", "value"),
        Input("opts", "value"), Input("ut", "data"),
        Input("ca-set", "value"), Input("ca-ref", "value"),
        Input("ca-ants", "value"), Input("ca-uvmin", "value"),
        Input("ca-legsnr", "value"), Input("ca-log", "value"),
        Input("ca-all", "n_clicks"), Input("ca-none", "n_clicks"),
        State("ca-sel", "value"))
    def _ca_options(pol, avg, snr, opts, ut, count, ref, ants, uvmin, legsnr,
                    log, a, b, cur):
        p = params(pol, avg, snr, opts, ut)
        excl = excluded(ants)
        ca = camp_frame(p, count, ref, float(uvmin or 0), float(legsnr or 0),
                        excl, log == "log")
        quads = sorted(ca["quadrangle"].unique()) if ca is not None else []
        options = [{"label": q, "value": q} for q in quads]
        msg = f"{0 if ca is None else len(ca)} closure amplitudes on {len(quads)} quadrangles"
        if excl:
            msg += f", without {', '.join(sorted(excl))}"
        if ctx.triggered_id == "ca-none":
            return options, [], msg
        if ctx.triggered_id == "ca-all" or not cur:
            return options, quads, msg
        keep = [q for q in cur if q in quads]
        return options, keep or quads, msg

    @app.callback(
        Output("ca-main", "figure"),
        Input("pol", "value"), Input("avg", "value"), Input("snr", "value"),
        Input("opts", "value"), Input("ut", "data"), Input("style", "value"),
        Input("ca-set", "value"), Input("ca-ref", "value"),
        Input("ca-ants", "value"), Input("ca-uvmin", "value"),
        Input("ca-legsnr", "value"), Input("ca-sel", "value"),
        Input("ca-facet", "value"), Input("ca-log", "value"),
        Input("ca-clip", "value"), Input("theme", "data"))
    def _ca_fig(pol, avg, snr, opts, ut, style, count, ref, ants, uvmin,
                legsnr, sel, facet, log, clip, theme):
        p = params(pol, avg, snr, opts, ut)
        mono, circ, _ = style_flags(style)
        excl = excluded(ants)
        cmp = comparing()
        ca = camp_frame(p, count, ref, float(uvmin or 0), float(legsnr or 0),
                        excl, log == "log")
        sub = ca[ca["quadrangle"].isin(sel or [])] if ca is not None else None
        ylab = "log closure amplitude" if log == "log" else "closure amplitude"
        yr = (fg.robust_range(sub["camp"])
              if (sub is not None and "clip" in (clip or [])) else None)
        if yr and log != "log":
            yr = [max(yr[0], 1e-3), yr[1]]
        grp = "dataset" if cmp else "quadrangle"
        st = fg.dataset_style_map(dataset_labels()) if cmp else None
        if cmp:
            mono = circ = False
        drop = f", without {'+'.join(sorted(excl))}" if excl else ""
        return fg.grouped_scatter(
            sub, "time", "camp", "sigma_camp", grp, theme,
            xlabel="UT [hours]", ylabel=ylab,
            hover=[("UT [h]", "time", ".4f"), ("value", "camp", ".4f"),
                   ("σ", "sigma_camp", ".4f"), ("S/N", "snr", ".1f")],
            errorbars="err" in (opts or []), facet=(facet == "facet"),
            hline=0 if log == "log" else 1, logy=(log != "log"), yrange=yr,
            mono=mono, same_symbol=circ, styles=st,
            text_col="quadrangle" if cmp else None,
            height=None if facet == "facet" else 560,
            title=f"{ylab} — Stokes {pol}, {count} set, ref {ref}{drop}")

    # ---- click -> record -------------------------------------------------
    def record_card(row, kind, pol):
        def item(k, v):
            return html.Div([html.Span(k, className="k"), html.B(v)],
                            className="kv")
        head, items = "", []
        if kind == "vis":
            head = f"{row['baseline']}   ·   UT {hours_to_hms(row['time'])}"
            items = [
                item("Stokes", pol),
                item("amplitude", f"{row['amp']:.5f} ± {row['sigma']:.5f} Jy"),
                item("phase", f"{row['phase']:.2f} ± {row['sigma_phase']:.2f} °"),
                item("S/N", f"{row['snr']:.2f}"),
                item("u, v", f"{row['u']/1e9:.4f}, {row['v']/1e9:.4f} Gλ"),
                item("|uv|", f"{row['uvdist']/1e9:.4f} Gλ  "
                             f"({1/ (row['uvdist']+1e-30) * 206264.806e6:.1f} μas)"),
                item("integration", f"{row['tint']:.1f} s"),
                item("re, im", f"{row['re']:.5f}, {row['im']:.5f} Jy"),
            ]
        elif kind == "cp":
            head = f"triangle {row['triangle']}   ·   UT {hours_to_hms(row['time'])}"
            items = [
                item("closure phase", f"{row['cphase']:.3f} ± {row['sigma_cp']:.3f} °"),
                item("legs", f"{row['t1']}–{row['t2']}, {row['t2']}–{row['t3']}, "
                             f"{row['t3']}–{row['t1']}"),
                item("S/N", f"{row['snr']:.2f}"),
                item("|uv| range", f"{row['uvmin']/1e9:.3f} – {row['uvmax']/1e9:.3f} Gλ"),
            ]
        else:
            head = f"quadrangle {row['quadrangle']}   ·   UT {hours_to_hms(row['time'])}"
            items = [
                item("value", f"{row['camp']:.4f} ± {row['sigma_camp']:.4f}"),
                item("ratio", f"(|{row['t1']}{row['t2']}|·|{row['t3']}{row['t4']}|) / "
                              f"(|{row['t1']}{row['t4']}|·|{row['t2']}{row['t3']}|)"),
                item("S/N", f"{row['snr']:.2f}"),
                item("|uv| range", f"{row['uvmin']/1e9:.3f} – {row['uvmax']/1e9:.3f} Gλ"),
            ]
        head_children = [html.Div(head, className="info-head")]
        if "dataset" in row and row["dataset"] is not None:
            head_children.insert(0, html.Div(str(row["dataset"]),
                                             className="info-dataset"))
        return html.Div(head_children + [html.Div(items, className="info-grid")])

    def click_index(clickData):
        if not clickData or not clickData.get("points"):
            return None
        cd = clickData["points"][0].get("customdata")
        return int(cd[0]) if cd else None

    @app.callback(Output("vis-info", "children"), Input("vis-main", "clickData"),
                  State("pol", "value"), State("avg", "value"),
                  State("snr", "value"), State("opts", "value"),
                  State("ut", "data"))
    def _vis_click(click, pol, avg, snr, opts, ut):
        i = click_index(click)
        if i is None:
            return no_update
        tab = vis_frame(params(pol, avg, snr, opts, ut))
        return record_card(tab.loc[i], "vis", pol)

    @app.callback(Output("cp-info", "children"), Input("cp-main", "clickData"),
                  State("pol", "value"), State("avg", "value"),
                  State("snr", "value"), State("opts", "value"),
                  State("ut", "data"), State("cp-set", "value"),
                  State("cp-ref", "value"), State("cp-ants", "value"),
                  State("cp-uvmin", "value"), State("cp-legsnr", "value"))
    def _cp_click(click, pol, avg, snr, opts, ut, count, ref, ants, uvmin, legsnr):
        i = click_index(click)
        if i is None:
            return no_update
        cp = cphase_frame(params(pol, avg, snr, opts, ut), count, ref,
                          float(uvmin or 0), float(legsnr or 0), excluded(ants))
        return record_card(cp.loc[i], "cp", pol)

    @app.callback(Output("ca-info", "children"), Input("ca-main", "clickData"),
                  State("pol", "value"), State("avg", "value"),
                  State("snr", "value"), State("opts", "value"),
                  State("ut", "data"), State("ca-set", "value"),
                  State("ca-ref", "value"), State("ca-ants", "value"),
                  State("ca-uvmin", "value"), State("ca-legsnr", "value"),
                  State("ca-log", "value"))
    def _ca_click(click, pol, avg, snr, opts, ut, count, ref, ants, uvmin,
                  legsnr, log):
        i = click_index(click)
        if i is None:
            return no_update
        ca = camp_frame(params(pol, avg, snr, opts, ut), count, ref,
                        float(uvmin or 0), float(legsnr or 0), excluded(ants),
                        log == "log")
        return record_card(ca.loc[i], "ca", pol)

    # ---- export ----------------------------------------------------------
    export_states = []
    for gid in GRAPH_IDS:
        export_states += [State(gid, "figure"), State(gid, "relayoutData")]

    @app.callback(Output("export-dl", "data"), Output("export-msg", "children"),
                  Input("export-btn", "n_clicks"),
                  State("export-target", "value"), State("export-fmt", "value"),
                  *export_states, prevent_initial_call=True)
    def _export(n, target, fmt, *vals):
        figdict, relayout = dict(zip(GRAPH_IDS,
                                     zip(vals[::2], vals[1::2])))[target]
        if not figdict or not figdict.get("data"):
            return no_update, "that plot has not been drawn yet"
        fig = fg.apply_view(go.Figure(figdict), relayout)
        prefix = "comparison" if comparing() else f"{UV.source}_{UV.mjd}"
        name = f"{prefix}_{target}.{fmt}"
        if fmt == "html":
            return (dcc.send_string(
                fig.to_html(include_plotlyjs=True, full_html=True), name),
                f"saved {name} (interactive)")
        height = int(fig.layout.height or 600)
        width = int(fig.layout.width or 1400)
        try:
            img = fig.to_image(format=fmt, width=width, height=height,
                               scale=2 if fmt == "png" else 1)
        except Exception as exc:                    # kaleido / chrome missing
            return no_update, f"{fmt.upper()} export failed: {exc}"[:300]
        return (dcc.send_bytes(lambda buf: buf.write(img), name),
                f"saved {name} ({width}×{height})")

    # ---- table -----------------------------------------------------------
    def current_frame(kind, pol, avg, snr, opts, ut):
        p = params(pol, avg, snr, opts, ut)
        ref0 = all_sites()[0] if comparing() else UV.sites[0]
        if kind == "vis":
            d = vis_frame(p)
        elif kind == "cp":
            d = cphase_frame(p, "min", ref0, 0.1, 0.0, ())
        else:
            d = camp_frame(p, "min", ref0, 0.1, 0.0, (), True)
        if d is None:
            return pd.DataFrame()
        d = d.copy()
        if kind == "vis":
            d["u"] /= 1e9
            d["v"] /= 1e9
            d["uvdist"] /= 1e9
        else:
            d["uvmin"] /= 1e9
            d["uvmax"] /= 1e9
        # dataset first (compare mode), then a friendly UT column after time
        ut_pos = 1
        if "dataset" in d.columns:
            d.insert(0, "dataset", d.pop("dataset"))
            ut_pos = 2
        d.insert(ut_pos, "UT", [hours_to_hms(t) for t in d["time"]])
        return d.round(6)

    @app.callback(Output("tbl-wrap", "children"), Output("tbl-count", "children"),
                  Input("tbl-kind", "value"), Input("pol", "value"),
                  Input("avg", "value"), Input("snr", "value"),
                  Input("opts", "value"), Input("ut", "data"),
                  Input("theme", "data"))
    def _table(kind, pol, avg, snr, opts, ut, theme):
        d = current_frame(kind, pol, avg, snr, opts, ut)
        ink = fg.INK[theme]
        tbl = dash_table.DataTable(
            data=d.head(3000).to_dict("records"),
            columns=[{"name": c, "id": c} for c in d.columns],
            page_size=25, sort_action="native", filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"fontFamily": fg.FONT, "fontSize": 12,
                        "padding": "6px 10px", "backgroundColor": ink["surface"],
                        "color": ink["primary"], "border": f"1px solid {ink['grid']}"},
            style_header={"backgroundColor": ink["plane"], "fontWeight": 600,
                          "color": ink["secondary"]})
        note = f"{len(d)} rows"
        if len(d) > 3000:
            note += " (showing first 3000 — download for all)"
        return tbl, note

    @app.callback(Output("tbl-download", "data"), Input("tbl-dl", "n_clicks"),
                  State("tbl-kind", "value"), State("pol", "value"),
                  State("avg", "value"), State("snr", "value"),
                  State("opts", "value"), State("ut", "data"),
                  prevent_initial_call=True)
    def _download(n, kind, pol, avg, snr, opts, ut):
        d = current_frame(kind, pol, avg, snr, opts, ut)
        prefix = "comparison" if comparing() else f"{UV.source}_{UV.mjd}"
        name = f"{prefix}_{kind}_{pol}.csv"
        return dcc.send_data_frame(d.to_csv, name, index=False)


# --------------------------------------------------------------------------
def _quiet_dev_server_banner():
    """Drop Werkzeug's 'This is a development server...' notice.

    The whole start-up banner is one info-level record on the ``werkzeug``
    logger, so a filter keyed on the warning text removes it (and the redundant
    ' * Running on' line -- we print our own URL) while leaving the per-request
    access logs alone.
    """
    logging.getLogger("werkzeug").addFilter(
        lambda rec: "This is a development server" not in rec.getMessage())


def _free_port(host, port, tries=20):
    """First bindable port at or after ``port``.

    A leftover viewer (or any other program) sitting on the default port used
    to make the launch die with "Address already in use"; instead we walk up to
    the next free port so a double-click always starts *something*.
    """
    for p in range(port, port + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return port                                        # give up; let it error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("uvfits", nargs="*", default=[],
                    help="one uvfits file to open, or two/three to compare; "
                         "omit to pick them in the browser")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open a browser tab automatically")
    args = ap.parse_args()

    app = Dash(__name__, title="VLBI Viewer", update_title=None,
               assets_folder="assets", suppress_callback_exceptions=True)

    if args.uvfits:
        print(f"Loading {len(args.uvfits)} file(s) ...")
        set_datasets([uvdata.UVData.load(f, name=os.path.basename(f))
                      for f in args.uvfits])
        for ds in DATASETS:
            s = ds.summary()
            print(f"  {ds.filename}: {s['source']}  MJD {s['mjd']}  "
                  f"{s['n_vis']} visibilities  {s['n_sites']} sites  "
                  f"{s['n_baselines']} baselines")
        if comparing():
            print(f"  comparison mode — {len(DATASETS)} datasets overlaid")
        app.layout = app_shell(build_layout(UV, UV.filename), "loaded")
    else:
        print("No file given — pick one (or several) in the browser.")
        app.layout = app_shell(None, "landing")

    register(app)

    _quiet_dev_server_banner()
    port = _free_port(args.host, args.port)
    if port != args.port:
        print(f"  port {args.port} was busy — using {port} instead")
    url = f"http://{args.host}:{port}"
    print(f"  {url}")
    # Pop a browser tab open once the server is up.  Skipped in --debug because
    # Werkzeug's reloader would otherwise fire it in both processes.
    if not args.no_browser and not args.debug:
        threading.Timer(1.3, lambda: webbrowser.open(url)).start()
    # threaded so the fast visibility plots aren't queued behind a slow
    # per-dataset closure computation (comparison mode fires several callbacks)
    app.run(host=args.host, port=port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()

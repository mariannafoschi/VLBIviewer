"""Light-weight reader and closure-quantity calculator for VLBI uvfits files.

Only depends on numpy / pandas / astropy so the viewer can run anywhere.
Conventions (baseline ordering, sigma from weights, minimal closure sets)
follow eht-imaging so results are directly comparable; see
``validate_against_ehtim.py`` for the numerical cross-check.
"""

from __future__ import annotations

import itertools as it
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from astropy.io import fits

# Circular-polarisation products in the order they appear on the STOKES axis
CIRC_POLS = ["RR", "LL", "RL", "LR"]
# What the viewer offers the user
POL_CHOICES = ["I", "Q", "U", "V", "RR", "LL", "RL", "LR"]


def _jd_from_group(data, header):
    """Reconstruct the Julian date of every visibility group."""
    parnames = [p.upper() for p in data.parnames]
    idx = [i for i, p in enumerate(parnames) if p.startswith("DATE")]
    jd = np.zeros(len(data), dtype="d")
    for n, i in enumerate(idx):
        scal = header.get(f"PSCAL{i + 1}", 1.0)
        zero = header.get(f"PZERO{i + 1}", 0.0)
        col = data.par(i).astype("d")
        # PZERO on the first DATE column carries the reference JD; on the
        # second it is (almost always) zero.
        jd += scal * col + (zero if n == 0 or zero < 1e5 else 0.0)
    return jd


@dataclass
class UVData:
    """All visibilities of one uvfits file, plus derived quantities."""

    filename: str
    source: str
    mjd: int
    rf: float                      # reference frequency [Hz]
    bw: float                      # total bandwidth [Hz]
    sites: np.ndarray              # site names in array-table order
    df: pd.DataFrame               # one row per (time, baseline)
    scans: np.ndarray | None       # (nscan, 2) start/stop in hours
    _avg_cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, filename, name: str | None = None) -> "UVData":
        """Read a uvfits file.

        ``filename`` may be a path or any file-like object (e.g. an
        ``io.BytesIO`` holding an uploaded file).  ``name`` overrides the
        display name stored on the returned object; it defaults to the path
        when ``filename`` is a string and to ``"uploaded"`` otherwise.
        """
        hdulist = fits.open(filename)
        head = hdulist[0].header
        data = hdulist[0].data

        antab = hdulist["AIPS AN"]
        tnames = np.array([s.strip() for s in antab.data["ANNAME"]])
        tnums = antab.data["NOSTA"].astype(int) - 1
        order = np.argsort(tnums)
        sites = tnames[order]

        # ---- frequency axis -------------------------------------------------
        ctypes = [head.get(f"CTYPE{i}", "") for i in range(2, 8)]
        freq_ax = ctypes.index("FREQ") + 2
        nchan = head[f"NAXIS{freq_ax}"]
        ch_bw = head[f"CDELT{freq_ax}"]
        rf = head[f"CRVAL{freq_ax}"]
        try:
            fqtab = hdulist["AIPS FQ"].data
            if_freqs = np.atleast_1d(np.squeeze(fqtab["IF FREQ"])).astype(float)
            if_bw = np.atleast_1d(np.squeeze(fqtab["TOTAL BANDWIDTH"])).astype(float)
        except (KeyError, IndexError):
            if_freqs, if_bw = np.array([0.0]), np.array([nchan * ch_bw])
        nif = len(if_freqs)
        bw = float(np.sum(if_bw))

        # ---- group parameters ----------------------------------------------
        jd = _jd_from_group(data, head)
        mjd = int(np.min(jd) - 2400000.5)
        time = (jd - 2400000.5 - mjd) * 24.0                    # hours

        bl = data.par("BASELINE").astype(int)
        a1 = bl // 256 - 1
        a2 = bl - 256 * (bl // 256) - 1
        t1, t2 = sites[a1], sites[a2]

        try:
            tint = np.asarray(data.par("INTTIM"), dtype=float)
        except KeyError:
            tint = np.zeros(len(bl))

        uu = np.asarray(data.par("UU---SIN"), dtype=float) * rf   # lambda
        vv = np.asarray(data.par("VV---SIN"), dtype=float) * rf

        # ---- visibilities ---------------------------------------------------
        # DATA axes: (ngroup, dec, ra, IF, chan, stokes, [re, im, weight])
        arr = data.data
        arr = arr.reshape(len(bl), -1, nif, nchan, arr.shape[-2], 3)[:, 0]
        ncorr = arr.shape[-2]

        cols = {
            "time": time, "tint": tint, "t1": t1, "t2": t2,
            "u": uu, "v": vv, "uvdist": np.hypot(uu, vv),
        }
        for k, pol in enumerate(CIRC_POLS):
            if k >= ncorr:
                cols[pol] = np.full(len(bl), np.nan + 1j * np.nan)
                cols["sigma_" + pol] = np.full(len(bl), np.nan)
                continue
            re, im, wt = arr[..., k, 0], arr[..., k, 1], arr[..., k, 2]
            good = (wt > 0) & np.isfinite(wt) & np.isfinite(re) & np.isfinite(im)
            wt = np.where(good, wt, np.nan)
            vis = np.where(good, re + 1j * im, np.nan)
            # inverse-variance average over IF/channel
            with np.errstate(invalid="ignore", divide="ignore"):
                wsum = np.nansum(wt, axis=(1, 2))
                wsum = np.where(np.any(good, axis=(1, 2)), wsum, np.nan)
                cols[pol] = np.nansum(vis * wt, axis=(1, 2)) / wsum
                cols["sigma_" + pol] = 1.0 / np.sqrt(wsum)

        df = pd.DataFrame(cols)
        # Stokes from circular products
        df["I"] = 0.5 * (df["RR"] + df["LL"])
        df["Q"] = 0.5 * (df["RL"] + df["LR"])
        df["U"] = -0.5j * (df["RL"] - df["LR"])
        df["V"] = 0.5 * (df["RR"] - df["LL"])
        for s, (a, b) in {"I": ("RR", "LL"), "Q": ("RL", "LR"),
                          "U": ("RL", "LR"), "V": ("RR", "LL")}.items():
            df["sigma_" + s] = 0.5 * np.sqrt(df["sigma_" + a] ** 2
                                             + df["sigma_" + b] ** 2)

        # Where only one parallel hand survived, use it as the Stokes I estimator
        # (eht-imaging's allow_singlepol).  Flagged in "single_pol" so the viewer
        # can mark those points.
        rr_bad, ll_bad = ~np.isfinite(df["RR"]), ~np.isfinite(df["LL"])
        df["single_pol"] = (rr_bad ^ ll_bad)
        for bad, good in ((rr_bad & ~ll_bad, "LL"), (ll_bad & ~rr_bad, "RR")):
            df.loc[bad, "I"] = df.loc[bad, good]
            df.loc[bad, "sigma_I"] = df.loc[bad, "sigma_" + good]

        df["baseline"] = df["t1"] + "-" + df["t2"]
        df = df.sort_values(["time", "baseline"], kind="stable").reset_index(drop=True)

        # ---- scan table -----------------------------------------------------
        try:
            nx = hdulist["AIPS NX"].data
            scans = np.array([[s["TIME"] - 0.5 * s["TIME INTERVAL"],
                               s["TIME"] + 0.5 * s["TIME INTERVAL"]] for s in nx]) * 24.0
        except KeyError:
            scans = None

        hdulist.close()
        if name is None:
            name = filename if isinstance(filename, str) else "uploaded"
        return cls(filename=name, source=head.get("OBJECT", "?").strip(),
                   mjd=mjd, rf=float(rf), bw=bw, sites=sites, df=df, scans=scans)

    # ------------------------------------------------------------- accessors
    @property
    def baselines(self) -> list[str]:
        return sorted(self.df["baseline"].unique())

    @property
    def available_pols(self) -> list[str]:
        return [p for p in POL_CHOICES if np.isfinite(self.df[p]).any()]

    def site_order(self) -> dict:
        """Rank of each site in the array table (sets the closure reference)."""
        return {s: i for i, s in enumerate(self.sites)}

    def summary(self) -> dict:
        d = self.df
        return {
            "source": self.source, "mjd": self.mjd,
            "freq_GHz": self.rf / 1e9, "bw_GHz": self.bw / 1e9,
            "n_vis": len(d), "n_sites": len(self.sites),
            "n_baselines": d["baseline"].nunique(),
            "t_start": float(d["time"].min()), "t_stop": float(d["time"].max()),
            "n_scans": 0 if self.scans is None else len(self.scans),
        }

    # ------------------------------------------------------------- averaging
    def averaged(self, avg: str | float = "none") -> pd.DataFrame:
        """Coherently average each baseline in time.

        ``avg`` is ``"none"``, ``"scan"``, or a number of seconds.
        """
        key = str(avg)
        if key in self._avg_cache:
            return self._avg_cache[key]

        d = self.df
        if avg == "none" or avg is None:
            out = d.copy()
        else:
            if avg == "scan":
                if self.scans is None:
                    out = d.copy()
                    self._avg_cache[key] = out
                    return out
                edges = np.unique(np.concatenate([self.scans[:, 0], self.scans[:, 1]]))
                bin_id = np.searchsorted(edges, d["time"].values, side="right")
            else:
                dt = float(avg) / 3600.0
                bin_id = np.floor((d["time"].values - d["time"].min()) / dt).astype(int)
            out = self._average_bins(d, bin_id)

        self._avg_cache[key] = out
        return out

    @staticmethod
    def _average_bins(d: pd.DataFrame, bin_id: np.ndarray) -> pd.DataFrame:
        d = d.assign(_bin=bin_id)
        pols = POL_CHOICES
        rows = []
        for (_, bl), g in d.groupby(["_bin", "baseline"], sort=False):
            row = {"time": np.average(g["time"], weights=None),
                   "tint": g["tint"].sum(), "baseline": bl,
                   "t1": g["t1"].iloc[0], "t2": g["t2"].iloc[0],
                   "u": g["u"].mean(), "v": g["v"].mean(),
                   "n_avg": len(g)}
            row["uvdist"] = np.hypot(row["u"], row["v"])
            for p in pols:
                vis, sig = g[p].values, g["sigma_" + p].values
                good = np.isfinite(vis) & np.isfinite(sig) & (sig > 0)
                if not good.any():
                    row[p], row["sigma_" + p] = np.nan + 0j, np.nan
                    continue
                w = 1.0 / sig[good] ** 2
                row[p] = np.sum(vis[good] * w) / np.sum(w)
                row["sigma_" + p] = 1.0 / np.sqrt(np.sum(w))
            rows.append(row)
        return (pd.DataFrame(rows)
                .sort_values(["time", "baseline"], kind="stable")
                .reset_index(drop=True))

    # ------------------------------------------------------------- selection
    def table(self, pol: str = "I", avg: str | float = "none",
              snr_cut: float = 0.0, debias: bool = False) -> pd.DataFrame:
        """Per-visibility table of amplitude / phase / errors for one pol."""
        d = self.averaged(avg)
        vis, sig = d[pol].values, d["sigma_" + pol].values
        amp = np.abs(vis)
        if debias:
            amp = np.sqrt(np.clip(amp ** 2 - sig ** 2, 0.0, None))
        snr = amp / sig
        out = pd.DataFrame({
            "time": d["time"].values, "baseline": d["baseline"].values,
            "t1": d["t1"].values, "t2": d["t2"].values,
            "u": d["u"].values, "v": d["v"].values,
            "uvdist": d["uvdist"].values, "tint": d["tint"].values,
            "amp": amp, "sigma": sig, "snr": snr,
            "phase": np.degrees(np.angle(vis)),
            # error propagation for the phase of a complex Gaussian
            "sigma_phase": np.degrees(np.where(snr > 0, 1.0 / snr, np.nan)),
            "re": np.real(vis), "im": np.imag(vis),
        })
        out = out[np.isfinite(out["amp"]) & np.isfinite(out["sigma"])]
        if snr_cut > 0:
            out = out[out["snr"] >= snr_cut]
        return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Closure quantities
# --------------------------------------------------------------------------
def amp_debias(amp, sigma):
    """Debiased amplitude sqrt(A^2 - sigma^2), floored at sigma for snr < 1."""
    amp, sigma = np.abs(np.asarray(amp, float)), np.abs(np.asarray(sigma, float))
    hi = amp > sigma
    return np.sqrt(np.where(hi, amp ** 2 - sigma ** 2, sigma ** 2))


def _tri_minimal(sites_present, order, ref=None):
    """Non-redundant triangles: every triangle contains the reference site.

    Gives (N-1)(N-2)/2 closure phases for N sites, the number of independent
    closure phases -- same construction as ehtim's ``tri_minimal_set``.
    """
    ordered = sorted(sites_present, key=lambda s: order.get(s, 99))
    if ref is not None and ref in ordered:
        ordered.remove(ref)
        ordered = [ref] + ordered
    if len(ordered) < 3:
        return []
    r, rest = ordered[0], ordered[1:]
    return [(r, a, b) for a, b in it.combinations(rest, 2)]


def _quad_minimal(sites_present, order, ref=None):
    """Non-redundant quadrangles, N(N-3)/2 of them (ehtim ``quad_minimal_set``)."""
    ordered = sorted(sites_present, key=lambda s: order.get(s, 99))
    if ref is not None and ref in ordered:
        ordered.remove(ref)
        ordered = [ref] + ordered
    if len(ordered) < 4:
        return []
    r = ordered[0]
    quads = []
    for i in range(3, len(ordered)):
        for j in range(1, i):
            k = 1 if j == i - 1 else j + 1
            quads.append((r, ordered[i], ordered[j], ordered[k]))
    return quads


def _tri_maximal(sites_present, order, ref=None):
    ordered = sorted(sites_present, key=lambda s: order.get(s, 99))
    return list(it.combinations(ordered, 3))


def _quad_maximal(sites_present, order, ref=None):
    ordered = sorted(sites_present, key=lambda s: order.get(s, 99))
    out = []
    for q in it.combinations(ordered, 4):
        out += [(q[0], q[1], q[2], q[3]),
                (q[0], q[2], q[1], q[3]),
                (q[0], q[1], q[3], q[2])]
    return out


def _time_dicts(tab: pd.DataFrame):
    """Group rows by timestamp -> {(site_a, site_b): row-index}."""
    for t, g in tab.groupby("time", sort=True):
        d = {}
        for i, r in zip(g.index, g.itertuples()):
            d[(r.t1, r.t2)] = i
        yield t, g, d


def _lookup(d, tab, a, b):
    """Visibility a->b, conjugating if the file stores b->a."""
    if (a, b) in d:
        i = d[(a, b)]
        return (tab["amp"].values[i],
                np.radians(tab["phase"].values[i]),
                tab["sigma"].values[i],
                tab["u"].values[i], tab["v"].values[i])
    if (b, a) in d:
        i = d[(b, a)]
        return (tab["amp"].values[i],
                -np.radians(tab["phase"].values[i]),
                tab["sigma"].values[i],
                -tab["u"].values[i], -tab["v"].values[i])
    return None


def closure_phases(tab: pd.DataFrame, order: dict, count: str = "min",
                   ref: str | None = None, uv_min: float = 0.0,
                   snr_cut: float = 0.0) -> pd.DataFrame:
    """Closure phases in degrees.

    ``count='min'`` returns a non-redundant set built on a reference station;
    ``count='max'`` returns every unique triangle.  Triangles containing a
    baseline shorter than ``uv_min`` (in lambda) are dropped -- those are the
    trivial intra-site triangles whose closure phase is identically zero.
    """
    build = _tri_minimal if count == "min" else _tri_maximal
    rows = []
    for t, g, d in _time_dicts(tab):
        present = set(g["t1"]).union(g["t2"])
        for tri in build(present, order, ref):
            legs = [_lookup(d, tab, tri[0], tri[1]),
                    _lookup(d, tab, tri[1], tri[2]),
                    _lookup(d, tab, tri[2], tri[0])]
            if any(l is None for l in legs):
                continue
            amps = np.array([l[0] for l in legs])
            phis = np.array([l[1] for l in legs])
            sigs = np.array([l[2] for l in legs])
            uvd = np.array([np.hypot(l[3], l[4]) for l in legs])
            if uv_min > 0 and np.any(uvd < uv_min):
                continue
            snr_leg = amps / sigs
            if snr_cut > 0 and np.any(snr_leg < snr_cut):
                continue
            cp = np.degrees(np.angle(np.exp(1j * phis.sum())))
            err = np.degrees(np.sqrt(np.sum(1.0 / snr_leg ** 2)))
            rows.append({
                "time": t, "triangle": "-".join(tri),
                "t1": tri[0], "t2": tri[1], "t3": tri[2],
                "cphase": cp, "sigma_cp": err,
                "snr": 1.0 / np.sqrt(np.sum(1.0 / snr_leg ** 2)),
                "uvmin": uvd.min(), "uvmax": uvd.max(),
            })
    cols = ["time", "triangle", "t1", "t2", "t3", "cphase", "sigma_cp", "snr",
            "uvmin", "uvmax"]
    return pd.DataFrame(rows, columns=cols)


def closure_amplitudes(tab: pd.DataFrame, order: dict, count: str = "min",
                       ref: str | None = None, uv_min: float = 0.0,
                       snr_cut: float = 0.0, debias: bool = True,
                       log: bool = True) -> pd.DataFrame:
    """Closure amplitudes  |V12||V34| / (|V14||V23|)  (log by default)."""
    build = _quad_minimal if count == "min" else _quad_maximal
    rows = []
    for t, g, d in _time_dicts(tab):
        present = set(g["t1"]).union(g["t2"])
        for q in build(present, order, ref):
            legs = [_lookup(d, tab, q[0], q[1]),   # numerator
                    _lookup(d, tab, q[2], q[3]),
                    _lookup(d, tab, q[0], q[3]),   # denominator
                    _lookup(d, tab, q[1], q[2])]
            if any(l is None for l in legs):
                continue
            amps = np.array([l[0] for l in legs])
            sigs = np.array([l[2] for l in legs])
            uvd = np.array([np.hypot(l[3], l[4]) for l in legs])
            if uv_min > 0 and np.any(uvd < uv_min):
                continue
            snr_leg = amps / sigs
            if snr_cut > 0 and np.any(snr_leg < snr_cut):
                continue
            a = amp_debias(amps, sigs) if debias else amps
            snr = a / sigs
            var = np.sum(1.0 / snr ** 2)
            if log:
                camp = np.sum(np.log(a[:2])) - np.sum(np.log(a[2:]))
                if debias:  # Blackburn et al. 2020, eq. 32
                    camp += 0.5 * (1 / snr[0] ** 2 + 1 / snr[1] ** 2
                                   - 1 / snr[2] ** 2 - 1 / snr[3] ** 2)
                err = np.sqrt(var)
            else:
                camp = (a[0] * a[1]) / (a[2] * a[3])
                err = camp * np.sqrt(var)
                if debias:
                    camp = camp / (1 + 1 / snr[2] ** 2 + 1 / snr[3] ** 2)
            rows.append({
                "time": t, "quadrangle": "-".join(q),
                "t1": q[0], "t2": q[1], "t3": q[2], "t4": q[3],
                "camp": camp, "sigma_camp": err,
                "snr": 1.0 / np.sqrt(var), "uvmin": uvd.min(), "uvmax": uvd.max(),
            })
    cols = ["time", "quadrangle", "t1", "t2", "t3", "t4", "camp", "sigma_camp",
            "snr", "uvmin", "uvmax"]
    return pd.DataFrame(rows, columns=cols)


def hours_to_hms(h: float) -> str:
    # Not wrapped at 24 h: tracks that cross midnight keep a continuous time
    # axis (time may exceed 24 h), and the header UT label, click records and
    # table must agree with that axis instead of folding back to 0.
    h = float(h)
    hh = int(h)
    mm = int((h - hh) * 60)
    ss = (h - hh - mm / 60) * 3600
    return f"{hh:02d}:{mm:02d}:{ss:05.2f}"

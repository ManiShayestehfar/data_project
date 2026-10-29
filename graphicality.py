"""Power-law graphicality, degree repair and configuration-model erasure."""

import hashlib
import json
import platform
from functools import lru_cache
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL = "IID discrete power law on 1,...,n-1, conditioned on even sum"
TARGET_METRICS = ["Q", "a", "c_star", "c_last", "h_star", "internal", "external", "f"]
ERASURE_METRICS = ["a", "erased_a", "extra_a", "loop_a", "parallel_a"]


def settings(mode="full"):
    if mode == "full":
        return dict(mode=mode, sizes=[100, 300, 1000, 3000, 10000],
                    gammas=np.round(np.arange(-0.5, 3.01, 0.1), 3).tolist(),
                    repeats=200, bootstraps=1000, local_gammas=[1.3, 1.5, 1.8],
                    erase_sizes=[200, 500, 1000], erase_gammas=[0.5, 1.3, 1.8, 2.5],
                    erase_repeats=200, pairings=5)
    if mode == "quick":
        return dict(mode=mode, sizes=[100, 300, 1000],
                    gammas=[-0.5, 0, 0.5, 1, 1.3, 1.5, 1.8, 2, 2.5, 3],
                    repeats=20, bootstraps=300, local_gammas=[1.3, 1.5, 1.8],
                    erase_sizes=[100, 300], erase_gammas=[0.5, 1.3, 1.8, 2.5],
                    erase_repeats=10, pairings=3)
    raise ValueError('mode must be "quick" or "full"')


def rng_for(seed, stream, n=0, gamma=0, rep=0, pairing=0):
    key = int(round(1000 * (gamma + 10)))
    return np.random.default_rng(np.random.SeedSequence(
        [seed, stream, int(n), key, int(rep), int(pairing)]))


@lru_cache(maxsize=None)
def degree_law(n, gamma):
    if n < 2:
        raise ValueError("n must be at least 2")
    values = np.arange(1, n, dtype=np.int64)
    weights = values.astype(float) ** (-gamma)
    return values, weights / weights.sum()


def generate_degrees(n, gamma, rng):
    values, probabilities = degree_law(n, float(gamma))
    while True:
        d = rng.choice(values, size=n, p=probabilities)
        if d.sum() % 2 == 0:
            return np.sort(d)[::-1]


def quantile_degrees(n, gamma):
    values, probabilities = degree_law(n, float(gamma))
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0
    d = values[np.searchsorted(cdf, (np.arange(n) + 0.5) / n)]
    if d.sum() % 2:
        d[np.flatnonzero(d < n - 1)[0]] += 1
    return np.sort(d)[::-1]


def _slacks_sorted(d):
    n = len(d)
    k = np.arange(1, n + 1, dtype=np.int64)
    prefix = np.r_[0, np.cumsum(d)]
    counts = np.searchsorted(-d, -k, side="right")
    m = np.maximum(k, counts)
    return k * (k - 1) + k * np.maximum(counts - k, 0) + prefix[-1] - prefix[m] - prefix[k]


def erdos_gallai_deficit(d):
    return _slacks_sorted(np.sort(np.asarray(d, dtype=np.int64))[::-1])


def eg_summary(d):
    d = np.asarray(d)
    if d.ndim != 1 or not len(d) or not np.issubdtype(d.dtype, np.integer):
        raise ValueError("degrees must be a nonempty integer vector")
    if np.any(d < 0) or np.any(d >= len(d)) or d.sum() % 2:
        raise ValueError("degrees must lie in 0,...,n-1 and have even sum")
    d = np.sort(d.astype(np.int64))[::-1]
    delta = _slacks_sorted(d)
    S = int(d.sum())
    F = max(0, -int(delta.min()))
    Q = 2 * ((F + 1) // 2)
    ks = int(delta.argmin() + 1) if F else np.nan
    kl = int(np.flatnonzero(delta < 0)[-1] + 1) if F else np.nan
    h = float(d[:int(ks)].sum() / S) if F else np.nan
    internal = ks * (ks - 1) / S if F else np.nan
    external = float(np.minimum(int(ks), d[int(ks):]).sum() / S) if F else np.nan
    return dict(S=S, F=F, Q=Q, a=Q/S if S else np.nan, graphical=(Q == 0),
                k_star=ks, k_last=kl, c_star=ks/len(d), c_last=kl/len(d),
                h_star=h, internal=internal, external=external,
                f=F/S if F else np.nan, density=S/(len(d)*(len(d)-1)) if len(d)>1 else 0)


def enrich_targets(frame):
    t = frame.copy()
    t["graphical"] = t.Q.eq(0)
    t["f"] = (t.F/t.S).where(t.Q > 0)
    t["internal"] = t.k_star*(t.k_star-1)/t.S
    t["external"] = t.h_star-t.internal-t.f
    t["density"] = t.S/(t.n*(t.n-1))
    assert np.allclose(t.Q, 2*((t.F+1)//2))
    assert t.S.mod(2).eq(0).all() and t.Q.between(0, t.S).all()
    assert (t.loc[t.Q > 0, "external"] >= -1e-12).all()
    return t


def wilson_interval(successes, total):
    z = 1.959963984540054
    p = successes/total
    denominator = 1+z*z/total
    centre = (p+z*z/(2*total))/denominator
    radius = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return max(0., centre-radius), min(1., centre+radius)


def mean_ci(values, rng, bootstraps):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return np.nan, np.nan, np.nan, 0, "no observations"
    mean = float(x.mean())
    if x.size < 2 or np.ptp(x) == 0:
        status = "all zero; bootstrap uninformative" if mean == 0 else "constant sample; bootstrap uninformative"
        return mean, np.nan, np.nan, x.size, status
    means = x[rng.integers(x.size, size=(bootstraps, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return mean, float(lo), float(hi), x.size, "percentile bootstrap"


def summarise(frame, metrics, seed, bootstraps, conditioned=False):
    keys = ["n", "gamma"] + (["graphical"] if conditioned else [])
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        n, gamma = int(key[0]), float(key[1])
        row = dict(zip(keys, key))
        row["targets"] = len(group)
        for metric in metrics:
            token = int.from_bytes(hashlib.sha256(metric.encode()).digest()[:4], "little")
            stream = 90 + token + (int(bool(key[2])) + 1 if conditioned else 0)
            mean, lo, hi, count, status = mean_ci(group[metric], rng_for(seed, stream, n, gamma), bootstraps)
            row.update({metric: mean, metric+"_lo": lo, metric+"_hi": hi,
                        metric+"_count": count, metric+"_ci_method": status})
        successes = int(group.graphical.sum())
        row["graphical_count"], row["non_graphical_count"] = successes, len(group)-successes
        row["p_graphical"] = successes/len(group)
        row["p_graphical_lo"], row["p_graphical_hi"] = wilson_interval(successes, len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_run(out, cfg, seed):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/"run_parameters.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous.get("seed") != seed or previous.get("model") != MODEL:
            raise ValueError("Existing results use a different seed or model; choose a new output directory.")
    elif any((out/name).exists() for name in ["targets.csv", "erasure_pairings.csv"]):
        raise ValueError("Cached data have no run_parameters.json; choose a new output directory.")
    else:
        manifest.write_text(json.dumps(dict(seed=seed, model=MODEL, status="running",
            python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
            matplotlib=matplotlib.__version__), indent=2)+"\n")
    return out


def _cached(path, keys):
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(frame) and frame.duplicated(keys).any():
        raise ValueError(f"Duplicate cache keys in {path}")
    records = frame.to_dict("records")
    seen = set(frame[keys].itertuples(index=False, name=None)) if len(frame) else set()
    return records, seen


def target_experiment(cfg, seed, out):
    out = prepare_run(out, cfg, seed)
    records, seen = _cached(out/"targets.csv", ["n", "gamma", "rep"])
    for n in cfg["sizes"]:
        added = 0
        for gamma in cfg["gammas"]:
            for rep in range(cfg["repeats"]):
                if (n, gamma, rep) not in seen:
                    d = generate_degrees(n, gamma, rng_for(seed, 0, n, gamma, rep))
                    records.append(dict(n=n, gamma=gamma, rep=rep, **eg_summary(d)))
                    added += 1
        pd.DataFrame(records).to_csv(out/"targets.csv", index=False)
        print(f"Targets n={n}: {added} new")
    all_targets = enrich_targets(pd.DataFrame(records))
    all_targets.to_csv(out/"targets.csv", index=False)
    active = all_targets[all_targets.n.isin(cfg["sizes"]) & all_targets.gamma.isin(cfg["gammas"])
                         & (all_targets.rep < cfg["repeats"])].copy()
    summary = summarise(active, TARGET_METRICS, seed, cfg["bootstraps"])
    summary.to_csv(out/"target_summary.csv", index=False)
    quantiles = pd.DataFrame([dict(n=n, gamma=g, **eg_summary(quantile_degrees(n, g)))
                             for n in cfg["sizes"] for g in cfg["gammas"]])
    quantiles.to_csv(out/"quantile_benchmark.csv", index=False)
    return active, summary, quantiles


def erase_pairing(d, rng):
    n = len(d)
    stubs = np.repeat(np.arange(n, dtype=np.int64), d)
    rng.shuffle(stubs)
    pairs = stubs.reshape(-1, 2)
    lo, hi = pairs.min(axis=1), pairs.max(axis=1)
    keep = lo != hi
    loops = int((~keep).sum())
    edges = np.unique(lo[keep]*n+hi[keep])
    parallel = int(keep.sum()-edges.size)
    degrees = np.bincount(edges//n, minlength=n)+np.bincount(edges % n, minlength=n)
    assert np.all(degrees <= d)
    assert int(np.sum(d)-degrees.sum()) == 2*(loops+parallel)
    return degrees, loops, parallel


def erasure_experiment(cfg, seed, out):
    out = prepare_run(out, cfg, seed)
    records, seen = _cached(out/"erasure_pairings.csv", ["n", "gamma", "rep", "pairing"])
    known_targets = {}
    for row in records:
        key = (row["n"], row["gamma"], row["rep"])
        signature = (int(row["S"]), int(row["Q"]))
        if key in known_targets and known_targets[key] != signature:
            raise ValueError("Cached pairings have inconsistent target measurements.")
        known_targets[key] = signature
    for n in cfg["erase_sizes"]:
        added = 0
        for gamma in cfg["erase_gammas"]:
            for rep in range(cfg["erase_repeats"]):
                missing = [j for j in range(cfg["pairings"]) if (n, gamma, rep, j) not in seen]
                if not missing:
                    continue
                d = generate_degrees(n, gamma, rng_for(seed, 1, n, gamma, rep))
                result = eg_summary(d)
                S, Q = result["S"], result["Q"]
                if (n, gamma, rep) in known_targets and known_targets[n, gamma, rep] != (S, Q):
                    raise ValueError("Cached target cannot be reproduced; use the original package versions or a new output directory.")
                for j in missing:
                    degrees, loops, parallel = erase_pairing(d, rng_for(seed, 2, n, gamma, rep, j))
                    D = int(S-degrees.sum())
                    assert Q <= D <= S
                    records.append(dict(n=n, gamma=gamma, rep=rep, pairing=j, S=S, Q=Q, D=D,
                        a=Q/S, erased_a=D/S, extra_a=(D-Q)/S, loops=loops, parallel=parallel))
                    added += 1
        pd.DataFrame(records).to_csv(out/"erasure_pairings.csv", index=False)
        print(f"Erasure n={n}: {added} new pairings")
    all_pairs = pd.DataFrame(records)
    all_pairs["graphical"] = all_pairs.Q.eq(0)
    all_pairs["loop_a"] = 2*all_pairs.loops/all_pairs.S
    all_pairs["parallel_a"] = 2*all_pairs.parallel/all_pairs.S
    assert np.allclose(all_pairs.D, 2*(all_pairs.loops+all_pairs.parallel))
    assert (all_pairs.D >= all_pairs.Q).all()
    all_pairs.to_csv(out/"erasure_pairings.csv", index=False)
    pairs = all_pairs[all_pairs.n.isin(cfg["erase_sizes"]) & all_pairs.gamma.isin(cfg["erase_gammas"])
                      & (all_pairs.rep < cfg["erase_repeats"]) & (all_pairs.pairing < cfg["pairings"])].copy()
    per_target = pairs.groupby(["n", "gamma", "rep"], as_index=False)[
        ["S", "Q"]+ERASURE_METRICS].mean()
    per_target["graphical"] = per_target.Q.eq(0)
    per_target.to_csv(out/"erasure_targets.csv", index=False)
    summary = summarise(per_target, ERASURE_METRICS, seed, cfg["bootstraps"])
    summary.to_csv(out/"erasure_summary.csv", index=False)
    conditional = summarise(per_target, ["erased_a", "extra_a", "loop_a", "parallel_a"],
                            seed, cfg["bootstraps"], conditioned=True)
    conditional.to_csv(out/"erasure_by_graphicality.csv", index=False)
    variance_rows = []
    for (n, gamma), group in pairs.groupby(["n", "gamma"]):
        per = group.groupby("rep").erased_a
        between, within = per.mean().var(ddof=1), per.var(ddof=1).mean()
        pairing_component = within/cfg["pairings"]
        variance_rows.append(dict(n=n, gamma=gamma, target_mean_variance=between,
            mean_within_target_variance=within, pairing_component=pairing_component,
            estimated_target_component=max(0., between-pairing_component),
            pairing_fraction=pairing_component/between if between > 0 else np.nan))
    pd.DataFrame(variance_rows).to_csv(out/"variance_components.csv", index=False)
    return pairs, per_target, summary, conditional


def finish_run(cfg, seed, out):
    path = Path(out)/"run_parameters.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    versions = dict(python=platform.python_version(), numpy=np.__version__,
                    pandas=pd.__version__, matplotlib=matplotlib.__version__)
    history = previous.get("history", []) if previous else []
    if previous:
        entry = {k: previous[k] for k in ["seed", "repeats", "erase_repeats", "pairings",
                                         "python", "numpy", "pandas", "matplotlib"] if k in previous}
        if entry not in history:
            history.append(entry)
    metadata = dict(**cfg, seed=seed, model=MODEL, **versions, history=history,
        cache="Existing keyed samples reused; missing replications appended. Versions in history may differ.",
        rng="SeedSequence([seed, stream, n, round(1000*(gamma+10)), rep, pairing])",
        streams={"targets": 0, "erasure_targets": 1, "pairings": 2,
                 "bootstrap": "90 + first 4 bytes of SHA256(metric), little endian; conditional groups add 1 or 2"},
        intervals="Pointwise 95%: Wilson for graphicality; percentile bootstrap across targets for means. Constant samples have missing mean intervals, not zero-width intervals.",
        conditioning="Obstruction metrics condition on Q>0. Erasure pairings averaged within each target.",
        quantiles="Midpoint inverse CDF; increase one smallest eligible degree by one if sum is odd.",
        half_limits="Gamma=1/2 continuum limit; finite-n Monte Carlo intervals exclude approximation error.")
    path.write_text(json.dumps(metadata, indent=2)+"\n")


def _style():
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.spines.top": False,
                        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .2,
                        "lines.linewidth": 1.8})


def _save(fig, out, name, show=True):
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(Path(out)/f"{name}.{extension}", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _band(ax, table, x, y, label, **kwargs):
    t = table.sort_values(x)
    line, = ax.plot(t[x], t[y], label=label, **kwargs)
    ax.fill_between(t[x], t[y+"_lo"], t[y+"_hi"], color=line.get_color(), alpha=.12, linewidth=0)


def _points(ax, table, metric, label, colour, marker="o", offset=0):
    t = table.sort_values("gamma")
    y = t[metric].to_numpy(float)
    lo, hi = t[metric+"_lo"].to_numpy(float), t[metric+"_hi"].to_numpy(float)
    ax.scatter(t.gamma+offset, y, color=colour, marker=marker, label=label, s=30)
    valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    ax.errorbar(t.gamma.to_numpy()[valid]+offset, y[valid],
        yerr=np.maximum(0, np.vstack([y[valid]-lo[valid], hi[valid]-y[valid]])),
        fmt="none", color=colour, capsize=3, linewidth=1)


def plot_targets(summary, cfg, out, show=True):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for n, table in summary.groupby("n"):
        for ax, metric in zip(axes, ["p_graphical", "Q", "a"]):
            _band(ax, table, "gamma", metric, f"n={n}")
    for ax, title, label in zip(axes, ["Exact graphicality", "Minimum discrepancy", "Relative discrepancy"],
                                [r"$P(Q=0)$", r"Mean $Q$", r"Mean $Q/S$"]):
        ax.set(xlabel=r"$\gamma$", ylabel=label, title=title)
    axes[0].set_ylim(-.02, 1.02)
    axes[2].set_ylim(bottom=0)
    axes[2].legend(fontsize=8)
    _save(fig, out, "01_graphicality_and_repair", show)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for gamma in cfg["local_gammas"]:
        table = summary[np.isclose(summary.gamma, gamma)]
        for ax, metric in zip(axes, ["c_star", "c_last", "h_star"]):
            _band(ax, table, "n", metric, rf"$\gamma={gamma}$", marker="o", markersize=4)
    for ax, label in zip(axes, [r"Mean $k^*/n$", r"Mean $k_{\mathrm{last}}/n$", r"Mean $h_*$"]):
        ax.set(xscale="log", xlabel="$n$", ylabel=label, ylim=(0, 1.02))
    axes[-1].legend()
    _save(fig, out, "02_obstruction_concentration", show)


def plot_capacity(summary, cfg, out, show=True):
    fig, axes = plt.subplots(1, len(cfg["local_gammas"]), figsize=(12, 3.7), squeeze=False)
    for ax, gamma in zip(axes[0], cfg["local_gammas"]):
        t = summary[np.isclose(summary.gamma, gamma)].sort_values("n")
        valid = t[["internal", "external", "f"]].notna().all(axis=1)
        t = t[valid]
        ax.stackplot(t.n, t.internal, t.external, t.f,
                     labels=["Internal capacity", "External capacity", "Violation"], alpha=.65)
        ax.plot(t.n, t.h_star, "k-o", markersize=3, label="Demand")
        ax.set(xscale="log", xlabel="$n$", ylabel="Fraction of total degree demand",
               ylim=(0, 1), title=rf"$\gamma={gamma}$; conditional on $Q>0$")
    axes[0, -1].legend(fontsize=8)
    _save(fig, out, "05_capacity_decomposition", show)


def plot_erasure(summary, conditional, out, show=True):
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for colour, (n, t) in zip(colours, summary.groupby("n")):
        _points(axes[0], t, "a", f"n={n}: minimum", colour, "o")
        _points(axes[0], t, "erased_a", f"n={n}: erased", colour, "s")
        _points(axes[1], t, "extra_a", f"n={n}", colour)
    for ax, label, title in zip(axes, ["Mean degree discrepancy / S", r"Mean $(D-Q)/S$"],
                                ["Minimum versus erasure", "Additional discrepancy"]):
        ax.set(xlabel=r"$\gamma$", ylabel=label, ylim=(0, .65), title=title)
        ax.legend(fontsize=8)
    _save(fig, out, "03_erasure_cost", show)
    groups = list(summary.groupby("n"))
    fig, axes = plt.subplots(1, len(groups), figsize=(4*len(groups), 3.7), squeeze=False)
    for ax, (n, t) in zip(axes[0], groups):
        t = t.sort_values("gamma")
        x = np.arange(len(t))
        ax.bar(x, t.parallel_a, label="Extra parallel edges", color="C0")
        ax.bar(x, t.loop_a, bottom=t.parallel_a, label="Loops", color="C1")
        y = t.erased_a.to_numpy()
        ax.errorbar(x, y, yerr=np.maximum(0, np.vstack([y-t.erased_a_lo, t.erased_a_hi-y])),
                    fmt="none", color="black", capsize=3)
        ax.set(xticks=x, xticklabels=[f"{g:g}" for g in t.gamma], xlabel=r"$\gamma$",
               ylabel="Mean erased degree / S", title=f"n={n}", ylim=(0, .65))
    axes[0, -1].legend(fontsize=8)
    _save(fig, out, "06_erasure_mechanisms", show)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, graphical in zip(axes, [True, False]):
        for j, (n, t) in enumerate(conditional[conditional.graphical == graphical].groupby("n")):
            _points(ax, t, "erased_a", f"n={n}", colours[j], offset=.012*j)
        ax.set(xlabel=r"$\gamma$", ylabel="Conditional mean D/S", ylim=(0, .65),
               title="Graphical targets" if graphical else "Non-graphical targets")
        if ax.collections:
            ax.legend(fontsize=8)
    _save(fig, out, "07_erasure_conditioned", show)


def plot_theory(summary, quantiles, cfg, out, show=True):
    selected = sorted(set([cfg["sizes"][0], cfg["sizes"][len(cfg["sizes"])//2], cfg["sizes"][-1]]))
    fig, axes = plt.subplots(1, len(selected), figsize=(4*len(selected), 3.7), squeeze=False)
    for ax, n in zip(axes[0], selected):
        t, q = summary[summary.n == n], quantiles[quantiles.n == n].sort_values("gamma")
        _band(ax, t, "gamma", "a", "Monte Carlo mean")
        ax.plot(q.gamma, q.a, "--", color="C1", label="Finite-n quantiles")
        ax.set(xlabel=r"$\gamma$", ylabel="Relative discrepancy", title=f"n={n}", ylim=(0, .6))
    axes[0, -1].legend(fontsize=8)
    _save(fig, out, "08_quantile_comparison", show)
    h = (3-np.sqrt(5))/2
    limits = dict(a=(5*np.sqrt(5)-11)/2, c_star=h, c_last=.5, h_star=1-(1-h)**3)
    labels = dict(a=r"Mean $Q/S$", c_star=r"Mean $k^*/n$",
                  c_last=r"Mean $k_{\mathrm{last}}/n$", h_star=r"Mean $h_*$")
    t = summary[np.isclose(summary.gamma, .5)].sort_values("n")
    q = quantiles[np.isclose(quantiles.gamma, .5)].sort_values("n")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    comparisons = []
    for ax, (metric, limit) in zip(axes.flat, limits.items()):
        _band(ax, t, "n", metric, "Monte Carlo mean", marker="o", markersize=4)
        ax.plot(q.n, q[metric], "--", color="C1", label="Finite-n quantiles")
        ax.axhline(limit, color="black", linestyle=":", label="Continuum limit")
        ax.set(xscale="log", xlabel="$n$", ylabel=labels[metric], title=r"$\gamma=1/2$")
        for row in t.itertuples(index=False):
            qr = q[q.n == row.n].iloc[0]
            comparisons.append(dict(n=row.n, metric=metric, observed=getattr(row, metric),
                lo=getattr(row, metric+"_lo"), hi=getattr(row, metric+"_hi"),
                quantile=qr[metric], limit=limit))
    axes[0, 0].legend(fontsize=8)
    pd.DataFrame(comparisons).to_csv(Path(out)/"half_exponent_comparison.csv", index=False)
    _save(fig, out, "09_half_exponent_limits", show)


def plot_examples(cfg, seed, out, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    records = []
    n = cfg["erase_sizes"][-1]
    for ax, gamma in zip(axes, [1.3, 2.5]):
        d = generate_degrees(n, gamma, rng_for(seed, 1, n, gamma, 0))
        degrees, _, _ = erase_pairing(d, rng_for(seed, 2, n, gamma, 0, 0))
        result = eg_summary(d)
        ranks = np.arange(1, n+1)
        ax.plot(ranks, d, label="Target")
        ax.plot(ranks, degrees, label="After erasure", alpha=.75)
        ax.set(xscale="log", yscale="symlog", ylim=(0, None), xlabel="Vertex rank in target",
               ylabel="Degree", title=rf"$\gamma={gamma}$: $Q/S={result['a']:.3f}$, $D/S={(d.sum()-degrees.sum())/d.sum():.3f}$")
        records.extend(dict(n=n, gamma=gamma, rep=0, pairing=0, rank=int(i), target=int(x), erased=int(y))
                       for i, x, y in zip(ranks, d, degrees))
    axes[0].legend()
    pd.DataFrame(records).to_csv(Path(out)/"degree_examples.csv", index=False)
    _save(fig, out, "04_degree_examples", show)


def run_all(cfg, seed, out, show=False):
    targets, main, quantiles = target_experiment(cfg, seed, out)
    plot_targets(main, cfg, out, show)
    plot_capacity(main, cfg, out, show)
    pairs, per_target, erasure, conditional = erasure_experiment(cfg, seed, out)
    plot_erasure(erasure, conditional, out, show)
    plot_theory(main, quantiles, cfg, out, show)
    plot_examples(cfg, seed, out, show)
    finish_run(cfg, seed, out)
    print(f"Complete: {len(targets)} targets; {len(pairs)} pairings. Results: {Path(out).resolve()}")

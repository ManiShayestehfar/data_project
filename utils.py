"""Numerical tools; experiment design and analysis live in the notebook."""
import hashlib
import json
import platform
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, vstack
import graphicality as gp

VERSION = 'extensions-1'


def rng(seed, *key):
    token = hashlib.sha256(json.dumps([VERSION, seed, *key], sort_keys=True).encode()).digest()
    return np.random.default_rng(np.frombuffer(token, dtype='<u4'))


@lru_cache(maxsize=256)
def law(n, gamma, alpha=1.):
    if n < 2 or not 0 < alpha <= 1:
        raise ValueError('Require n >= 2 and 0 < alpha <= 1.')
    K = min(n-1, max(1, int(np.floor(n**alpha))))
    support = np.arange(1, K+1, dtype=np.int64)
    weights = support.astype(float)**(-gamma)
    return support, np.cumsum(weights/weights.sum())


def sample(n, gamma, generator, alpha=1.):
    support, cdf = law(n, gamma, alpha)
    if len(support) == 1 and n % 2:
        raise ValueError('Even-sum conditioning has zero probability.')
    cdf[-1] = 1.
    while True:
        d = support[np.searchsorted(cdf, generator.random(n))]
        if d.sum() % 2 == 0:
            return np.sort(d)[::-1]


def rate(gamma):
    return (gamma-1)*(2-gamma)/gamma


def measure(d, gamma):
    result = gp.eg_summary(d)
    S, Q, n = result['S'], result['Q'], len(d)
    result.update(retained=(S-Q)/S, E_max=(S-Q)//2)
    if 1 < gamma < 2:
        t = int(np.ceil(n**(1/gamma)))
        m = int(np.count_nonzero(d >= t))
        B = int(d[d < t].sum())
        clique = min(m, t+1)
        lower = clique*(clique-1)//2
        upper = min(S//2, B+m*(m-1)//2)
        assert lower <= result['E_max'] <= upper
        result.update(E_scaled=result['E_max']/n**(2/gamma),
                      retained_scaled=result['retained']*n**rate(gamma),
                      E_lower=lower, E_upper=upper,
                      lower_scaled=lower/n**(2/gamma), upper_scaled=upper/n**(2/gamma))
    return result


class Cache:
    def __init__(self, path, seed):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    def get(self, experiment, key, compute):
        full_key = json.dumps([VERSION, np.__version__, self.seed, experiment, key], sort_keys=True)
        token = hashlib.sha256(full_key.encode()).hexdigest()
        path = self.path / (token+'.json')
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['key'] != full_key:
                raise ValueError('Cache key mismatch.')
            return saved['value']
        value = compute()
        with tempfile.NamedTemporaryFile(mode='w', dir=self.path, suffix='.tmp', delete=False) as handle:
            json.dump(dict(key=full_key, value=value), handle)
            temporary = handle.name
        os.replace(temporary, path)
        return value


def summarise(frame, keys, metrics, seed=5441, bootstraps=500):
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        row['targets'] = len(group)
        for metric in metrics:
            mean, lo, hi, count, status = gp.mean_ci(group[metric], rng(seed, 'ci', str(key), metric), bootstraps)
            row.update({metric: mean, metric+'_lo': lo, metric+'_hi': hi,
                        metric+'_count': count, metric+'_status': status})
        if 'graphical' in group:
            row['p_graphical'] = group.graphical.mean()
            row['p_graphical_lo'], row['p_graphical_hi'] = gp.wilson_interval(int(group.graphical.sum()), len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def lines(ax, table, x, y, group='gamma', logx=False, logy=False):
    for label, part in table.groupby(group):
        part = part.sort_values(x)
        xx = part[x].to_numpy(float)
        yy = part[y].to_numpy(float)
        line, = ax.plot(xx, yy, 'o-', ms=4, label=f'{group}={label:g}')
        if y+'_lo' in part:
            ax.fill_between(xx, part[y+'_lo'].to_numpy(float), part[y+'_hi'].to_numpy(float),
                            color=line.get_color(), alpha=.15)
    ax.set(xlabel=x, ylabel=y)
    if logx: ax.set_xscale('log')
    if logy: ax.set_yscale('log')
    ax.legend(frameon=False)
    ax.grid(alpha=.2)


def savefig(fig, out, name):
    fig.tight_layout()
    fig.savefig(Path(out)/(name+'.png'), dpi=180, bbox_inches='tight')
    fig.savefig(Path(out)/(name+'.pdf'), bbox_inches='tight')
    plt.show()


def erasure_row(d, generator, max_stubs=5_000_000):
    S = int(d.sum())
    if S > max_stubs:
        raise MemoryError(f'{S} stubs exceeds explicit limit {max_stubs}; lower erasure sizes.')
    realised, loops, parallel = gp.erase_pairing(d, generator)
    Q = gp.eg_summary(d)['Q']
    D = int(S-realised.sum())
    assert Q <= D <= S
    return dict(S=S, Q=Q, D=D, loops=loops, parallel=parallel)


def efficiency(pairs):
    p = pairs.copy()
    denominator = p.S-p.Q
    p['eta'] = np.divide(p.S-p.D, denominator, out=np.full(len(p), np.nan), where=denominator.to_numpy()>0)
    p['retained'] = 1-p.Q/p.S
    p['erased_retained'] = 1-p.D/p.S
    p['gap'] = (p.D-p.Q)/p.S
    p['loop_loss'] = 2*p.loops/p.S
    p['parallel_loss'] = 2*p.parallel/p.S
    p['graphical'] = p.Q.eq(0)
    assert p.eta.dropna().between(-1e-12, 1+1e-12).all()
    return p


def capped_optimum(d, profile_ks=(), time_limit=30.):
    """Exact binary edge optimisation, including extrema across all optima."""
    d = np.asarray(d, dtype=np.int64)
    gp.eg_summary(d)
    if not np.all(d[:-1] >= d[1:]):
        raise ValueError('Pass sorted degrees to give ranks a fixed meaning.')
    n = len(d)
    u, v = np.triu_indices(n, 1)
    m = len(u)
    A = csc_matrix((np.ones(2*m), (np.r_[u, v], np.r_[np.arange(m), np.arange(m)])), shape=(n, m))
    bounds = Bounds(np.zeros(m), np.ones(m))
    integrality = np.ones(m)
    def solve(cost, matrix, low, high):
        fit = milp(cost, integrality=integrality, bounds=bounds,
                   constraints=LinearConstraint(matrix, low, high),
                   options={'time_limit': time_limit, 'mip_rel_gap': 0.})
        if fit.status != 0 or fit.x is None:
            raise RuntimeError(f'Exact optimum not certified: {fit.message}')
        x = np.rint(fit.x).astype(int)
        if not np.allclose(fit.x, x, atol=1e-5):
            raise RuntimeError('Nonintegral solver output.')
        realised = np.asarray(A@x).ravel().astype(int)
        assert np.all(realised <= d)
        return x, realised
    x, realised = solve(-np.ones(m), A, np.zeros(n), d)
    E = int(x.sum())
    Q = int(d.sum()-2*E)
    assert Q == gp.eg_summary(d)['Q'], 'MILP disagrees with the deficit repair identity.'
    fixed = vstack([A, csc_matrix(np.ones((1,m)))], format='csc')
    low, high = np.r_[np.zeros(n), E], np.r_[d, E]
    envelope = []
    for k in profile_ks:
        weights = np.asarray(A[:k].sum(axis=0)).ravel()
        _, min_degree = solve(weights, fixed, low, high)
        _, max_degree = solve(-weights, fixed, low, high)
        rmin = int(d[:k].sum()-max_degree[:k].sum())
        rmax = int(d[:k].sum()-min_degree[:k].sum())
        envelope.append(dict(k=int(k), repair_min=rmin, repair_max=rmax,
                             R_min=rmin/Q if Q else np.nan, R_max=rmax/Q if Q else np.nan))
    return dict(E_max=E, Q=Q, degrees=realised.tolist(), repair=(d-realised).tolist(), envelope=envelope)


def metadata(path, config, seed):
    Path(path).write_text(json.dumps(dict(version=VERSION, seed=seed, config=config,
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
        model='IID discrete power law, whole-target rejection on odd degree sum',
        repair='Q=2 ceil(F/2); cross-checked by independent capped MILP on small cases',
        uncertainty='Pointwise target-bootstrap means; Wilson graphicality intervals. Pairings averaged within targets.',
        cache='Per-sample parameters, seed, code schema and NumPy version; atomic per-sample JSON files.'), indent=2)+'\n')

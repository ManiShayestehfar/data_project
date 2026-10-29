import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import graphicality as gp


class MathematicalChecks(unittest.TestCase):
    def test_slacks_against_direct_formula(self):
        rng = np.random.default_rng(70)
        for n in [2, 7, 31, 100]:
            for _ in range(12):
                d = np.sort(rng.integers(0, n, size=n))[::-1]
                direct = [k*(k-1)+np.minimum(k, d[k:]).sum()-d[:k].sum()
                          for k in range(1, n+1)]
                np.testing.assert_array_equal(gp.erdos_gallai_deficit(d), direct)

    def test_repair_against_all_small_graphs(self):
        for n in range(2, 6):
            edges = list(itertools.combinations(range(n), 2))
            degree_vectors = set()
            for bits in range(1 << len(edges)):
                d = np.zeros(n, dtype=int)
                for j, (u, v) in enumerate(edges):
                    if bits >> j & 1:
                        d[u] += 1
                        d[v] += 1
                degree_vectors.add(tuple(np.sort(d)[::-1]))
            graphical = np.array(list(degree_vectors))
            for target in itertools.combinations_with_replacement(range(n), n):
                if sum(target) % 2:
                    continue
                target = np.array(target[::-1])
                exact = np.abs(graphical-target).sum(axis=1).min()
                self.assertEqual(gp.eg_summary(target)["Q"], exact)

    def test_capacity_and_erasure(self):
        for gamma in [.5, 1.3, 1.8, 2.5]:
            d = gp.generate_degrees(100, gamma, gp.rng_for(13, 0, 100, gamma))
            r = gp.eg_summary(d)
            if r["F"]:
                self.assertAlmostEqual(r["f"], r["h_star"]-r["internal"]-r["external"])
                self.assertLessEqual(r["k_star"], r["k_last"])
            for j in range(3):
                realised, loops, parallel = gp.erase_pairing(d, gp.rng_for(13, 2, 100, gamma, 0, j))
                D = r["S"]-realised.sum()
                self.assertEqual(D, 2*(loops+parallel))
                self.assertGreaterEqual(D, r["Q"])

    def test_quantile_bounds_and_parity(self):
        for n in [2, 3, 20, 101]:
            for gamma in [-.5, .5, 1.3, 2.5]:
                d = gp.quantile_degrees(n, gamma)
                self.assertEqual(d.sum() % 2, 0)
                self.assertTrue(np.all((d >= 1) & (d < n)))
                self.assertTrue(np.all(d[:-1] >= d[1:]))

    def test_zero_sample_uncertainty(self):
        mean, lo, hi, count, status = gp.mean_ci(np.zeros(20), np.random.default_rng(0), 100)
        self.assertEqual((mean, count), (0., 20))
        self.assertTrue(np.isnan(lo) and np.isnan(hi))
        self.assertIn("uninformative", status)
        self.assertGreater(gp.wilson_interval(0, 20)[1], 0)
        with self.assertRaises(ValueError):
            gp.eg_summary(np.array([1, 0]))

    def test_resumption_and_added_replications(self):
        cfg = gp.settings("quick")
        cfg.update(sizes=[20], gammas=[1.3], repeats=3, bootstraps=20,
                   erase_sizes=[20], erase_gammas=[1.3], erase_repeats=2, pairings=2)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            targets, _, _ = gp.target_experiment(cfg, 4, out)
            pairs, _, _, _ = gp.erasure_experiment(cfg, 4, out)
            saved = pairs.copy()
            gp.finish_run(cfg, 4, out)
            cfg["erase_repeats"] = 3
            pairs, _, _, _ = gp.erasure_experiment(cfg, 4, out)
            self.assertEqual((len(targets), len(pairs)), (3, 6))
            np.testing.assert_array_equal(pairs.iloc[:4].D, saved.D)
            cfg["pairings"] = 3
            pairs, _, _, _ = gp.erasure_experiment(cfg, 4, out)
            self.assertEqual(len(pairs), 9)
            with self.assertRaises(ValueError):
                gp.prepare_run(out, cfg, 99)


if __name__ == "__main__":
    unittest.main()

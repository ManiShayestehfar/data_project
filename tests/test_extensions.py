import itertools
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import graphicality as gp
import utils as u

class ExtensionChecks(unittest.TestCase):
    def test_bounds_against_exact_capped_graph(self):
        for gamma in [1.3, 1.8]:
            for rep in range(4):
                d = u.sample(12, gamma, u.rng(4, gamma, rep))
                measured = u.measure(d, gamma)
                exact = u.capped_optimum(d, [1, 6, 12])
                self.assertEqual(exact['E_max'], measured['E_max'])
                self.assertLessEqual(measured['E_lower'], exact['E_max'])
                self.assertGreaterEqual(measured['E_upper'], exact['E_max'])
                for row in exact['envelope']:
                    self.assertLessEqual(row['repair_min'], row['repair_max'])
                self.assertEqual(exact['envelope'][-1]['repair_min'], exact['Q'])
                self.assertEqual(exact['envelope'][-1]['repair_max'], exact['Q'])

    def test_profile_extrema_by_enumeration(self):
        d = np.array([3, 3, 0, 0])
        exact = u.capped_optimum(d, [1, 2, 4])
        degrees = []
        edges = list(itertools.combinations(range(4), 2))
        for bits in range(1 << len(edges)):
            realised = np.zeros(4, dtype=int)
            for j, (a, b) in enumerate(edges):
                if bits >> j & 1:
                    realised[a] += 1
                    realised[b] += 1
            if np.all(realised <= d) and realised.sum() == 2*exact['E_max']:
                degrees.append(realised)
        for row in exact['envelope']:
            losses = [int((d-realised)[:row['k']].sum()) for realised in degrees]
            self.assertEqual(row['repair_min'], min(losses))
            self.assertEqual(row['repair_max'], max(losses))

    def test_cutoff_and_conditioning(self):
        for alpha in [.4, .7, 1.]:
            d = u.sample(31, 1.5, u.rng(1, alpha), alpha)
            self.assertEqual(d.sum() % 2, 0)
            self.assertLessEqual(d.max(), u.law(31, 1.5, alpha)[0][-1])
        with self.assertRaises(ValueError):
            u.sample(3, 1.5, u.rng(1), .1)

    def test_efficiency_and_zero_denominator(self):
        pairs = pd.DataFrame([dict(S=100, Q=80, D=90, loops=1, parallel=44),
                              dict(S=2, Q=2, D=2, loops=1, parallel=0)])
        result = u.efficiency(pairs)
        self.assertEqual(result.eta.iloc[0], .5)
        self.assertTrue(np.isnan(result.eta.iloc[1]))
        with self.assertRaises(MemoryError):
            u.erasure_row(np.array([2, 2, 2, 2]), u.rng(2), max_stubs=2)

    def test_cache_and_seed_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'samples.sqlite'
            cache = u.Cache(path, 1)
            self.assertEqual(cache.get('x', [1], lambda: {'v': 5}), {'v': 5})
            self.assertEqual(cache.get('x', [1], lambda: 1/0), {'v': 5})
            other = u.Cache(path, 2)
            self.assertEqual(other.get('x', [1], lambda: {'v': 6}), {'v': 6})
        np.testing.assert_array_equal(u.rng(1, 'a').integers(100, size=8),
                                      u.rng(1, 'a').integers(100, size=8))

if __name__ == '__main__':
    unittest.main()

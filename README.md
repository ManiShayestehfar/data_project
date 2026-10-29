# Retained connectivity and degree repair

Open **DATA5441_connectivity.ipynb**. It contains the experiment design, loops, plots and interpretation prompts. The default quick mode runs every experiment. Set `MODE = 'full'` in the first code cell for the larger design, then Run All from the project directory.

```bash
python -m pip install -r requirements.txt
python run.py --mode quick
python run.py --mode full
python -m unittest discover -s tests -v
```

If local kernel sockets are unavailable, use `python run.py --mode quick --in-process`. This executes the same cells in order through IPython and saves a notebook with outputs. The CLI always writes `DATA5441_connectivity_<mode>_executed.ipynb`; it preserves the editable source notebook.

## Experiments

1. Retained-connectivity scaling: optimal retained edges, rescaled retained fraction, samplewise clique/capacity bounds and target-bootstrap log–log slope diagnostics.
2. Erasure efficiency: reanalysis of the supplied full pairings, followed by independent new targets. Targetwise retained-edge efficiency, additive gap, parallel-edge loss and graphicality-conditioned summaries.
3. Cutoff intervention: support truncated at floor(n**alpha), comparing exact graphicality with relative repair cost near alpha=1/gamma.
4. Repair location: small exact binary edge optimisations; minimum/maximum top-rank repair over all capped optima and an illustrative optimal repair.

`utils.py` contains reusable numerical tools. `graphicality.py` retains the previous numerical core; the new notebook does not call its old experiment pipeline. `baseline/` contains the original full `targets.csv`, `erasure_pairings.csv` and provenance manifest. Historical and new targets remain separate.

## Run sizes

| Experiment | Quick | Full |
| --- | --- | --- |
| Scaling | 5 sizes, 100–10,000; 3 exponents; 20 targets/cell | 5 sizes, 1,000–100,000; 3 exponents; 100 targets/cell |
| Erasure | 3 sizes; 4 exponents; 10 targets/cell; 3 pairings | 4 sizes; 4 exponents; 100 targets/cell; 5 pairings |
| Cutoff | 3 sizes; 2 exponents; 4 cutoffs/exponent; 15 targets/cell | 5 sizes; 2 exponents; 6 cutoffs/exponent; 100 targets/cell |
| Exact profiles | 2 selected examples, n=24 | 2 selected examples, n=40 |

Scaling only needs sorted degree vectors; it does not construct a large graph. Random erasure explicitly constructs stubs and stops if a target exceeds five million stubs. Exact optimisation checks solver certification and raises on failure or timeout. The selected profile examples condition on nongraphicality and are not a population sample.

## Reproducibility and interpretation

Outputs live in `results/<mode>/`: CSV tables, PNG/PDF figures, a per-sample JSON cache, parameters and package versions. Cache keys include sample parameters, seed, code schema and NumPy version. Increase replication counts to append samples; change the utility schema version if changing numerical semantics. Close the notebook before modifying a cache externally. Do not run concurrent writers to the same result directory.

All mean intervals are pointwise percentile bootstrap intervals across independent targets. Repeated pairings are averaged within a target first. Graphicality uses Wilson intervals. Constant or singleton samples have missing mean intervals, rather than misleading zero-width intervals. Undefined efficiency is excluded and counted. Monte Carlo intervals do not cover finite-size approximation error. Quick mode is exploratory.

The formula Q=2*ceil(F/2) remains an imported minimum-deficiency identity. Independent MILP and exhaustive small-graph tests check its implementation, not its general proof. Verify the exact source theorem before the final report. The new scaling derivation assumes a fixed exponent strictly between 1 and 2 with the original linear cutoff. Rescaled boundedness does not imply a limiting constant. The cutoff scale is established literature, and no novelty claim is made here.

The packaged quick notebook was executed in process because this host blocks Jupyter kernel sockets. Full mode is configured but has not been run in this update.

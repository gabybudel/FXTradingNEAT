# FXTradingNEAT

Neuroevolution (NEAT) trading strategies for currency markets, benchmarked
against classic FX factor strategies: carry (CAR), momentum (MOM), and the
volatility risk premium (VRP).

The project evolves small feed-forward neural networks with the
[NEAT algorithm](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
(via [neat-python](https://github.com/CodeReclaimers/neat-python)) that map
per-currency market features to a single "investment attractiveness" score.
Each month, currencies are cross-sectionally ranked by this score; the top 20%
are bought and the bottom 20% sold, forming a long–short portfolio whose
excess return (over covered interest parity, i.e. versus 1-month forwards) is
the genome's fitness.

This code originates from an MSc Financial Economics thesis (Erasmus
University Rotterdam, 2018). It is research code with a Spyder-style
interactive workflow — see [Limitations](#limitations) before running.

## Methodology

**Universe.** 16 currencies against USD: AUD, BRL, CAD, CHF, DKK, EUR, GBP,
JPY, KRW, MXN, NOK, NZD, SEK, SGD, TWD, ZAR. Monthly observations from
1997-01 to 2018-04; the last 60 months (5 years) are held out as the test set.

**Network inputs (15, min-max scaled on the training period only):**

| # | Feature |
|---|---------|
| 1–3 | Spot returns, lags 1–3 |
| 4–6 | Excess returns (spot vs. lagged forward), lags 1–3 |
| 7–10 | Realized volatility (1-year rolling, from daily returns), current + lags 1–3 |
| 11 | 1-year implied volatility (Bloomberg) |
| 12 | Volatility risk premium (realized − implied) |
| 13 | Forward premium |
| 14 | Bid–offer spread |
| 15 | NBER recession indicator |

**Output.** A single sigmoid activation per currency. For each month with at
least 3 currencies having complete data, currencies are sorted by activation;
the top `pf_size = 20%` are held long and the bottom 20% short, equally
weighted. Returns are computed with a one-month signal lag and account for
transaction costs through bid (EB) / offer (EO) quotes.

**Fitness.** Two variants are trained:

- `neat_trader_sharpe.py` — fitness is the annualized Sharpe ratio of the
  training-period portfolio returns.
- `neat_trader_ret.py` — fitness is the annualized mean return.

**Benchmarks.** Carry (sort on forward premium), momentum (sort on trailing
3-month return), and VRP (sort on volatility risk premium) portfolios are
constructed in `data.py` with the same 20%/20% long–short scheme.

## Repository layout

| File | Purpose |
|------|---------|
| `data.py` | Loads raw Datastream/Bloomberg data from `Processed/FX_data.xlsx`, computes returns, realized/implied volatility, VRP, benchmark strategies, and scaled NEAT input features. Run first; later scripts consume its results. |
| `neat_trader_sharpe.py` | Trains a NEAT network with Sharpe-ratio fitness from `Scaled/scaled_full.xlsx`; pickles the winning network to `winner_net.pckl`. Standalone. |
| `neat_trader_ret.py` | Same, with mean-return fitness; pickles the winning *genome* to `winner_net_ret.pckl`. Standalone. |
| `evaluate_ml.py` | Evaluates a trained network on train/test periods against the benchmarks and writes `Strategy_Returns.xlsx` plus LaTeX tables. Requires `data.py` globals in the session. |
| `evaluate_secondrun.py` | Variant of `evaluate_ml.py` for a second training run. Requires `data.py` globals in the session. |
| `plots.py` | Cumulative-return and volatility plots with NBER recession shading. Requires `Strategy_Returns.xlsx` (and session globals for the volatility plot). |
| `strategies.py` | Snippet: return/volatility/VRP computations (subset of `data.py`). Requires session globals. |
| `returns_table.py` | Snippet: LaTeX summary-table writer for two return panels. |
| `eval_ret_net.py` | Snippet: restore a NEAT checkpoint and rebuild the winner network. |
| `visualize.py` | Network/statistics plotting utilities from the neat-python examples. |
| `neat_example.py` | The canonical neat-python XOR example (sanity check). |
| `neat_example2.py` | Legacy copy of `neat_trader_sharpe.py` with hardcoded Windows paths; kept for provenance, superseded. |
| `config-feedforward` | NEAT hyperparameters: population 50, 15 inputs, 1 output, 1 initial hidden node, sigmoid activations, up to 300 generations. |
| `model.py` | Empty scaffold (imports only). |

## Data requirements

The input data are **not** included in this repository (licensed Datastream /
Bloomberg data). Two Excel workbooks are expected:

- `Processed/FX_data.xlsx` — sheets `SP_ER_monthly`, `SP_EB_monthly`,
  `SP_EO_monthly` (spot mid/bid/offer), `1F_ER_monthly`, `1F_EB_monthly`,
  `1F_EO_monthly` (1-month forwards), `SP_ER_daily`, `FW_ER_daily`,
  `ImpVol_Bloom` (1-year implied vols), and `Recession` (NBER indicator,
  column `RECESS`). Quotes are foreign-currency-per-USD and are inverted to
  USD-per-foreign-currency on load; inversion swaps the roles of bid and
  offer quotes.
- `Scaled/scaled_full.xlsx` — the scaled feature matrices produced from
  `data.py` (one sheet per feature, see `neat_trader_sharpe.py` for sheet
  names) plus `Ret_Ex_Long`, `Ret_Ex_Short`, and `NBER`.

`scaled.xlsx` in the repository root is a sample of the scaled-feature format.

## Running

```bash
pip install -r requirements.txt
# Graphviz binaries are also needed for network drawings:
#   macOS: brew install graphviz    Debian/Ubuntu: apt install graphviz

# Train (standalone, needs Scaled/scaled_full.xlsx):
python neat_trader_sharpe.py
python neat_trader_ret.py
```

Training prints per-generation statistics, checkpoints every 5 generations
(`neat-checkpoint-*`), and on completion draws the winning network and
fitness/speciation plots.

The evaluation and plotting scripts (`evaluate_*.py`, `plots.py`,
`strategies.py`) were written for an interactive IPython/Spyder session: they
reference variables created by running `data.py` in the same namespace
(`%run data.py`, then `%run evaluate_ml.py`). They will raise `NameError` if
executed as standalone scripts.

## Limitations

- **Interactive-session coupling.** Several scripts depend on `data.py`
  side effects rather than imports (see above). They are kept as-is to
  preserve the original thesis workflow.
- **No look-ahead controls beyond the signal lag.** Feature scaling uses
  training-period min/max only (correct), but implied-volatility dates are
  force-aligned to realized-volatility dates (`imp_vol.index =
  exp_vol_ph.index` in `data.py`), which assumes both series are complete and
  ordered identically.
- **Fitness evaluation is slow.** `eval_genomes` loops over dates and
  currencies in Python; a 50-genome population over ~190 months takes minutes
  per generation.
- **Single train/test split.** No cross-validation or walk-forward analysis;
  results are sensitive to the 2013–2018 test window.
- **Randomness is not seeded.** NEAT runs (and the tie-breaking permutation in
  `neat_trader_ret.py`) are not reproducible across runs.

## License

MIT — see [LICENSE](LICENSE).

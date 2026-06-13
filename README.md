# numbers

## Project Overview

This is a Python-based lottery number prediction system that uses machine learning to analyze historical lottery data and predict future numbers. The project supports three game types: Lotto, MultiMulti, and Szybkie600.

The model is a multi-label classifier: it outputs `P(number j appears in the next draw)` for every
number in the pool, trained on **leak-free** features (features for draw *i* are computed only from
draws before *i*). Evaluation reports honest baselines (uniform random, frequency) and an exact
binomial significance test — for a fair lottery, expect lift ≈ 1.0x and p-value > 0.05.

## Environment Setup

The project uses uv for Python package management:

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Core Commands

### Running the Application

The CLI is organised into subcommands. A global `--game` selects the game (default: `MultiMulti`).

> **Data is supplied manually.** Download the results JSON yourself and save it as `data/<Game>.json`
> (see [Data Requirements](#data-requirements)) before running `update`/`train`.

```bash
# Predict the next draw: top-n probabilities, grouped MC-dropout view, and bets
uv run src/main.py predict --game MultiMulti
uv run src/main.py predict --game MultiMulti --target 2,3,7,17 --approaches 200
uv run src/main.py predict --game MultiMulti --bets-count 10 --bets-size 4

# Append a hit-count histogram: how often the top-n picks hit 0,1,...,n actual numbers
# across every draw, next to the hypergeometric random baseline (static model, in-sample)
uv run src/main.py predict --game MultiMulti --histogram

# Load local JSON + rebuild features + train + predict
uv run src/main.py update --game MultiMulti

# Train with custom hyperparameters (defaults shown — calibrated on MultiMulti so the
# validation curve stays flat at the theoretical floor instead of rising from overfitting)
uv run src/main.py train --game MultiMulti --epochs 100 --lr 0.001 --hidden-dims 256,128 \
    --batch-size 32 --patience 10 --seed 42 --weight-decay 1e-4

# Train to the end: --patience 0 disables early stopping (all epochs run, final weights saved)
uv run src/main.py train --game MultiMulti --epochs 100 --patience 0

# --histogram works on train/update too (appended after the post-training prediction)
uv run src/main.py train --game Szybkie600 --epochs 70 --patience 0 --histogram

# Backtest vs random + frequency baselines (static = fast, in-sample sanity check)
uv run src/main.py evaluate --game MultiMulti --last-n 50
# Honest walk-forward backtest: retrains a fresh model per draw (slow)
uv run src/main.py evaluate --game MultiMulti --last-n 50 --retrain
```

Legacy flag-style invocation is still supported for backward compatibility:

```bash
uv run src/main.py --game MultiMulti --train
uv run src/main.py --game MultiMulti --update
uv run src/main.py --game MultiMulti --target 2,3,7,17
```

### Code Quality
```bash
uv run ruff format        # format
uv run ruff check         # lint
uv run pytest             # run the test suite
uv run pre-commit install # enable the ruff pre-commit hook (one-time)
```

## Architecture

### Core Components

- **`src/main.py`**: Entry point with subcommand-based CLI argument parsing
- **`src/processing/config.py`**: Game definitions, paths, constants, and logging setup
- **`src/processing/games.json`**: Game parameters (`n`/`k`/`prefix`/`bets`) — edit to tune games without code changes
- **`src/processing/game.py`**: Data loading, leak-free feature engineering, bet generation, orchestration
- **`src/processing/train.py`**: PyTorch multi-label classifier (BCE loss), seeded training with temporal
  train/val split, early stopping, metadata-carrying checkpoints
- **`src/processing/evaluate.py`**: Walk-forward backtesting with random + frequency baselines, binomial
  significance test, and probability calibration
- **`tests/`**: pytest suite incl. a no-leakage property test for the feature builder

### Data Flow

1. **Data Acquisition**: You manually download JSON data and save it to `data/<Game>.json`
2. **Database Storage**: Draws filtered by `gameType` (API dumps bundle side games like LottoPlus)
   and stored in DuckDB ordered by `draw_seq`
3. **Feature Engineering**: For every draw *i*, k-dim vectors (normalized frequency, recency, streak,
   rolling-window frequency + previous-draw aggregates) computed **only from draws before *i***
4. **Model Training**: MLP outputs k logits = per-number probability of appearing in the next draw
5. **Prediction**: Full probability ranking of all `k` numbers, a grouped view from `--approaches`
   Monte Carlo dropout passes (`x100` = numbers in the top-n of every pass), and deterministic bets
   walked through all `k` ranked numbers (consecutive chunks, then next-best rank combinations)

### Game Configuration

Each game type has specific parameters defined in `src/processing/games.json`:
- **n**: Number of balls drawn
- **k**: Total numbers in the pool
- **prefix**: Used for file naming
- **prizes** (optional): a list of plans, each `{size, stake, payouts}` — bet `size` numbers (the top-`size`
  predictions) at `stake`, `payouts[h]` won at `h` hits (length `size+1`). When present, `--histogram`
  adds one hit-histogram + theoretical profit/loss per plan. MultiMulti has two plans (bet-5 and bet-10);
  Szybkie600/Lotto one. Amounts come from each game's official prize table (base stake, without bonus
  options). Lotto's higher tiers are pari-mutuel (variable); its payouts use the regulamin's guaranteed
  minimums (3→24 zł fixed, 4→36 zł min, 6→2 000 000 zł min pool), so the result is a lower-bound estimate.
- **bets**: `{count, size}` — generate `count` deterministic bets of `size` numbers from the full
  k-number ranking in probability order: bet 1 = the `size` best numbers, bet 2 = the next `size`, ...
  until the pool is exhausted; further bets restart from the top with the next-best rank combinations
  (no repeated combination, no repeated number within a bet). If fewer than `count` unique
  combinations exist, all are returned.

To add a new game, add a member to `GameType` in `config.py` and a matching entry in `games.json`.

### File Structure

- `data/`: Contains JSON source data, DuckDB databases, trained models (.pth), and training plots (.png)
- `src/processing/`: Core business logic modules
- Configuration files: `pyproject.toml`, `ruff.toml`, `pyrightconfig.json`

## Data Requirements

Data is **not** fetched by the app — download it yourself and save it as `./data/{GameType}.json`
(matching the game `prefix`, e.g. `data/MultiMulti.json`). Open the draw-results API in a browser and
save the JSON response:
- API endpoint: `https://www.lotto.pl/api/lotteries/draw-results/by-gametype?game=MultiMulti&index=1&size=10000&sort=drawDate&order=DESC`
- The expected shape is `{"totalRows": N, "items": [ { "results": [ { "drawSystemId", "gameType", "resultsJson", "specialResults" } ] } ], ... }`.
- For games with more than `size` draws, fetch successive `index` pages and merge their `items` arrays into one file.
- Make sure the `game=` query parameter matches the file name — results whose `gameType` does not match
  the requested game are ignored, and a dump containing only other games is rejected with a clear error.

## Environment Variables

- `DATA_DIR`: Directory for data files (default: "data/")

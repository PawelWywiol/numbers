# numbers

## Project Overview

This is a Python-based lottery number prediction system that uses machine learning to analyze historical lottery data and predict future numbers. The project supports three game types: Lotto, MultiMulti, and Szybkie600.

## Environment Setup

The project uses uv for Python package management:

```bash
uv env
source .venv/bin/activate
uv sync
```

## Core Commands

### Running the Application

The CLI is organised into subcommands. A global `--game` selects the game (default: `MultiMulti`).

> **Data is supplied manually.** Download the results JSON yourself and save it as `data/<Game>.json`
> (see [Data Requirements](#data-requirements)) before running `update`/`train`.

```bash
# Predict the next draw (default command when none given)
uv run src/main.py predict --game MultiMulti
uv run src/main.py predict --game MultiMulti --target 2,3,7,17 --approaches 200

# Load local JSON + preprocess + train + predict
uv run src/main.py update --game MultiMulti

# Train with custom hyperparameters
uv run src/main.py train --game MultiMulti --epochs 100 --lr 0.001 --hidden-dim 128 --batch-size 32

# Backtest prediction quality vs a random baseline
uv run src/main.py evaluate --game MultiMulti --last-n 50        # add --retrain for walk-forward
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
pre-commit install        # enable the ruff pre-commit hook (one-time)
```

## Architecture

### Core Components

- **`src/main.py`**: Entry point with subcommand-based CLI argument parsing
- **`src/processing/config.py`**: Game definitions, paths, constants, and logging setup
- **`src/processing/games.json`**: Game parameters (`n`/`k`/`prefix`) — edit to tune games without code changes
- **`src/processing/game.py`**: Data loading, feature engineering, train/predict orchestration
- **`src/processing/train.py`**: PyTorch MLP model, training, and prediction
- **`src/processing/evaluate.py`**: Backtesting against historical draws with a random baseline
- **`tests/`**: pytest suite covering config, prediction post-processing, preprocessing, and eval math

### Data Flow

1. **Data Acquisition**: You manually download JSON data and save it to `data/<Game>.json`
2. **Database Storage**: Stores results in DuckDB database files
3. **Preprocessing**: Calculates distribution, step patterns, and repeat frequencies
4. **Model Training**: Uses PyTorch MLP to learn patterns from preprocessed data
5. **Prediction**: Generates multiple predictions and ranks numbers by frequency

### Game Configuration

Each game type has specific parameters defined in `src/processing/games.json`:
- **n**: Number of balls drawn
- **k**: Total numbers in the pool
- **prefix**: Used for file naming

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
- The expected shape is `{"totalRows": N, "items": [ { "results": [ { "drawSystemId", "resultsJson", "specialResults" } ] } ], ... }`.
- For games with more than `size` draws, fetch successive `index` pages and merge their `items` arrays into one file.

## Environment Variables

- `DATA_DIR`: Directory for data files (default: "data/")

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
```bash
# Basic prediction for MultiMulti (default)
uv run src/main.py

# Specify game type
uv run src/main.py --game Lotto
uv run src/main.py --game MultiMulti
uv run src/main.py --game Szybkie600

# Update results from API and retrain
uv run src/main.py --game MultiMulti --update

# Train the model
uv run src/main.py --game MultiMulti --train

# Predict with target comparison
uv run src/main.py --game MultiMulti --target 2,3,7,17,18,20,21,27,29,30,31,33,36,38,44,49,51,64,70,74 --train
```

### Code Quality
```bash
# Format and lint code
ruff format
ruff check
```

## Architecture

### Core Components

- **`src/main.py`**: Entry point with CLI argument parsing
- **`src/processing/game.py`**: Game logic, data processing, and API integration
- **`src/processing/train.py`**: Machine learning model training and prediction

### Data Flow

1. **Data Acquisition**: Downloads JSON data from lotto.pl API
2. **Database Storage**: Stores results in DuckDB database files
3. **Preprocessing**: Calculates distribution, step patterns, and repeat frequencies
4. **Model Training**: Uses PyTorch MLP to learn patterns from preprocessed data
5. **Prediction**: Generates multiple predictions and ranks numbers by frequency

### Game Configuration

Each game type has specific parameters defined in `GAMES_CONFIG`:
- **n**: Number of balls drawn
- **k**: Total numbers in the pool
- **prefix**: Used for file naming

### File Structure

- `data/`: Contains JSON source data, DuckDB databases, trained models (.pth), and training plots (.png)
- `src/processing/`: Core business logic modules
- Configuration files: `pyproject.toml`, `ruff.toml`, `pyrightconfig.json`

## Data Requirements

Download lottery results from lotto.pl and save as JSON files in `./data/` directory:
- Example API endpoint: `https://www.lotto.pl/api/lotteries/draw-results/by-gametype?game=MultiMulti&index=0&size=100000&sort=drawDate&order=DESC`
- Save as `./data/{GameType}.json`

## Environment Variables

- `DATA_DIR`: Directory for data files (default: "data/")

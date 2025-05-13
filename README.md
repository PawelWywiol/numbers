# numbers

It's just a simple number guessing game. Made for fun.

Doesn't work at all. Do not try to fool yourself 🙃

## Installation

```bash
uv env
source .venv/bin/activate
uv sync
```

## Environment verification

```bash
which python
uv pip list
```

## Source data preparation

Available game types: `Lotto | MultiMulti | Szybkie600`

Download results from [lotto.pl](https://www.lotto.pl/) and save them in the `./data` folder as e.g. `./data/MultiMulti.json`.

Sample data:

[https://www.lotto.pl/api/lotteries/draw-results/by-gametype?game=MultiMulti&index=0&size=100000&sort=drawDate&order=DESC](https://www.lotto.pl/api/lotteries/draw-results/by-gametype?game=MultiMulti&index=0&size=100000&sort=drawDate&order=DESC)

## Usage

```bash
uv run src/main.py --help
usage: main.py [-h] [--game GAME] [--update] [--train] [--target TARGET]

Predict game results.

optional arguments:
  -h, --help       show this help message and exit
  --game GAME      Game type (Lotto, MultiMulti, Szybkie600).
  --update         Update game results.
  --train          Train the model.
  --target TARGET  Target date for prediction (numbers separated by commas).
```

```bash
uv run src/main.py --game MultiMulti --target 2,3,7,17,18,20,21,27,29,30,31,33,36,38,44,49,51,64,70,74 --train
```

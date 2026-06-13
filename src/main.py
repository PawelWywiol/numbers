"""CLI entry point: update, train, predict and evaluate lottery game models.

Draw-results JSON is supplied manually — download it yourself and save it as ``data/<Game>.json``
(see README). The CLI never fetches data.
"""

from __future__ import annotations

import argparse

from processing import game
from processing.config import DEFAULT_APPROACHES, GameType, configure_logging
from processing.evaluate import evaluate_game


def _add_game_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--game", type=str, default="MultiMulti", help="Game type (Lotto, MultiMulti, Szybkie600).")


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dims", type=str, default="256,128", help="Comma-separated layer sizes.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience in epochs; 0 disables it (train all epochs, save final weights).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="L2 regularization for Adam (default: 1e-4, calibrated so val loss stays flat).",
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Append a hit-count distribution (top-n vs every actual draw) after predicting.",
    )


def _add_bet_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bets-count", type=int, help="Override the number of bets from games.json.")
    parser.add_argument("--bets-size", type=int, help="Override the bet size from games.json.")


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with subcommands and backward-compatible legacy flags."""
    parser = argparse.ArgumentParser(description="Predict lottery game results.")

    # Legacy flags (pre-subcommand CLI) kept for backward compatibility.
    _add_game_arg(parser)
    parser.add_argument("--update", action="store_true", help="Legacy: load JSON + preprocess + train.")
    parser.add_argument("--train", action="store_true", help="Legacy: train then predict.")
    parser.add_argument("--target", type=str, help="Legacy: comma-separated numbers to score hits against.")

    sub = parser.add_subparsers(dest="command")

    p_update = sub.add_parser("update", help="Load JSON, rebuild features, then train and predict.")
    _add_game_arg(p_update)
    _add_train_args(p_update)

    p_train = sub.add_parser("train", help="Train the model, then predict.")
    _add_game_arg(p_train)
    _add_train_args(p_train)

    p_predict = sub.add_parser("predict", help="Predict the next draw and generate bets.")
    _add_game_arg(p_predict)
    p_predict.add_argument("--target", type=str, help="Comma-separated numbers to score hits against.")
    p_predict.add_argument("--seed", type=int, default=42, help="Seed for MC-dropout passes.")
    p_predict.add_argument(
        "--approaches",
        type=int,
        default=DEFAULT_APPROACHES,
        help="MC-dropout forward passes for the grouped prediction view.",
    )
    p_predict.add_argument(
        "--histogram",
        action="store_true",
        help="Append a hit-count distribution (top-n vs every actual draw) below the bets.",
    )
    _add_bet_args(p_predict)

    p_eval = sub.add_parser("evaluate", help="Backtest prediction quality vs random and frequency baselines.")
    _add_game_arg(p_eval)
    p_eval.add_argument("--last-n", type=int, default=20)
    p_eval.add_argument("--retrain", action="store_true", help="True walk-forward: retrain per draw (slow, honest).")
    p_eval.add_argument("--seed", type=int, default=42)

    return parser


def _split_target(target: str | None) -> list[str] | None:
    return target.split(",") if target else None


def _parse_hidden_dims(value: str) -> list[int]:
    return [int(dim) for dim in value.split(",") if dim.strip()]


def _train(game_type: GameType, args: argparse.Namespace) -> None:
    game.train_game_results(
        game_type,
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        patience=args.patience,
        weight_decay=args.weight_decay,
    )


def main() -> None:
    """Parse arguments and dispatch to the requested command."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    try:
        game_type = GameType.from_str(args.game)
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "update":
        game.resolve_results(game_type)
        game.preprocess_results(game_type)
        _train(game_type, args)
        game.predict_game_results(game_type, histogram=args.histogram)
        return

    if args.command == "train":
        _train(game_type, args)
        game.predict_game_results(game_type, histogram=args.histogram)
        return

    if args.command == "predict":
        game.predict_game_results(
            game_type,
            _split_target(args.target),
            bets_count=args.bets_count,
            bets_size=args.bets_size,
            approaches=args.approaches,
            seed=args.seed,
            histogram=args.histogram,
        )
        return

    if args.command == "evaluate":
        evaluate_game(game_type, last_n=args.last_n, retrain=args.retrain, seed=args.seed)
        return

    # Legacy / default path (no subcommand).
    if args.update:
        game.resolve_results(game_type)
        game.preprocess_results(game_type)
    if args.train or args.update:
        game.train_game_results(game_type)
    game.predict_game_results(game_type, _split_target(args.target))


if __name__ == "__main__":
    main()

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


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with subcommands and backward-compatible legacy flags."""
    parser = argparse.ArgumentParser(description="Predict lottery game results.")

    # Legacy flags (pre-subcommand CLI) kept for backward compatibility.
    _add_game_arg(parser)
    parser.add_argument("--update", action="store_true", help="Legacy: load JSON + preprocess + train.")
    parser.add_argument("--train", action="store_true", help="Legacy: train then predict.")
    parser.add_argument("--target", type=str, help="Legacy: comma-separated numbers to score hits against.")

    sub = parser.add_subparsers(dest="command")

    p_update = sub.add_parser("update", help="Load JSON, preprocess, then train and predict.")
    _add_game_arg(p_update)

    p_train = sub.add_parser("train", help="Train the model, then predict.")
    _add_game_arg(p_train)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--lr", type=float, default=0.001)
    p_train.add_argument("--hidden-dim", type=int, default=128)
    p_train.add_argument("--batch-size", type=int, default=32)

    p_predict = sub.add_parser("predict", help="Predict the next draw.")
    _add_game_arg(p_predict)
    p_predict.add_argument("--target", type=str, help="Comma-separated numbers to score hits against.")
    p_predict.add_argument("--approaches", type=int, default=DEFAULT_APPROACHES)

    p_eval = sub.add_parser("evaluate", help="Backtest prediction quality vs a random baseline.")
    _add_game_arg(p_eval)
    p_eval.add_argument("--last-n", type=int, default=20)
    p_eval.add_argument("--retrain", action="store_true", help="True walk-forward: retrain per draw (slow).")

    return parser


def _split_target(target: str | None) -> list[str] | None:
    return target.split(",") if target else None


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
        game.train_game_results(game_type)
        game.predict_game_results(game_type)
        return

    if args.command == "train":
        game.train_game_results(
            game_type,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        game.predict_game_results(game_type)
        return

    if args.command == "predict":
        game.predict_game_results(game_type, _split_target(args.target), approaches=args.approaches)
        return

    if args.command == "evaluate":
        evaluate_game(game_type, last_n=args.last_n, retrain=args.retrain)
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

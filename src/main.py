import argparse

from processing import game


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict game results.")
    parser.add_argument("--game", type=str, help="Game type (MultiMulti, Szybkie600).")
    parser.add_argument("--train", action="store_true", help="Train the model.")

    args = parser.parse_args()

    game_type = game.GameType.MultiMulti
    train = False

    if args.game:
        try:
            game_type = game.GameType.from_str(args.game)
        except ValueError as _:
            game_type = game.GameType.MultiMulti

    if args.train:
        train = True

    if train:
        game.resolve_results(game_type)
        game.preprocess_results(game_type)
        game.train_game_results(game_type)

    game.predict_game_results(game_type)


if __name__ == "__main__":
    main()

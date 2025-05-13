import argparse

from processing import game


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict game results.")
    parser.add_argument("--game", type=str, help="Game type (Lotto, MultiMulti, Szybkie600).")
    parser.add_argument("--update", action="store_true", help="Update game results.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--target", type=str, help="Target date for prediction (numbers separated by commas).")

    args = parser.parse_args()

    game_type = game.GameType.MultiMulti
    update = False
    train = False

    if args.game:
        try:
            game_type = game.GameType.from_str(args.game)
        except ValueError as _:
            game_type = game.GameType.MultiMulti

    if args.update:
        update = True

    if args.train:
        train = True

    if update:
        game.resolve_results(game_type)
        game.preprocess_results(game_type)

    if train or update:
        game.train_game_results(game_type)

    target = ""
    if args.target:
        target = args.target.split(",")

    game.predict_game_results(game_type, target)


if __name__ == "__main__":
    main()

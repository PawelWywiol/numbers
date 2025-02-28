from processing import game


def main() -> None:
    game_type = game.GameType.MultiMulti
    game.resolve_results(game_type)
    game.preprocess_results(game_type)
    game.train_game_results(game_type)
    game.predict_game_results(game_type)


if __name__ == "__main__":
    main()

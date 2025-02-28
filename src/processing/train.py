import json

import duckdb
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def load_data(db_file: str) -> pd.DataFrame:
    conn = duckdb.connect(db_file)
    query = """
        SELECT distribution, step, repeats, draw_numbers FROM results ORDER BY draw_id ASC
    """
    results = conn.execute(query).df()
    conn.close()

    for col in ["distribution", "step", "repeats", "draw_numbers"]:
        results[col] = results[col].apply(json.loads)

    return results


def preprocess_data(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [dist + step + rep for dist, step, rep in zip(df["distribution"], df["step"], df["repeats"])],
        dtype=torch.float32,
    )
    y = torch.tensor(df["draw_numbers"].tolist(), dtype=torch.float32)
    return x, y


def predict_next_draw(model: nn.Module, last_data: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        prediction = model(last_data.unsqueeze(0))
    return prediction.squeeze(0)


def process_predictions(predictions: torch.Tensor, min_val: int = 1, max_val: int = 80, k: int = 20) -> list[int]:
    """Process raw model predictions into valid lottery numbers."""
    rounded_predictions = torch.round(predictions).int().tolist()
    if isinstance(rounded_predictions, list):
        rounded_predictions = torch.tensor(rounded_predictions)

    unique_numbers = list(set(min(max(int(n.item()), min_val), max_val) for n in rounded_predictions.flatten()))  # noqa: C401

    while len(unique_numbers) < k:
        num = torch.randint(min_val, max_val + 1, (1,)).item()
        if num not in unique_numbers:
            unique_numbers.append(num)
    return sorted(unique_numbers[:k])


def train_results(  # noqa: PLR0913
    db_file: str,
    model_file: str,
    hidden_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> None:
    results = load_data(db_file)
    x, y = preprocess_data(results)

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim, output_dim = x.shape[1], y.shape[1]
    model = MLP(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")  # noqa: T201

    torch.save(model.state_dict(), model_file)
    print(f"Model trained and saved as '{model_file}'")  # noqa: T201


def predict_results(
    db_file: str,
    model_file: str,
    approaches: int = 1,
    distribution_size: int = 80,
    hidden_dim: int = 128,
) -> dict:
    results = load_data(db_file)
    x, y = preprocess_data(results)

    input_dim, output_dim = x.shape[1], y.shape[1]
    model = MLP(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_file))

    distribution = [0] * distribution_size

    for _ in range(approaches):
        last_x = x[-1].unsqueeze(0)
        next_draw = predict_next_draw(model, last_x)
        predictions = process_predictions(next_draw)

        for num in predictions:
            distribution[num - 1] += 1

        distribution_min = min(distribution)

        if distribution_min > 0:
            break

    distribution_pairs = [(i + 1, distribution[i]) for i in range(distribution_size)]
    distribution_pairs = [pair for pair in distribution_pairs if pair[1] > 0]
    sorted_distribution = sorted(distribution_pairs, key=lambda x: x[1], reverse=True)

    grouped_distribution = {}
    for num, count in sorted_distribution:
        if count not in grouped_distribution:
            grouped_distribution[count] = []
        grouped_distribution[count].append(num)

    return grouped_distribution

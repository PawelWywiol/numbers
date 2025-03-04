import json

import duckdb
import matplotlib.pyplot as plt
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


def process_predictions(predictions: torch.Tensor, n: int = 20, k: int = 80, min_val: int = 1) -> list[int]:
    """Process raw model predictions into valid lottery numbers."""
    rounded_predictions = torch.round(predictions).int().tolist()
    if isinstance(rounded_predictions, list):
        rounded_predictions = torch.tensor(rounded_predictions)

    unique_numbers = list(set(min(max(int(t.item()), min_val), k) for t in rounded_predictions.flatten()))  # noqa: C401

    while len(unique_numbers) < n:
        num = torch.randint(min_val, k + 1, (1,)).item()
        if num not in unique_numbers:
            unique_numbers.append(num)
    return sorted(unique_numbers[:n])


def train_results(  # noqa: PLR0913
    db_file: str,
    model_file: str,
    plot_file: str,
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

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg_train_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(avg_train_loss)

        # Walidacja
        model.eval()
        with torch.no_grad():
            val_outputs = model(x)
            val_loss = criterion(val_outputs, y).item()
            val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")  # noqa: T201

    torch.save(model.state_dict(), model_file)
    print(f"Model trained and saved as '{model_file}'")  # noqa: T201

    # Wykres strat
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train vs Validation Loss")
    plt.savefig(plot_file)
    plt.close()

    min_train_loss = 0.01

    # Analiza i sugestie
    if train_losses[-1] < min_train_loss:
        print("🔹 Model może być przeuczony. Spróbuj zmniejszyć 'hidden_dim' lub liczbę epok.")  # noqa: T201
    if train_losses[-1] > val_losses[-1]:
        print("🔹 Możliwe przeuczenie. Spróbuj zmniejszyć 'epochs' lub 'hidden_dim'.")  # noqa: T201
    if train_losses[-1] > 10 * val_losses[-1]:
        print("🔹 Możliwe przeuczenie, duża różnica strat.")  # noqa: T201
    if val_losses[-1] > train_losses[-1]:
        print("🔹 Możliwe niedouczenie. Spróbuj zwiększyć 'epochs' lub zmniejszyć 'learning_rate'.")  # noqa: T201
    if val_losses[-1] > train_losses[-1] * 2:
        print("🔹 Model może się niedouczać. Spróbuj zmniejszyć 'batch_size' lub zwiększyć 'hidden_dim'.")  # noqa: T201


def predict_results(  # noqa: PLR0913
    db_file: str,
    model_file: str,
    approaches: int,
    n: int,
    k: int,
    hidden_dim: int = 128,
) -> dict:
    results = load_data(db_file)
    x, y = preprocess_data(results)

    input_dim, output_dim = x.shape[1], y.shape[1]
    model = MLP(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_file))

    distribution = [0] * k

    for _ in range(approaches):
        last_x = x[-1].unsqueeze(0)
        next_draw = predict_next_draw(model, last_x)
        predictions = process_predictions(next_draw, n, k)

        for num in predictions:
            distribution[num - 1] += 1

    distribution_pairs = [(i + 1, distribution[i]) for i in range(k)]
    distribution_pairs = [pair for pair in distribution_pairs if pair[1] > 0]
    sorted_distribution = sorted(distribution_pairs, key=lambda x: x[1], reverse=True)

    grouped_distribution = {}
    for num, count in sorted_distribution:
        if count not in grouped_distribution:
            grouped_distribution[count] = []
        grouped_distribution[count].append(num)

    return grouped_distribution

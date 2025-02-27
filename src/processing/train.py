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

    results["distribution"] = results["distribution"].apply(lambda x: json.loads(x))
    results["step"] = results["step"].apply(lambda x: json.loads(x))
    results["repeats"] = results["repeats"].apply(lambda x: json.loads(x))
    results["draw_numbers"] = results["draw_numbers"].apply(lambda x: json.loads(x))

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
    """Process raw model predictions into valid lottery numbers.

    Args:
        predictions: Raw model predictions
        min_val: Minimum allowed number (inclusive)
        max_val: Maximum allowed number (inclusive)
        k: Number of numbers to select

    Returns:
        List of k unique numbers between min_val and max_val
    """
    # Create probability distribution for numbers 1-80
    probs = torch.zeros(max_val)
    for i in range(len(predictions)):
        value = max(min(int(predictions[i].item() * max_val), max_val - 1), 0)
        probs[value] += 1.0

    # Normalize probabilities
    probs = torch.softmax(probs, dim=0)

    # Get top k unique values
    _, indices = torch.topk(probs, k=min(k, max_val))
    numbers = [int(idx.item()) + 1 for idx in indices]  # +1 because indices are 0-based

    # Ensure we have exactly k numbers (shouldn't be needed with topk)
    while len(numbers) < k:
        num = torch.randint(min_val, max_val + 1, (1,)).item()
        if num not in numbers:
            numbers.append(num)

    return sorted(numbers)


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

    input_dim = x.shape[1]
    output_dim = y.shape[1]

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
    print("Model trained and saved as '{model_file}'")  # noqa: T201

    # Get the last data point for prediction
    last_x = x[-1].unsqueeze(0)
    next_draw = predict_next_draw(model, last_x)

    # Process predictions to get valid lottery numbers
    predicted_numbers = process_predictions(next_draw)
    print(f"Predicted numbers for next draw: {predicted_numbers}")  # noqa: T201

    print("Training completed")  # noqa: T201

"""LAB 2 — Exercise: forecast a real time series with a recurrent network.

Dataset: daily minimum temperatures in Melbourne, Australia, 1981-1990 (3650 days).
Task:    given the temperatures of the last `window` days, predict tomorrow's minimum temperature.

Everything is ready: download, normalization, sliding windows, DataLoaders, training loop,
evaluation and plots. Your only job is to complete the TODOs in the `Forecaster` class below,
using the building blocks shown in rnn_tools.ipynb (RNN, LSTM, GRU, Linear, ...).

Run it with:
    python rnn_timeseries.py                         # default: window 30, 20 epochs
    python rnn_timeseries.py --window 60 --epochs 30

Goal: beat the "persistence" baseline (tomorrow = today), which the script prints for comparison.
"""
import argparse
import csv
import os
import time
import urllib.request

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"


# =============================================================================
# TODO: build your network
# =============================================================================
class Forecaster(nn.Module):
    """Recurrent network for one-step-ahead forecasting.

    Input:  a batch of windows of shape (batch, window, 1)
            (the normalized temperatures of the last `window` days)
    Output: a tensor of shape (batch, 1) with the (normalized) prediction for the next day.
    """

    def __init__(self):
        super().__init__()
        # TODO: define your layers here.
        #
        # Suggested structure (but feel free to experiment!):
        #   - a recurrent layer: nn.LSTM, nn.GRU or nn.RNN with
        #         input_size=1 (one value per day), a hidden_size of your choice (e.g. 32 or 64),
        #         and batch_first=True  <-- don't forget it!
        #   - a nn.Linear layer that maps the last hidden state to 1 output

        #self.rnn = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.rnn = nn.GRU(input_size=1, hidden_size=128, batch_first=True)
        self.linear = nn.Linear(128, 1)


    def forward(self, x):
        # TODO: pass x through the recurrent layer, take the output of the LAST time step
        # and pass it through the linear layer. Return a tensor of shape (batch, 1).
        # Remember that nn.LSTM returns (out, (h_n, c_n)), while nn.GRU / nn.RNN return (out, h_n).

        out, h_n = self.rnn(x)
        last_hidden_state = h_n[-1]  # Get the last hidden state from the LSTM output
        output = self.linear(last_hidden_state)  # Pass the last hidden state through the linear layer
        return output



# =============================================================================
# From here on everything is ready: you don't need to modify anything
# (but read it, it's the same recipe as rnn_tools.ipynb §5-6)
# =============================================================================
def load_series():
    """Download (the first time) and read the dataset. Returns a 1D tensor of temperatures."""
    path = os.path.join(DATA_DIR, "daily-min-temperatures.csv")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Downloading {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)                                   # skip the header "Date","Temp"
        values = [float(row[1].strip('"?')) for row in reader if row]
    return torch.tensor(values, dtype=torch.float32)


class SlidingWindowDataset(Dataset):
    """Input: `window` consecutive values. Target: the value right after."""

    def __init__(self, series, window):
        self.series = series
        self.window = window

    def __len__(self):
        return len(self.series) - self.window

    def __getitem__(self, i):
        x = self.series[i : i + self.window].unsqueeze(-1)      # (window, 1)
        y = self.series[i + self.window].unsqueeze(-1)          # (1,)
        return x, y


def get_data(window, batch_size, test_fraction=0.2):
    series = load_series()

    # Chronological split: the last 20% of the days (~2 years) is the test set
    split = int(len(series) * (1 - test_fraction))
    train_raw, test_raw = series[:split], series[split:]

    # Normalize with the statistics of the training set only
    mean, std = train_raw.mean(), train_raw.std()
    train_series = (train_raw - mean) / std
    test_series = (test_raw - mean) / std

    train_loader = DataLoader(SlidingWindowDataset(train_series, window), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SlidingWindowDataset(test_series, window), batch_size=512, shuffle=False)

    print(f"Series: {len(series)} days, train {len(train_raw)}, test {len(test_raw)}")
    print(f"Train mean {mean:.2f}°C, std {std:.2f}°C")
    print(f"Samples: train {len(train_loader.dataset)}, test {len(test_loader.dataset)}  (window = {window} days)\n")
    return series, split, train_loader, test_loader, mean, std


def check_model(model, window, device):
    xb = torch.zeros(4, window, 1, device=device)
    model.eval()
    with torch.no_grad():
        out = model(xb)
    if out.shape != (4, 1):
        raise ValueError(f"The network must return a tensor of shape (batch, 1), got {tuple(out.shape)}")
    print(f"Output shape OK: {tuple(xb.shape)} -> {tuple(out.shape)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # keeps RNN training stable
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu())
            targets.append(yb)
    return torch.cat(preds).squeeze(1), torch.cat(targets).squeeze(1)


def rmse_celsius(preds, targets, std):
    """RMSE converted back to degrees Celsius."""
    return (torch.sqrt(((preds - targets) ** 2).mean()) * std).item()


def persistence_baseline(loader, std):
    """Predict that tomorrow will be like today (the last value of the window)."""
    preds, targets = [], []
    for xb, yb in loader:
        preds.append(xb[:, -1, 0])
        targets.append(yb[:, 0])
    return rmse_celsius(torch.cat(preds), torch.cat(targets), std)


def plot_results(history, preds, targets, mean, std, baseline):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 2]})
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["test_rmse"], "o-", label="your model")
    ax1.axhline(baseline, color="tab:red", ls="--", label="persistence baseline")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("test RMSE (°C)")
    ax1.legend()
    ax1.set_title("Test error")

    n = 365
    ax2.plot((targets[:n] * std + mean), label="true", lw=1)
    ax2.plot((preds[:n] * std + mean), label="prediction", lw=1)
    ax2.set_xlabel("day of the test set")
    ax2.set_ylabel("min temperature (°C)")
    ax2.legend()
    ax2.set_title("First year of the test set")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=30, help="number of past days given as input")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_plot", action="store_true", help="skip the final plots")
    args = parser.parse_args()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}\n")

    series, split, train_loader, test_loader, mean, std = get_data(args.window, args.batch_size)

    if not args.no_plot:
        plt.figure(figsize=(12, 3))
        plt.plot(range(split), series[:split], lw=0.7, label="train")
        plt.plot(range(split, len(series)), series[split:], lw=0.7, label="test")
        plt.ylabel("min temperature (°C)")
        plt.xlabel("day")
        plt.title("Daily minimum temperatures, Melbourne 1981-1990")
        plt.legend()
        plt.show()

    baseline = persistence_baseline(test_loader, std)
    print(f"Persistence baseline (tomorrow = today): test RMSE {baseline:.3f}°C  <- try to beat this!\n")

    model = Forecaster().to(device)
    print(model, "\n")
    check_model(model, args.window, device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "test_rmse": []}
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        preds, targets = predict(model, test_loader, device)
        test_rmse = rmse_celsius(preds, targets, std)
        history["train_loss"].append(train_loss)
        history["test_rmse"].append(test_rmse)
        print(f"Epoch [{epoch + 1}/{args.epochs}]  train MSE: {train_loss:.4f}  "
              f"test RMSE: {test_rmse:.3f}°C  ({time.time() - start:.1f}s)")

    print(f"\nFinal test RMSE: {history['test_rmse'][-1]:.3f}°C   (persistence baseline: {baseline:.3f}°C)")

    save_path = os.path.join(HERE, "forecaster.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    if not args.no_plot:
        plot_results(history, preds, targets, mean, std, baseline)


if __name__ == "__main__":
    main()

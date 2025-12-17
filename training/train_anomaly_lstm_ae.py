import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

CSV_PATH = "outputs/timeseries_0p5s.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = ["flow", "density", "avg_speed", "speed_missing"]
SEQ_LEN = 20 # 20 langkah * 0.5s = 10 detik
BATCH_SIZE = 32
EPOCHS = 60
LR = 1e-3

HIDDEN = 64
LATENT = 16
NUM_LAYERS = 1
DROPOUT = 0.0

THRESH_PCT = 95 # p95 reconstruction error

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)

class SeqDataset(Dataset):
    def __init__(self, X_seq):
        self.X = torch.tensor(X_seq, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


class LSTMAutoEncoder(nn.Module):
    """
    Reconstruct input sequence.
    Encoder: LSTM -> latent vector (last hidden)
    Decoder: repeat latent -> LSTM -> per-timestep reconstruction
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        enc_out, (h_n, c_n) = self.encoder(x)
        h_last = h_n[-1]
        z = self.to_latent(h_last)

        # decode
        h0 = self.from_latent(z)
        dec_in = h0.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_in)
        x_hat = self.out(dec_out)
        return x_hat


def make_sequences(X, seq_len):
    # X: (N, F) -> (N-seq_len+1, seq_len, F)
    N = X.shape[0]
    if N < seq_len:
        raise ValueError(f"Data terlalu pendek: N={N}, butuh >= SEQ_LEN={seq_len}")
    return np.stack([X[i:i+seq_len] for i in range(N - seq_len + 1)], axis=0)


def main():
    print("DEVICE:", DEVICE)

    df = pd.read_csv(CSV_PATH)

    df["speed_missing"] = df["avg_speed"].isna().astype(int)

    if df["avg_speed"].notna().any():
        med = float(df["avg_speed"].median())
    else:
        med = 0.0
    df["avg_speed"] = df["avg_speed"].fillna(med)

    # ambil fitur
    X_raw = df[FEATURES].astype(float).values  # (N, F)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    X_seq = make_sequences(X, SEQ_LEN)  # (M, T, F)
    dataset = SeqDataset(X_seq)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # ---------- Model ----------
    model = LSTMAutoEncoder(
        input_dim=X_seq.shape[2],
        hidden_dim=HIDDEN,
        latent_dim=LATENT,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # ---------- Train ----------
    model.train()
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for batch in loader:
            batch = batch.to(DEVICE)
            pred = model(batch)
            loss = loss_fn(pred, batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{EPOCHS}  loss={np.mean(losses):.6f}")

    # ---------- Score ----------
    model.eval()
    with torch.no_grad():
        all_seq = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
        recon = model(all_seq).cpu().numpy()     # (M, T, F)
        orig = all_seq.cpu().numpy()

    # error per sequence per timestep
    err_seq_t = np.mean((recon - orig) ** 2, axis=2)  # (M, T)

    N = X.shape[0]
    err_t = np.zeros(N, dtype=np.float64)
    cnt_t = np.zeros(N, dtype=np.float64)

    M = err_seq_t.shape[0]
    for start in range(M):
        for t in range(SEQ_LEN):
            idx = start + t
            err_t[idx] += err_seq_t[start, t]
            cnt_t[idx] += 1

    err_t = err_t / np.maximum(cnt_t, 1)

    # ---------- Threshold ----------
    thr = np.percentile(err_t[~np.isnan(err_t)], THRESH_PCT)
    is_anom = err_t > thr

    # ---------- Save scores ----------
    out_scores = df.copy()
    out_scores["recon_error"] = err_t
    out_scores["is_anomaly"] = is_anom.astype(int)
    out_scores.to_csv(os.path.join(OUT_DIR, "anomaly_scores.csv"), index=False)

    out_anom = out_scores[out_scores["is_anomaly"] == 1][["t_start", "t_end", "flow", "density", "avg_speed", "recon_error"]]
    out_anom.to_csv(os.path.join(OUT_DIR, "anomalies.csv"), index=False)

    print(f"[OK] threshold(p{THRESH_PCT}) = {thr:.6f}")
    print(f"[OK] anomalies = {int(is_anom.sum())} / {len(is_anom)}")
    print("[OK] saved:",
          os.path.join(OUT_DIR, "anomaly_scores.csv"),
          os.path.join(OUT_DIR, "anomalies.csv"))

    # ---------- Plot ----------
    t = out_scores["t_end"].values

    plt.figure()
    plt.plot(t, out_scores["avg_speed"].values, label="avg_speed")
    # mark anomalies
    an_t = t[is_anom]
    an_v = out_scores["avg_speed"].values[is_anom]
    plt.scatter(an_t, an_v, label="anomaly", marker="x")

    plt.xlabel("time (s)")
    plt.ylabel("avg_speed")
    plt.title("Traffic Avg Speed + Anomaly Flags")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "anomaly_plot.png"), dpi=200)
    print("[OK] saved plot:", os.path.join(OUT_DIR, "anomaly_plot.png"))


if __name__ == "__main__":
    main()

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from collections import defaultdict

# ---------------- CONFIG ----------------
TRACKS_CSV = "outputs/tracks_log.csv"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# adaptive speed limit
BASELINE_WARMUP_SEC = 10.0 
PCTL = 90 # p90
MARGIN_KMH = 5.0 # limit = p90 + margin

# overspeed rule
OVERSPEED_MIN_SEC = 2.0
FPS_FALLBACK = 30.0

# sequence dataset
SEQ_LEN = 30 
STRIDE = 5

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4

# model
HIDDEN = 64
NUM_LAYERS = 1
DROPOUT = 0.1

SPEED_LIMIT_KMH = 100.0 # fixed rule
OVERSPEED_MIN_SEC = 1.5 # > 1 second

MAX_SPEED_KMH = 150
GAP_FRAMES_BREAK = 5

# prediction threshold
PRED_TH = 0.6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)
# --------------------------------------


def infer_fps(df: pd.DataFrame) -> float:
    """
    Estimasi fps dari time_s kalau ada; kalau tidak, fallback.
    """
    if "time_s" not in df.columns:
        return FPS_FALLBACK
    t = df["time_s"].values
    if len(t) < 20:
        return FPS_FALLBACK
    dt = np.diff(t)
    dt = dt[(dt > 0) & (dt < 1)]
    if len(dt) == 0:
        return FPS_FALLBACK
    fps = 1.0 / np.median(dt)
    if fps < 5 or fps > 240:
        return FPS_FALLBACK
    return float(fps)

# def build_adaptive_limit(df: pd.DataFrame, fps: float) -> float:
#     warmup_end = BASELINE_WARMUP_SEC
#     w = df[(df["time_s"] <= warmup_end) & (df["speed_kmh"].notna())]
#     speeds = w["speed_kmh"].astype(float).values
#     speeds = speeds[(speeds > 0) & (speeds < 200)]
#     if len(speeds) < 30:
#         # fallback: pakai global p90 kalau warmup kurang
#         speeds = df["speed_kmh"].dropna().astype(float).values
#         speeds = speeds[(speeds > 0) & (speeds < 200)]
#     p = float(np.percentile(speeds, PCTL)) if len(speeds) else 60.0
#     return p + MARGIN_KMH

def build_speed_limit_fixed() -> float:
    return SPEED_LIMIT_KMH


def teacher_label_per_track(track_df: pd.DataFrame, speed_limit: float, fps: float) -> pd.DataFrame:
    """
    Buat label overspeed per frame untuk satu track_id:
    overspeed jika speed > limit konsisten >= OVERSPEED_MIN_SEC.
    """
    min_frames = int(round(OVERSPEED_MIN_SEC * fps))
    s = track_df["speed_kmh"].astype(float).values
    valid = (s > 0) & (s < MAX_SPEED_KMH)

    over = valid & (s > speed_limit)

    # consecutive run length
    run = np.zeros(len(over), dtype=np.int32)
    c = 0
    for i, v in enumerate(over):
        if v:
            c += 1
        else:
            c = 0
        run[i] = c

    y = np.zeros(len(over), dtype=np.int32)
    i = 0
    while i < len(over):
        if over[i]:
            j = i
            while j < len(over) and over[j]:
                j += 1
            seg_len = j - i
            if seg_len >= min_frames:
                y[i:j] = 1
            i = j
        else:
            i += 1

    out = track_df.copy()
    out["teacher_overspeed"] = y
    out["speed_limit"] = speed_limit
    return out


def summarize_events(df_frames: pd.DataFrame, label_col: str, fps: float) -> pd.DataFrame:
    events = []
    for tid, g in df_frames.groupby("track_id"):
        g = g.sort_values("frame_idx").reset_index(drop=True)

        y = g[label_col].astype(int).values
        frames = g["frame_idx"].astype(int).values

        i = 0
        while i < len(y):
            if y[i] == 1:
                j = i + 1
                while j < len(y) and y[j] == 1:
                    # kalau tracking putus (gap frame besar), stop
                    if frames[j] - frames[j-1] > GAP_FRAMES_BREAK:
                        break
                    j += 1

                seg = g.iloc[i:j]
                events.append({
                    "track_id": int(tid),
                    "start_time": float(seg["time_s"].iloc[0]),
                    "end_time": float(seg["time_s"].iloc[-1]),
                    "duration_s": float(seg["time_s"].iloc[-1] - seg["time_s"].iloc[0] + (1.0 / fps)),
                    "max_speed": float(seg["speed_kmh"].max(skipna=True)),
                    "avg_speed": float(seg["speed_kmh"].mean(skipna=True))
                })
                i = j
            else:
                i += 1
    return pd.DataFrame(events)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, h = self.gru(x)
        h_last = h[-1]               # (B, H)
        logit = self.head(h_last).squeeze(-1)
        return logit


def make_track_sequences(df: pd.DataFrame, feature_cols, label_col, seq_len, stride):
    """
    Build sequences per track_id.
    Each sample label = majority label in the window
    biar model nangkep onset.
    """
    X_list, y_list, meta = [], [], []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame_idx").reset_index(drop=True)
        X = g[feature_cols].values.astype(np.float32)
        y = g[label_col].values.astype(np.int32)
        t = g["time_s"].values.astype(np.float32)
        fidx = g["frame_idx"].values.astype(np.int32)

        if len(g) < seq_len:
            continue

        for start in range(0, len(g) - seq_len + 1, stride):
            end = start + seq_len
            Xw = X[start:end]
            yw = y[start:end]
            # label window: 1 kalau ada overspeed di window
            y_label = 1 if yw.max() == 1 else 0

            X_list.append(Xw)
            y_list.append(y_label)
            meta.append({
                "track_id": int(tid),
                "start_frame": int(fidx[start]),
                "end_frame": int(fidx[end-1]),
                "start_time": float(t[start]),
                "end_time": float(t[end-1])
            })

    if not X_list:
        raise RuntimeError("Tidak ada sequence yang kebentuk. Coba kecilkan SEQ_LEN.")
    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int64), meta


def split_by_track_stratified(df: pd.DataFrame, label_col="teacher_overspeed", test_ratio=0.25):
    # track -> apakah pernah overspeed
    track_any = df.groupby("track_id")[label_col].max().astype(int)
    pos_ids = track_any[track_any == 1].index.to_numpy()
    neg_ids = track_any[track_any == 0].index.to_numpy()

    np.random.shuffle(pos_ids)
    np.random.shuffle(neg_ids)

    n_pos_test = max(1, int(len(pos_ids) * test_ratio)) if len(pos_ids) else 0
    n_neg_test = max(1, int(len(neg_ids) * test_ratio)) if len(neg_ids) else 0

    test_ids = set(pos_ids[:n_pos_test].tolist() + neg_ids[:n_neg_test].tolist())

    train_df = df[~df["track_id"].isin(test_ids)].copy()
    test_df  = df[df["track_id"].isin(test_ids)].copy()
    return train_df, test_df



def main():
    df = pd.read_csv(TRACKS_CSV)

    need_cols = ["frame_idx", "time_s", "track_id", "cx", "cy", "speed_kmh"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' tidak ada di CSV. Pastikan tracks_log.csv dari extractor terbaru.")

    df = df.dropna(subset=["track_id", "time_s", "frame_idx"]).copy()
    df["track_id"] = df["track_id"].astype(int)
    df["frame_idx"] = df["frame_idx"].astype(int)
    df["time_s"] = df["time_s"].astype(float)

    fps = infer_fps(df)
    speed_limit = build_speed_limit_fixed()
    print(f"[INFO] fixed speed_limit = {speed_limit:.2f} km/h, min_dur={OVERSPEED_MIN_SEC:.2f}s")


    print(f"[INFO] fps≈{fps:.2f}")
    print(f"[INFO] adaptive speed_limit = p{PCTL}+{MARGIN_KMH} = {speed_limit:.2f} km/h")

    # teacher label per track
    labeled = []
    for tid, g in df.groupby("track_id"):
        labeled.append(teacher_label_per_track(g.sort_values("frame_idx"), speed_limit, fps))
    dfL = pd.concat(labeled, ignore_index=True)

    teacher_events = summarize_events(dfL, "teacher_overspeed", fps)
    teacher_events.to_csv(OUT_DIR / "teacher_events.csv", index=False)

    dfL["speed_missing"] = dfL["speed_kmh"].isna().astype(int)

    med = float(dfL["speed_kmh"].dropna().median()) if dfL["speed_kmh"].notna().any() else 0.0
    dfL["speed_kmh_filled"] = dfL["speed_kmh"].fillna(med)

    # acceleration (delta speed)
    dfL = dfL.sort_values(["track_id", "frame_idx"]).copy()
    dfL["acc_kmh"] = dfL.groupby("track_id")["speed_kmh_filled"].diff().fillna(0.0)

    # relative speed to limit (helps)
    dfL["speed_over"] = dfL["speed_kmh_filled"] - speed_limit

    feature_cols = ["speed_kmh_filled", "acc_kmh", "speed_over", "speed_missing"]
    label_col = "teacher_overspeed"

    # scale features using train split only
    train_df, test_df = split_by_track_stratified(dfL, label_col="teacher_overspeed", test_ratio=0.25)

    scaler = StandardScaler()
    train_feats = scaler.fit_transform(train_df[feature_cols].values.astype(np.float32))
    test_feats  = scaler.transform(test_df[feature_cols].values.astype(np.float32))

    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    train_df_scaled[feature_cols] = train_feats
    test_df_scaled[feature_cols] = test_feats

    # sequences
    Xtr, ytr, meta_tr = make_track_sequences(train_df_scaled, feature_cols, label_col, SEQ_LEN, STRIDE)
    Xte, yte, meta_te = make_track_sequences(test_df_scaled, feature_cols, label_col, SEQ_LEN, STRIDE)

    # handle class imbalance (overspeed likely rare)
    pos = ytr.sum()
    neg = len(ytr) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(DEVICE)
    print(f"[INFO] train samples: {len(ytr)}  pos={int(pos)} neg={int(neg)}  pos_weight={float(pos_weight):.2f}")

    train_loader = DataLoader(SeqDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader  = DataLoader(SeqDataset(Xte, yte), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


    model = GRUClassifier(input_dim=Xtr.shape[2], hidden_dim=HIDDEN, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("[DEBUG] tracks pos/neg:",
      dfL.groupby("track_id")["teacher_overspeed"].max().value_counts().to_dict())
    # ----- train -----
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logit = model(xb)
            loss = loss_fn(logit, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:02d}/{EPOCHS} loss={np.mean(losses):.5f}")

    # ----- eval on test -----
    model.eval()
    all_probs = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logit = model(xb)
            prob = torch.sigmoid(logit).cpu().numpy()
            all_probs.append(prob)
            all_true.append(yb.numpy())

    probs = np.concatenate(all_probs)
    ytrue = np.concatenate(all_true)

    assert len(meta_te) == len(probs), f"meta_te({len(meta_te)}) != probs({len(probs)})"

    pred_windows = pd.DataFrame(meta_te)
    pred_windows["prob_overspeed"] = probs
    pred_windows["pred_overspeed"] = (probs >= PRED_TH).astype(int)
    pred_windows["teacher_window"] = yte 
    pred_windows.to_csv(OUT_DIR / "overspeed_pred_windows.csv", index=False)

    ypred = (probs >= PRED_TH).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(ytrue, ypred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(ytrue, probs) if len(np.unique(ytrue)) > 1 else np.nan
    except Exception:
        auc = np.nan
    print(f"[EVAL vs teacher] Precision={p:.3f} Recall={r:.3f} F1={f1:.3f} AUC={auc:.3f}")
    print("[DEBUG] window positives: train=", int(ytr.sum()), "test=", int(yte.sum()))


    # ----- export per-window predictions for manual verification -----
    pred_windows["pred_overspeed"] = ypred
    pred_windows["teacher_window"] = ytrue
    pred_windows.to_csv(OUT_DIR / "overspeed_pred_windows.csv", index=False)

    # ----- convert window predictions into frame-level probability (spread window prob to frames) -----
    # We'll produce per-frame prob by max over overlapping windows for a track.
    df_test = test_df_scaled[["track_id", "frame_idx", "time_s", "speed_kmh", "teacher_overspeed"]].copy()
    df_test["model_prob"] = 0.0

    prob_map = defaultdict(float)
    for row in pred_windows.itertuples(index=False):
        tid = int(row.track_id)
        sf = int(row.start_frame)
        ef = int(row.end_frame)
        pr = float(row.prob_overspeed)
        for fi in range(sf, ef + 1):
            key = (tid, fi)
            if pr > prob_map[key]:
                prob_map[key] = pr

    # apply map
    df_test["model_prob"] = [
        prob_map.get((int(tid), int(fi)), 0.0)
        for tid, fi in zip(df_test["track_id"].values, df_test["frame_idx"].values)
    ]
    df_test["model_pred"] = (df_test["model_prob"] >= PRED_TH).astype(int)

    df_test.to_csv(OUT_DIR / "overspeed_pred_frames.csv", index=False)

    # events from model_pred and teacher
    model_events = summarize_events(df_test.rename(columns={"model_pred": "model_overspeed"}),
                                   "model_overspeed", fps)
    model_events.to_csv(OUT_DIR / "overspeed_pred_events.csv", index=False)

    # save teacher events for test split only (for comparison)
    teacher_events_test = summarize_events(df_test.rename(columns={"teacher_overspeed": "teacher_overspeed"}),
                                          "teacher_overspeed", fps)
    teacher_events_test.to_csv(OUT_DIR / "teacher_events_testsplit.csv", index=False)

    # save config summary
    with open(OUT_DIR / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"fps≈{fps:.3f}\n")
        f.write(f"adaptive_speed_limit={speed_limit:.3f} (p{PCTL}+{MARGIN_KMH})\n")
        f.write(f"SEQ_LEN={SEQ_LEN}, STRIDE={STRIDE}, PRED_TH={PRED_TH}\n")
        f.write(f"Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}, AUC={auc:.4f}\n")

    print("[OK] saved:")
    print(" -", OUT_DIR / "overspeed_pred_windows.csv")
    print(" -", OUT_DIR / "overspeed_pred_frames.csv")
    print(" -", OUT_DIR / "overspeed_pred_events.csv")
    print(" -", OUT_DIR / "teacher_events.csv")
    print(" -", OUT_DIR / "run_summary.txt")


if __name__ == "__main__":
    main()
